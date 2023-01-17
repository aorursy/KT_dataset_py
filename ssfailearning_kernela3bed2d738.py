# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install git+https://github.com/qubvel/segmentation_models.pytorch
!pip install cython
!pip install git+https://github.com/lucasb-eyer/pydensecrf.git
!pip install geffnet
import cv2
import matplotlib.pyplot as plt
import geffnet
from fastai.vision import *
import torch.nn as nn
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image as I
import segmentation_models_pytorch as smp
from collections import OrderedDict
%matplotlib inline
cam_names=get_image_files('/kaggle/input/new-hulianwang/cam_result/cam_result')
plt_names=[]
for name in cam_names:
    name=str(name)
    name=name.split('/')[-1]
    plt_names.append(name)

    
name=plt_names[5]
cam_path=os.path.join('/kaggle/input/new-hulianwang/cam_result/cam_result',name)
image=cv2.imread(cam_path,0)
print(image)
plt.imshow(image,cmap='gray')
# transform1=A.Compose([
#     A.Resize(640, 640),
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
transform_valid=transforms.Compose([
    transforms.Resize([300,300]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
origin_transform=transforms.Compose([
    transforms.Resize([300,300])
])
class myDataset(Data.Dataset):
    def __init__(self):
        super(myDataset, self).__init__()
        self.path='/kaggle/input/new-hulianwang/new_data/unlabel'
        labels=['sofa','bed']
        label_dict={'sofa':0,'bed':1}
        self.images=[]
        self.labels=[]
        self.weak_labels=[]
        self.names=[]
        for label in labels:
            images_path=os.path.join(self.path,label)
            for name in get_image_files(images_path):
                name = str(name)
                if name.endswith('gif'):
                    continue
                self.names.append(name)
                self.labels.append(label_dict[label])

#                 image=cv2.imread(name)

#                 image=I.fromarray(np.uint8(image))
#                 image=transform_valid(image)

#                 self.images.append(image)

    def __getitem__(self, index):
        name=self.names[index]
        image=cv2.imread(name)
 
        image=I.fromarray(np.uint8(image))
    
        origin_image=image
        
        origin_image=origin_transform(origin_image)
        origin_image=np.array(origin_image)
        origin_image=torch.from_numpy(origin_image.transpose((2,0,1)))
        image=transform_valid(image)
        label=self.labels[index]
        
       
        return origin_image,image,label,name

    def __len__(self):
        return len(self.names)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Residual(nn.Module):
    def __init__(self, in_channel, R=8, k=2):
        super(Residual, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.R = R
        self.k = k
        out_channel = int(in_channel / R)
        self.fc1 = nn.Linear(in_channel, out_channel)
        fc_list = []
        for i in range(k):
            fc_list.append(nn.Linear(out_channel, 2 * in_channel))
        self.fc2 = nn.ModuleList(fc_list)

    def forward(self, x):
        x = self.avg(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.relu(x)
        result_list = []
        for i in range(self.k):
            result = self.fc2[i](x)
            result = 2 * torch.sigmoid(result) - 1
            result_list.append(result)
        return result_list


class Dynamic_relu_b(nn.Module):
    def __init__(self, inchannel, R=8, k=2):
        super(Dynamic_relu_b, self).__init__()
        self.lambda_alpha = 1
        self.lambda_beta = 0.5
        self.R = R
        self.k = k
        self.init_alpha = torch.zeros(self.k)
        self.init_beta = torch.zeros(self.k)
        self.init_alpha[0] = 1
        self.init_beta[0] = 1
        for i in range(1, k):
            self.init_alpha[i] = 0
            self.init_beta[i] = 0

        self.residual = Residual(inchannel)

    def forward(self, input):
        delta = self.residual(input)
        in_channel = input.shape[1]
        bs = input.shape[0]
        alpha = torch.zeros((self.k, bs, in_channel),device=input.device)
        beta = torch.zeros((self.k, bs, in_channel),device=input.device)
        for i in range(self.k):
            for j, c in enumerate(range(0, in_channel * 2, 2)):
                alpha[i, :, j] = delta[i][:, c]
                beta[i, :, j] = delta[i][:, c + 1]
        alpha1 = alpha[0]
        beta1 = beta[0]
        max_result = self.dynamic_function(alpha1, beta1, input, 0)
        for i in range(1, self.k):
            alphai = alpha[i]
            betai = beta[i]
            result = self.dynamic_function(alphai, betai, input, i)
            max_result = torch.max(max_result, result)
        return max_result
    def dynamic_function(self, alpha, beta, x, k):
        init_alpha = self.init_alpha[k]
        init_alpha=init_alpha.to(x.device)
        init_beta = self.init_beta[k]
        init_beta=init_beta.to(x.device)
        # lambda_alpha=self.lambda_alpha.to(x.device)
        # lambda_beta=self.lambda_beta.to(x.device)
        alpha = init_alpha +  self.lambda_alpha* alpha
        beta = init_beta + self.lambda_beta * beta
        bs = x.shape[0]
        channel = x.shape[1]
        results = torch.zeros_like(x,device=x.device)
        results = x * alpha.view(bs, channel, 1, 1) + beta.view(bs, channel, 1, 1)
        return results
class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        backbone=geffnet.efficientnet_b3(pretrained=True)
        act1=Dynamic_relu_b(40)
        act2=Dynamic_relu_b(1536)
        self.backbone = torch.nn.Sequential(OrderedDict([
            ('conv_stem',backbone.conv_stem),
            ('bn1',backbone.bn1),
            ('act1',act1),
            ('block',backbone.blocks),
            ('conv_head',backbone.conv_head),
            ('bn2',backbone.bn2),
            ('act2',act2)
        ]))
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.embedding_fc = nn.Linear(1536, 1024)
        self.fc = nn.Linear(1024, 2)
        self.local_conv = nn.Conv2d(1536, 512, 1)
        self.local_bn = nn.BatchNorm2d(512)
        self.local_bn.bias.requires_grad = False
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.fc.bias, 0)
        nn.init.kaiming_normal_(self.embedding_fc.weight, mode='fan_out')
        nn.init.constant_(self.embedding_fc.bias, 0)

    def forward(self,x):
    
        x=self.backbone(x)
        M=torch.mean(x,dim=1).unsqueeze(1)
#         M=F.upsample(M, size=(224, 224), mode='bilinear', align_corners=False)
        local_feat=x
        
        x=self.avg_pool(x)
        x=x.view(x.shape[0],-1)
         
        embedding=self.embedding_fc(x)
        embedding=nn.functional.normalize(embedding,p=2,dim=1)

        out=self.fc(embedding)



        local_feat = torch.mean(local_feat, -1, keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)
        if self.training:
            return torch.sigmoid(out),embedding,local_feat
        else:
            return torch.sigmoid(out),M


class GradCAM(object):
    def __init__(self,model):
        self.model=model
        target_layer=self.model.backbone.act2
        self.gradients=dict()
        self.activations=dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self,x,y):
        b,c,h,w=x.shape
        logit=self.model(x)
        score=logit[:,y].squeeze()
        print(score)
        self.model.zero_grad()
        score.backward(retain_graph=False)
        gradients=self.gradients['value']
        activations=self.activations['value']
        b,k,u,v=gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        return saliency_map,logit

    def __call__(self, x,y):
        return self.forward(x,y)

class GradCAMpp(GradCAM):
    def __init__(self,model):
        super(GradCAMpp, self).__init__(model)

    def forward(self,x,y):
       
        b, c, h, w = x.shape
        logit = self.model(x)
        print(logit)

        score = logit[:,y].squeeze()

        self.model.zero_grad()

        score.backward()
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit


import pydensecrf.densecrf as dcrf
def dense_crf(img, output_probs): #img为输入的图像，output_probs是经过网络预测后得到的结果
    h = output_probs.shape[0] #高度
    w = output_probs.shape[1] #宽度

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2) #NLABELS=2两类标注，车和不是车
    U = -np.log(output_probs) #得到一元势

    U = U.reshape((2, -1)) #NLABELS=2两类标注
    U = np.ascontiguousarray(U) #返回一个地址连续的数组

    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U) #设置一元势

    d.addPairwiseGaussian(sxy=20, compat=3) #设置二元势中高斯情况的值
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)#设置二元势众双边情况的值

    Q = d.inference(5) #迭代5次推理
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w)) #得列中最大值的索引结果

    return Q
checkpoint=torch.load('/kaggle/input/houssad-checkpoint/base_step.pt')['state']
checkpoint

class CamGrad:
    def __init__(self,model, return_one_always=True):
        self.model = model
        self.model.load_state_dict(
            new_dict
        )
       
      
        self.model.eval()
        self.prediction_threshold = 0.5
        self.set_target_layer_hook('backbone_bn2')
        self.return_one_always = return_one_always

        self.activation_maps = None
        self.grad = None

        self.func_on_target = lambda x: x

    def set_target_layer_hook(self, target_layer):
        modules_path = target_layer.split(".")
        module = self.model
        for subpath in modules_path:
            for name, current_module in module.named_children():
                if name == subpath:
                    module = current_module
                    break
            else:
                raise ValueError(
                    f"Module path {target_layer} is not valid for current module."
                )

        module.register_forward_hook(self.save_output)
        module.register_backward_hook(self.save_grad)

    def save_output(self, module, input_tensor, output_tensor):
        """Forward hook that saves output of target layer"""
        self.activation_maps = output_tensor.squeeze().detach()

    def save_grad(self, module, input_grad, output_grad):
        """Backward hook that saves gradients of output of target layer"""
        self.grad = output_grad[0].squeeze().detach()

    def forward(self, x, y=None):
        """
        Args:
            x: input tensor of size [1, C, H, W]
            y: target of shape [1, N] or None
        Returns:
            class_maps: list of maps corresponding to different classes
            y: if input was None returns predicted classes else y from input
        """
        assert x.size(0) == 1

        logits_pred = self.model(x).squeeze(0)
        if y is None:
            y = torch.where(torch.sigmoid(logits_pred) > self.prediction_threshold)[0]
            if y.shape[0] == 0 and self.return_one_always:
                y = torch.argmax(logits_pred, dim=0, keepdim=True)
        else:
            y = torch.where(y)[1]

        class_maps = []
        for label in y:
            self.model.zero_grad()
            self.func_on_target(logits_pred[label]).backward(retain_graph=True)

            weights = self.get_maps_weights()  # [K, ]

            class_map = torch.tensordot(
                weights, self.activation_maps, dims=((0,), (0,))
            )
            class_maps.append(class_map.detach())

        return torch.stack(class_maps), y
    
    @torch.no_grad()
    def get_maps_weights(self):
        return torch.mean(self.grad, dim=(1, 2))
length=len(checkpoint)
model=myNet()
new_key=list(model.state_dict().keys())
new_dict=model.state_dict()
old_key=list(checkpoint.keys())
old_key
for i in range(length):
    new_dict[new_key[i]]=checkpoint[old_key[i]]

def evaluate(model,valid_dl):
    steps=len(valid_dl)
    device=torch.device('cuda:0')
    step_loss=0
    step_metric=0
    with torch.no_grad():
        for _,images,labels,_ in valid_dl:
            images=images.to(device).float()
            cls_score,_=model(images)
            cls_score=cls_score.to('cpu')
            metric=accuracy(cls_score,labels)
            step_metric+=metric
    metric=step_metric/steps
    valid_loss=step_loss/steps
    print('metric:{}'.format(metric))
    return metric
dataset=myDataset()
valid_dl=Data.DataLoader(dataset,batch_size=4)
origin_image,image,label,name=dataset[1600]
model=myNet()
device=torch.device('cuda:0')
model=model.to(device)
model.load_state_dict(new_dict)
model.eval()
# evaluate(model,valid_dl)

image=torch.stack([image,image],dim=0)
print(image.shape)
image=image.cuda()
logit,M=model(image)


M=M[0].squeeze().detach().cpu().numpy()
M=M>0
plt.imshow(M,cmap='gray')

print(M)
origin_image=origin_image.numpy().transpose(1,2,0)
plt.imshow(origin_image)
plt_image=image.numpy().transpose((1,2,0))
plt.imshow(plt_image)
image=image.unsqueeze(0)
print(image.shape)

model_gram=CamGrad(model)
image=image.to(device)

print(label)
cam=model_gram(image,label)[0]
print(cam.shape)
heatmap = cv2.applyColorMap(np.uint8(255 * cam.squeeze().detach().cpu()), cv2.COLORMAP_JET)
heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
b, g, r = heatmap.split(1)
heatmap = torch.cat([r, g, b])
print(heatmap.shape)
heatmap=heatmap.numpy().transpose((1,2,0))
print(heatmap.shape)
plt.imshow(heatmap)
cam=cam.squeeze().detach().cpu().numpy()
print(cam.shape)

plt.imshow(cam,cmap='gray')
origin_image=origin_image.astype(np.uint8)
plt.imshow(origin_image)
crf=dense_crf(origin_image,cam)
print(crf.shape)
plt.imshow(crf,cmap='gray')

