import collections

import os

import shutil

from IPython.display import display

from tqdm import tqdm_notebook as tqdm



import numpy as np

from PIL import Image

import torch

import torchvision
# version

display(torch.__version__)

display(torch.version.cuda)

display(torch.backends.cudnn.version())

# display(torch.cuda.get_device_name(0))
# fix the random seed

SEED = 620402

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)
torch.cuda.is_available()
# set cudnn benchmark

torch.backends.cudnn.benchmark = True

# avoid some wave jitting in the results, we can set

torch.backends.cudnn.deterministic = True
# release some memory in the GPU memory

torch.cuda.empty_cache()
# or use the command line to reset gpu

#!nvidia-smi --gpu-reset -i [gpu_id]
# basic information in tensor

# tensor.type()

# tensor.size()

# tensor.dim()



x = torch.randn(3, 3)

display(x)



display(x.type(), x.size(), x.dim())
# set default tensor type, 

torch.set_default_tensor_type(torch.FloatTensor)



# x = x.cuda()

display(x.type())

x = x.cpu()

display(x.type())

x = x.long()

display(x.type())

x = x.float()

display(x.type())



# a common way for defining the tensor (cpu or gpu)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = x.to(device)

display(x.type())
# translation between torch.Tensor and np.ndarray



# torch.Tensor --> np.ndarray

ndarray = x.detach().cpu().numpy()

display(ndarray.dtype, ndarray.shape)



# np.ndarray --> torch.Tensor

tensor = torch.from_numpy(ndarray).float()

display(tensor.type())

# if ndarray has negative stride

tensor = torch.from_numpy(ndarray.copy()).float()

display(tensor.type())
# torch.Tensor and PIL.Image

# torch。Tensor --> PIL.Image

x = torch.rand(3, 128, 128)

image = Image.fromarray(torch.clamp(x * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())

display(image)

# using torchvision will be more convinient 

image = torchvision.transforms.functional.to_pil_image(x)

display(image)

image.save('hello.jpg')



# PIL.Image --> torch.Tensor

!ls

image_tensor = torch.from_numpy(np.asarray(Image.open('hello.jpg'))).permute(2, 0, 1).float()/255.0

display(image_tensor.size())

image_tensor = torchvision.transforms.functional.to_tensor(Image.open('hello.jpg'))

display(image_tensor.size())
# np.ndarray and PIL.Image

# np.ndarray --> PIL.Image

ndarray = np.random.rand(128, 128, 3) * 255.0



image = Image.fromarray(ndarray.astype(np.uint8))

display(image)



# PIL.Image --> np.ndarray

ndarray = np.asarray(image)

display(ndarray.shape)
# get the value of a tensor with one element

x = torch.rand(1)

display(x)

x_value = x.item()

display(x_value)
# reshape tensor, which will deal with the tensor contiguously.

tensor_x = torch.rand(64, 256, 16, 16)

fc_input = tensor_x.reshape(64, -1)

display(fc_input.size())
# shuffle the data

tensor_x = torch.rand(64, 128, 16, 16)

display(tensor_x.size())

tensor_x_random = tensor_x[torch.randperm(tensor_x.size(0))]

display(tensor_x_random.size())
tensor = tensor_x[:, :, :, torch.arange(tensor_x.size(3)-1, -1, -1).long()]

display(tensor.size())
# copy tensor

# tensor.clone()

# tensor.detach()

# tensor.detach.clone()
# concat and stack tensor

tensor_list = [torch.rand(3, 224, 224) for i in range(16)]

display(len(tensor_list))



# concat

concat_tensor = torch.cat(tensor_list, dim=0)

display(concat_tensor.size())

stack_tensor = torch.stack(tensor_list, dim=0)

display(stack_tensor.size())
# one-hot label, which will be used in the last layer of a model.

num_classes = 10

# The tensor below is the label

tensor = torch.randint(0, num_classes, size=(64,1))

# display(tensor)

N = tensor.size(0)





one_hot = torch.zeros(N, num_classes).long()

display(one_hot)

one_hot.scatter_(1, tensor, 1)

display(one_hot)
# Get the nonzero or zero elements.

tensor = torch.randint(0, 2, size=(6, 1))

display(tensor)



display(torch.nonzero(tensor))

display(torch.nonzero(tensor==0))

display(torch.nonzero(tensor).size(0))

display(torch.nonzero(tensor==0).size(0))
tensor_size = (8, 2)

tensor1 = torch.ones(tensor_size)

tensor2 = torch.ones(tensor_size)

tensor3 = torch.zeros(tensor_size)



# tensors equal

# float tensor

display(torch.allclose(tensor1, tensor2), torch.allclose(tensor1, tensor3))

# long tensor

tensor1 = tensor1.long()

tensor2 = tensor2.long()

tensor3 = tensor3.long()

display(torch.equal(tensor1, tensor2), torch.equal(tensor1, tensor3))
# expand tensor

size = (64, 512)

tensor = torch.randn(size)

reshape_tensor = torch.reshape(tensor, size+(1, 1))

display(reshape_tensor.size())

expand_tensor = reshape_tensor.expand(size + (7, 7))

display(expand_tensor.size())
# matrix multiplication

# a matrix A with dimension (m, n) multiply the matrix B with dimension (n, p)

# (m, n) * (n, p) -> (m, p)

M = 3

N = 4

P = 5

A = torch.randn(M, N)

B = torch.randn(N, P)

C = torch.mm(A, B)

display(C.size())



# Batch matrix multiplication

batch_size = 64

A = torch.randn(batch_size, M, N)

B = torch.randn(batch_size, N, P)

C = torch.bmm(A, B)

display(C.size())



# element-wise multiplication

A = torch.randn(M, N)

B = torch.randn(M, N)

C = A * B

display(C.size())
# calculate the distance of A and B

m = 3

n = 4

d = 5

x = torch.randn(m, d)

y = torch.randn(n, d)



# the result may have m * n elements. The code can be used in VQ-VAE

sum_inter = torch.sum((x[:, None, :] - y)**2, dim=2)

display(sum_inter.size())

dist = torch.sqrt(sum_inter)

display(dist.size())
# get features from models with pretrained by imagenet

vgg16 = torchvision.models.vgg16(pretrained=True)

display(vgg16)

display('*******************************')

display(vgg16.features[:-1])

display('*******************************')

display(vgg16.features)

display('*******************************')

display(vgg16.classifier)
vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])

display(vgg16.classifier)
resnet18 = torchvision.models.resnet18(pretrained=True)

display(resnet18)

display('**********************************')

display(resnet18.layer1)

display('**********************************')

display(resnet18.fc)
list(resnet18.named_children())



resnet18_features = torch.nn.Sequential(collections.OrderedDict(

    list(resnet18.named_children())[:-1]

))

resnet18_features



image = torch.rand(32, 3, 224, 224)

with torch.no_grad():

    resnet18_features.eval()

    output_features = resnet18_features(image)

display(output_features.size())
list(resnet18.named_children())
%%time

# use some layer features

class feature_extractor(torch.nn.Module):

    def __init__(self, pretrained_model, layers_to_extractor):

        super(feature_extractor, self).__init__()

        self._model = pretrained_model

        self._model.eval()

        self._layers = layers_to_extractor

    

    def forward(self, x):

        with torch.no_grad():

            conv_representation = []

            for name, layer in self._model.named_children():

                x = layer(x)

                if name in self._layers:

                    conv_representation.append(x)

            return conv_representation

        

# test the function of feature_extractor

resnet152 = torchvision.models.resnet152(pretrained=True)

resnet152_features = torch.nn.Sequential(collections.OrderedDict(list(resnet152.named_children())[:-1]))

resnet152_features

layers = ['layer1', 'layer2', 'layer3', 'layer4']

feature_extractor_resnet152 = feature_extractor(pretrained_model=resnet152_features, layers_to_extractor=layers)



image = torch.rand(32, 3, 224, 224)

conv_representation = feature_extractor_resnet152(image)

display(len(conv_representation))

for i in range(len(conv_representation)):

    display(conv_representation[i].size())
# fine-tuning the network with the last fc layer

resnet18 = torchvision.models.resnet18(pretrained=True)



for param in resnet18.parameters():

    param.required_grad = False

resnet18.fc = torch.nn.Linear(512, 100)

optimizer = torch.optim.SGD(resnet18.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
# fine-tuning the network with the low lr of conv layer and high lr of fc layer.

resnet18 = torchvision.models.resnet18(pretrained=True)

finetuned_parameters = list(map(id, resnet18.fc.parameters()))

display(finetuned_parameters)

conv_parameters = (p for p in resnet18.parameters() if id(p) not in finetuned_parameters)



parameters = [{'params':conv_parameters, 'lr':1e-3},

              {'params':resnet18.fc.parameters()}]



optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
# the most commom conv layer definition

in_feature = 3

out_feature = 16

conv3 = torch.nn.Conv2d(in_feature, out_feature, 3, 1, 1, bias=True)

conv1 = torch.nn.Conv2d(in_feature, out_feature, 1, 1, 0, bias=True)



image = torch.rand(32, 3, 224, 224)

out_conv3 = conv3(image)

out_conv1 = conv1(image)



display(out_conv3.size(), out_conv1.size())
# Global average pooling

gap = torch.nn.AdaptiveAvgPool2d(output_size = 1)

out_gap = gap(image)

display(out_gap.size())
# bilinear pooling

N = 32

D = 64

H = 112

W = 112



X = torch.rand(N, D, H, W)

X = X.reshape(N, D, H*W)

X = torch.bmm(X, torch.transpose(X, 1, 2)) / (H * W)

display(X.size())

X = X.reshape(N, D*D)

X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)

X = torch.nn.functional.normalize(X)

display(X.size())
resnet18 = torchvision.models.resnet18(pretrained=True)

num_parameters = sum(torch.numel(parameter) for parameter in resnet18.parameters())

display(num_parameters)
# Note: model.modules() is different from model.children()

# common practise for initialization

for layer in resnet18.modules():

    if isinstance(layer, torch.nn.Conv2d):

        torch.nn.init.kaiming_normal_(layer.weight,  mode='fan_out', nonlinearity='relu')

        if layer.bias is not None:

            torch.nn.init.constatn_(layer.bias, val=0.0)

            

    if isinstance(layer, torch.nn.BatchNorm2d):

        torch.nn.init.constant_(layer.weight, val=1.0)

        torch.nn.init.constant_(layer.bias, val=0.0)

    

    if isinstance(layer, torch.nn.Linear):

        torch.nn.init.xavier_normal_(layer.weight)

        if layer.bias is not None:

            torch.nn.init.constant_(layer.bias, val=0.0)
# label smoothing

N = 32

C = 10

labels = torch.randint(0, 10, size=(32, 1))

smooth_labels = torch.full(size=(N, C), fill_value=0.1/(C-1))

display(smooth_labels)

smooth_labels.scatter_(1, labels, value=0.9)

display('*******************************')

display(smooth_labels)