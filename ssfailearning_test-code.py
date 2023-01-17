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
!pip install git+https://github.com/pabloppp/pytorch-tools -U
!pip install git+https://github.com/qubvel/segmentation_models.pytorch

import os
import albumentations as A
import numpy as np
import torch
import cv2
import torch.utils.data as Data
from fastai.vision import *
train_labeled_path='/kaggle/input/new-hulianwang/new_data/train_data'
label_dict={'sofa':1,'bed':0}
labels=['sofa','bed']
unlabel_path='/kaggle/input/new-hulianwang/new_data/unlabel'
test_label_path='/kaggle/input/new-hulianwang/new_data/val_data'
path_dict={
    'train':train_labeled_path,
    'unlabel':unlabel_path,
    'test':test_label_path
}

train_transform=A.Compose([
    A.Resize(640,640),
    A.Flip(),
    A.ShiftScaleRotate(scale_limit=0),
    A.Transpose(),

    A.OneOf(
        [
            A.IAASharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.6,
    ),

    A.OneOf(
        [
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ],
        p=0.6,
    ),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
    ], p=0.6),
    A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

])
valid_transform=A.Compose([
    A.Resize(640,640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

unlabel_transform=A.Compose([
    A.Resize(640,640),
    A.Flip(),
    A.ShiftScaleRotate(scale_limit=0),
    A.OneOf(
        [
            A.IAASharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.6,
    ),

    A.OneOf(
        [
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ],
        p=0.6,
    ),
    A.CoarseDropout(),
    A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

transform_list={
    'train':train_transform,
    'test':valid_transform,
    'unlabel':unlabel_transform
}
class myDataset(Data.Dataset):
    def __init__(self,path,transform,use_weak_label=False):
        self.path=path
        self.transform=transform
        self.image_paths=[]
        self.labels=[]
        self.masks=[]
        self.use_weak_label=use_weak_label
        self.weak_labels=[]
        for label in labels:
            images_path=os.path.join(self.path,label)
            masks_path=os.path.join(self.path,label,'masks')
            if use_weak_label:
                for name in get_image_files(images_path):
                    name = str(name)
                    name = name.split('/')[-1]
                    self.labels.append(label)
                    path=os.path.join(images_path, name)
                    self.image_paths.append(path)
#                     image_path = os.path.join(images_path, name)
                 
#                     image = cv2.imread(image_path)
#                     image = self.transform(image=image)['image']
#                     image = image / 255
#                     self.images.append(image)
#                     weak_mask=np.zeros((640,640,2))
#                     self.weak_labels.append(weak_mask)
            else:
                for name in get_image_files(os.path.join(images_path,'images')):
                    name = str(name)
                    name = name.split('/')[-1]
                    self.labels.append(label)
                    image_path = os.path.join(images_path,'images', name)
                    self.image_paths.append(image_path)
#                     image = cv2.imread(image_path)
               
#                     mask = np.zeros((640,640,2))
              
#                     augmented = self.transform(image=image, mask=mask)
#                     image, mask = augmented['image'], augmented['mask']
#                     image = image / 255
#                     mask[mask > 1] = 1
#                     self.masks.append(mask)
#                     self.images.append(image)
         
    def __getitem__(self, index):
        label=self.labels[index]
        path=self.image_paths[index]
        image=cv2.imread(path)
        
        if self.use_weak_label:
                    weak_label=np.zeros((640,640,2))
                    aug= self.transform(image=image,mask=weak_label)
                    image=aug['image']
                    mask=aug['mask']
                    image = image / 255
                    image=torch.from_numpy(image.transpose((2,0,1)))
                  
           
        else: 
            mask=np.zeros((640,640,2))
            aug=self.transform(image=image,mask=mask)
            image=aug['image']
            mask=aug['mask']
            image = image / 255
            image=torch.from_numpy(image.transpose((2,0,1)))
            
        return image,mask,label
            

    def __len__(self):
        return len(self.image_paths)





def create_dataloader(type,batch_size,shuffle=False):
    path=path_dict[type]

    use_weak_label=False
    if type=='unlabel':
        use_weak_label=True
    transform=transform_list[type]
    dataset=myDataset(path,transform,use_weak_label)
    dataloader=Data.DataLoader(dataset,batch_size,shuffle=shuffle)
    return dataloader






import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


def batch_pix_accuracy(output, target):
    _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union(output, target, num_class):
    _, predict = torch.max(output, 1)
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


def eval_metrics(output, target, num_classes, ignore_index=255):
    target = target.clone()
    target[target == ignore_index] = -1
    correct, labeled = batch_pix_accuracy(output.data, target)
    inter, union = batch_intersection_union(output.data, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(output, target):
    output = np.asarray(output)
    target = np.asarray(target)
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((output == target) * (target > 0))
    return pixel_correct, pixel_labeled


def inter_over_union(output, target, num_class):
    output = np.asarray(output) + 1
    target = np.asarray(target) + 1
    output = output * (target > 0)

    intersection = output * (output == target)
    area_inter, _ = np.histogram(intersection, bins=num_class, range=(1, num_class))
    area_pred, _ = np.histogram(output, bins=num_class, range=(1, num_class))
    area_lab, _ = np.histogram(target, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    return current / rampup_length


def cosine_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return 1 - float(.5 * (np.cos(np.pi * current / rampup_length) + 1))


def log_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return float(1 - np.exp(-5.0 * current / rampup_length))


def exp_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return float(np.exp(5.0 * (current / rampup_length - 1)))
ram_fun={
    'sigmoid_rampup':sigmoid_rampup,
    'linear_rampup':linear_rampup,
    'cosine_rampup':cosine_rampup,
    'log_rampup':log_rampup,
    'exp_rampup':exp_rampup

}


class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func=ram_fun[ramp_type]
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


def CE_loss(input_logits, target_targets, ignore_index=255, temperature=1):
    return F.cross_entropy(input_logits / temperature, target_targets, ignore_index=ignore_index)


class abCE_loss(nn.Module):
    """
    Annealed-Bootstrapped cross-entropy loss
    """

    def __init__(self, iters_per_epoch, epochs, num_classes, weight=None,
                 reduction='mean', thresh=0.7, min_kept=1, ramp_type='log_rampup'):
        super(abCE_loss, self).__init__()
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.thresh = thresh
        self.min_kept = min_kept
        self.ramp_type = ramp_type

        if ramp_type is not None:
            self.rampup_func =ram_fun[ramp_type]
            self.iters_per_epoch = iters_per_epoch
            self.num_classes = num_classes
            self.start = 1 / num_classes
            self.end = 0.9
            self.total_num_iters = (epochs - (0.6 * epochs)) * iters_per_epoch

    def threshold(self, curr_iter, epoch):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        current_rampup = self.rampup_func(cur_total_iter, self.total_num_iters)
        return current_rampup * (self.end - self.start) + self.start

    def forward(self, predict, target,  curr_iter, epoch):
        batch_kept = self.min_kept * target.size(0)
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == 255] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, )!=255
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()

        if self.ramp_type is not None:
            thresh = self.threshold(curr_iter=curr_iter, epoch=epoch)
        else:
            thresh = self.thresh

        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, thresh)
        loss_matrix = F.cross_entropy(predict, target,
                                      weight=self.weight.to(predict.device) if self.weight is not None else None,
                                      ignore_index=255, reduction='none')
        loss_matirx = loss_matrix.contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean')


def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    input_log_softmax = F.log_softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input_log_softmax, targets, reduction='mean')


def softmax_js_loss(inputs, targets, **_):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    epsilon = 1e-5

    M = (F.softmax(inputs, dim=1) + targets) * 0.5
    kl1 = F.kl_div(F.log_softmax(inputs, dim=1), M, reduction='mean')
    kl2 = F.kl_div(torch.log(targets + epsilon), M, reduction='mean')
    return (kl1 + kl2) * 0.5


def pair_wise_loss(unsup_outputs, size_average=True, nbr_of_pairs=8):
    """
    Pair-wise loss in the sup. mat.
    """
    if isinstance(unsup_outputs, list):
        unsup_outputs = torch.stack(unsup_outputs)

    # Only for a subset of the aux outputs to reduce computation and memory
    unsup_outputs = unsup_outputs[torch.randperm(unsup_outputs.size(0))]
    unsup_outputs = unsup_outputs[:nbr_of_pairs]

    temp = torch.zeros_like(unsup_outputs)  # For grad purposes
    for i, u in enumerate(unsup_outputs):
        temp[i] = F.softmax(u, dim=1)
    mean_prediction = temp.mean(0).unsqueeze(0)  # Mean over the auxiliary outputs
    pw_loss = ((temp - mean_prediction) ** 2).mean(0)  # Variance
    pw_loss = pw_loss.sum(1)  # Sum over classes
    if size_average:
        return pw_loss.mean()
    return pw_loss.sum()
import segmentation_models_pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
from itertools import chain
import cv2
import math,time
from torch.distributions.uniform import Uniform
import logging

encoder_checkpoint=''

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = int(sum([np.prod(p.size()) for p in model_parameters]))
        return f'\nNbr of trainable parameters: {nbr_params}'


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class Encoder(nn.Module):
    def __init__(self,pretrained=False):
        super(Encoder, self).__init__()
        model=segmentation_models_pytorch.encoders.get_encoder('efficientnet-b3',weights='imagenet')
        if pretrained:
            checkpoint=torch.load(encoder_checkpoint)
            model.load_state_dict(checkpoint['state'])
        self.base=model
        self.psp=_PSPModule(384,bin_sizes=[1,2,3,6])

    def forward(self,x):
        x=self.base(x)
        x=self.psp(x)
        return x

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()



def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle(nn.Module):
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels * (scale ** 2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return x


def upsample(in_channels, out_channels, upscale, kernel_size=3):
    # A series of x 2 upsamling until we get to the upscale we want
    layers = []
    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)
    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)


class MainDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(MainDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.upsample(x)
        return x


class DropOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _):
        x = self.upsample(self.dropout(x))
        return x


class FeatureDropDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(FeatureDropDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, _):
        x = self.feature_dropout(x)
        x = self.upsample(x)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x, _):
        x = self.feature_based_noise(x)
        x = self.upsample(x)
        return x


def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = decoder(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv


class VATDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _):
        r_adv = get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
        x = self.upsample(x + r_adv)
        return x


def guided_cutout(output, upscale, resize, erase=0.4, use_dropout=False):
    if len(output.shape) == 3:
        masks = (output > 0).float()
    else:
        masks = (output.argmax(1) > 0).float()

    if use_dropout:
        p_drop = random.randint(3, 6) / 10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        try:  # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:  # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w - min_w, max_h - min_h
            rnd_start_w = random.randint(0, int(bb_w * (1 - erase)))
            rnd_start_h = random.randint(0, int(bb_h * (1 - erase)))
            h_start, h_end = min_h + rnd_start_h, min_h + rnd_start_h + int(bb_h * erase)
            w_start, w_end = min_w + rnd_start_w, min_w + rnd_start_w + int(bb_w * erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)


class CutOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        maskcut = guided_cutout(pred, upscale=self.upscale, erase=self.erase, resize=(x.size(2), x.size(3)))
        x = x * maskcut
        x = self.upsample(x)
        return x


def guided_masking(x, output, upscale, resize, return_msk_context=True):
    if len(output.shape) == 3:
        masks_context = (output > 0).float().unsqueeze(1)
    else:
        masks_context = (output.argmax(1) > 0).float().unsqueeze(1)

    masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

    x_masked_context = masks_context * x
    if return_msk_context:
        return x_masked_context

    masks_objects = (1 - masks_context)
    x_masked_objects = masks_objects * x
    return x_masked_objects


class ContextMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ContextMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        x_masked_context = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                          upscale=self.upscale, return_msk_context=True)
        x_masked_context = self.upsample(x_masked_context)
        return x_masked_context


class ObjectMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ObjectMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, pred=None):
        x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                      upscale=self.upscale, return_msk_context=False)
        x_masked_obj = self.upsample(x_masked_obj)

        return x_masked_obj


class CCT(BaseModel):
    def __init__(self,num_classes,sup_loss=None, cons_w_unsup=None,  testing=False,
            pretrained=True,use_weak_lables=True, weakly_loss_w=0.4):
        super(CCT, self).__init__()
        self.model='semi'
     
        self.unsuper_loss=softmax_mse_loss
        self.unsup_loss_w=cons_w_unsup
        self.sup_loss_w=1
        self.softmax_temp=1
        self.sup_loss=sup_loss
        self.use_weak_labels=use_weak_lables
        self.weakly_loss_w=weakly_loss_w

        self.aux_constraint=False
        self.aux_constraint_w=1

        self.confidence_th=0.5
        self.confidence_masking=False

        self.encoder=Encoder(pretrained=pretrained)

        upscale=32
        num_out_ch=384
        decoder_in_ch=num_out_ch//4
        self.main_decoder=MainDecoder(upscale,decoder_in_ch,num_classes=num_classes)
        vat_decoder = [VATDecoder(upscale, decoder_in_ch, num_classes, xi=1e-6,
                                  eps=2.0) for _ in range(2)]
        drop_decoder = [DropOutDecoder(upscale, decoder_in_ch, num_classes,
                                       drop_rate=0.5, spatial_dropout=true)
                        for _ in range(6)]
        cut_decoder = [CutOutDecoder(upscale, decoder_in_ch, num_classes, erase=0.4)
                       for _ in range(6)]
        context_m_decoder = [ContextMaskingDecoder(upscale, decoder_in_ch, num_classes)
                             for _ in range(2)]
        object_masking = [ObjectMaskingDecoder(upscale, decoder_in_ch, num_classes)
                          for _ in range(2)]
        feature_drop = [FeatureDropDecoder(upscale, decoder_in_ch, num_classes)
                        for _ in range(6)]
        feature_noise = [FeatureNoiseDecoder(upscale, decoder_in_ch, num_classes,
                                             uniform_range=0.3)
                         for _ in range(6)]

        self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
                                           *context_m_decoder, *object_masking, *feature_drop, *feature_noise])

    def forward(self,x_l=None,target_l=None,x_ul=None,target_ul=None,curr_iter=None,epoch=None):
        if not self.training:
            return self.main_decoder(self.encoder(x_l))

        input_size = (x_l.size(2), x_l.size(3))
        output_l = self.main_decoder(self.encoder(x_l))
        if output_l.shape != x_l.shape:
            output_l = F.interpolate(output_l, size=input_size, mode='bilinear', align_corners=True)

        # Supervised loss


        loss_sup = self.sup_loss(output_l, target_l, curr_iter=curr_iter, epoch=epoch) * self.sup_loss_w

        # If supervised mode only, return
        if self.mode == 'supervised':
            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup
            return total_loss, curr_losses, outputs

        # If semi supervised mode
        elif self.mode == 'semi':
            # Get main prediction
            x_ul = self.encoder(x_ul)
            output_ul = self.main_decoder(x_ul)

            # Get auxiliary predictions
            outputs_ul = [aux_decoder(x_ul, output_ul.detach()) for aux_decoder in self.aux_decoders]
            targets = F.softmax(output_ul.detach(), dim=1)

            # Compute unsupervised loss
            loss_unsup = sum([self.unsuper_loss(inputs=u, targets=targets, \
                                                conf_mask=self.confidence_masking, threshold=self.confidence_th,
                                                use_softmax=False)
                              for u in outputs_ul])
            loss_unsup = (loss_unsup / len(outputs_ul))
            curr_losses = {'loss_sup': loss_sup}

            if output_ul.shape != x_l.shape:
                output_ul = F.interpolate(output_ul, size=input_size, mode='bilinear', align_corners=True)
            outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}

            # Compute the unsupervised loss
            weight_u = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
            loss_unsup = loss_unsup * weight_u
            curr_losses['loss_unsup'] = loss_unsup
            total_loss = loss_unsup + loss_sup

            # If case we're using weak lables, add the weak loss term with a weight (self.weakly_loss_w)
            if self.use_weak_lables:
                weight_w = (weight_u / self.unsup_loss_w.final_w) * self.weakly_loss_w
                loss_weakly = sum(
                    [self.sup_loss(outp, target_ul) for outp in outputs_ul]) / len(
                    outputs_ul)
                loss_weakly = loss_weakly * weight_w
                curr_losses['loss_weakly'] = loss_weakly
                total_loss += loss_weakly

            # Pair-wise loss
            if self.aux_constraint:
                pair_wise = pair_wise_loss(outputs_ul) * self.aux_constraint_w
                curr_losses['pair_wise'] = pair_wise
                loss_unsup += pair_wise

            return total_loss, curr_losses, outputs

    def get_backbone_params(self):
            return self.encoder.get_backbone_params()

    def get_other_params(self):
            if self.mode == 'semi':
                return chain(self.encoder.get_module_params(), self.main_decoder.parameters(),
                             self.aux_decoders.parameters())

            return chain(self.encoder.get_module_params(), self.main_decoder.parameters())




import torch
import torch.nn.functional as F
from itertools import cycle
import torch.nn as nn
from tqdm import tqdm
import os
import math
import logging
from torchtools.optim import RangerLars
import sys
class flat_and_anneal(nn.Module):
    def __init__(self, epochs, anneal_start=0.5, base_lr=0.001, min_lr=0):
        super(flat_and_anneal, self).__init__()
        self.epochs = epochs
        self.anneal_start = anneal_start
        self.base_lr = base_lr
        self.min_lr = min_lr

    def forward(self, epoch, optimizer):
        if epoch >= 30:
            epoch = epoch - 30
            for param in optimizer.param_groups:
                lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / 5)) / 2
                param['lr'] = lr

class BaseTrainer:
    def __init__(self, model, resume, iters_per_epoch,epochs,train_logger=None):
        self.model = model

        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = True
        self.start_epoch = 1
        self.improved = False

        # SETTING THE DEVICE

        self.device=torch.device('cuda:0')
        self.model.to(self.device)

        # CONFIGS
        self.epochs = epochs
        self.save_period ='result/'


        self.optimizer =RangerLars(model.parameters(),lr=0.001)
        model_params = sum([i.shape.numel() for i in list(model.parameters())])
        opt_params = sum([i.shape.numel() for j in self.optimizer.param_groups for i in j['params']])
        assert opt_params == model_params, 'some params are missing in the opt'

        self.lr_scheduler =flat_and_anneal(epochs=epochs)
        # MONITORING
        self.monitor = 'max Mean_IoU'
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = 10

        # CHECKPOINTS & TENSOBOARD




        if resume: self._resume_checkpoint(resume)

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu

        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            results = self._train_epoch(epoch)
            if self.do_validation and epoch % 5:
                results = self._valid_epoch(epoch)
                self.logger.info('\n\n')
                for k, v in results.items():
                    self.logger.info(f'         {str(k):15s}: {v}')

            if self.train_logger is not None:
                log = {'epoch': epoch, **results}
                self.train_logger.add_entry(log)

            # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
            if self.mnt_mode != 'off' and epoch % 5 == 0:
                try:
                    if self.mnt_mode == 'min':
                        self.improved = (log[self.mnt_metric] < self.mnt_best)
                    else:
                        self.improved = (log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                    break

                if self.improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stoping:
                    self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    self.logger.warning('Training Stoped')
                    break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)
        self.html_results.save()

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'monitor_best': self.mnt_best,

        }

        filename = os.path.join(self.checkpoint_dir, f'checkpoint.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(f'Error when loading: {e}')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        if "logger" in checkpoint.keys():
            self.train_logger = checkpoint['logger']
        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError

import torch
import torch.nn.functional as F
from itertools import cycle
import torch.nn as nn
from tqdm import tqdm
import os
from math import ceil
import logging
from torchtools.optim import RangerLars
import sys


import numpy as np
from torchvision.utils import make_grid
import PIL
def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    palette[-3:] = [255, 255, 255]
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

class Trainer(BaseTrainer):
    def __init__(self, model, resume, epochs,supervised_loader, unsupervised_loader, iter_per_epoch,
                 val_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, resume, iter_per_epoch,epochs, train_logger)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader


        self.wrt_mode, self.wrt_step = 'train_', 0


        self.num_classes = 2
        self.mode = 'semi'




    def _train_epoch(self, epoch):



        self.model.train()

        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=135)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
            tbar = tqdm(range(len(self.unsupervised_loader)), ncols=135)

        self._reset_metrics()
        for batch_idx in tbar:
            if self.mode == 'supervised':
                (input_l, target_l,l_labels), (input_ul, target_ul,ul_labels) = next(dataloader), (None, None)
            else:
                (input_l, target_l,l_labels), (input_ul, target_ul,ul_labels) = next(dataloader)
                input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)

            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            self.optimizer.zero_grad()

            total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                         curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1)
            total_loss = total_loss.mean()
            total_loss.backward()
            self.optimizer.step()

            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, target_ul, epoch - 1)

            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs

            tbar.set_description('T ({}) | Ls {:.2f} Lu {:.2f} Lw {:.2f} PW {:.2f} m1 {:.2f} m2 {:.2f}|'.format(
                epoch, self.loss_sup.average, self.loss_unsup.average, self.loss_weakly.average,
                self.pair_wise.average, self.mIoU_l, self.mIoU_ul))

            self.lr_scheduler(epoch - 1,self.optimizer)

        return

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'
        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')
                output = self.model(data)
                output = output[:, :, :H, :W]

                # LOSS
                loss = F.cross_entropy(output, target)
                total_loss_val.update(loss.item())

                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes)
                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                # LIST OF IMAGE TO VIZ (15 images)


                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                               "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}

                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format(epoch,
                                                                                                             total_loss_val.average,
                                                                                                             pixAcc,
                                                                                                             mIoU))

                self._save_checkpoint(epoch, save_best=self.improved)


    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}

    def _update_losses(self, cur_losses):
        if "loss_sup" in cur_losses.keys():
            self.loss_sup.update(cur_losses['loss_sup'].mean().item())
        if "loss_unsup" in cur_losses.keys():
            self.loss_unsup.update(cur_losses['loss_unsup'].mean().item())
        if "loss_weakly" in cur_losses.keys():
            self.loss_weakly.update(cur_losses['loss_weakly'].mean().item())
        if "pair_wise" in cur_losses.keys():
            self.pair_wise.update(cur_losses['pair_wise'].mean().item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes)
        self._update_seg_metrics(*seg_metrics_l, True)
        seg_metrics_l = self._get_seg_metrics(True)
        self.mIoU_l, self.pixel_acc_l, self.class_iou_l = seg_metrics_l.values()

        if self.mode == 'semi':
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes)
            self._update_seg_metrics(*seg_metrics_ul, False)
            seg_metrics_ul = self._get_seg_metrics(False)
            self.mIoU_ul, self.pixel_acc_ul, self.class_iou_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }

    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
        current_rampup = self.model.module.unsup_loss_w.current_rampup
        self.writer.add_scalar('train/Unsupervised_rampup', current_rampup, self.wrt_step)

    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                    else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0) // len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

        if self.mode == 'semi':
            outputs_ul_np = outputs['unsup_pred'].data.max(1)[1].cpu().numpy()
            targets_ul_np = target_ul.data.cpu().numpy()
            imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_ul, outputs_ul_np, targets_ul_np)]
            self._add_img_tb(imgs, 'unsupervised')


import torch
import torch.nn.functional as F
from itertools import cycle
import torch.nn as nn
from tqdm import tqdm
import os
import math
import logging
from torchtools.optim import RangerLars
import sys
torch.manual_seed(42)


def main():
    ramp_up=0.1
    epochs=100
    unsupervised_w=30

    supervised_loader=create_dataloader('train',6)
    unsupervised_loader=create_dataloader('unlabel',6)
    test_loader=create_dataloader('test',6)
    iter_per_epoch=len(unsupervised_loader)
    sup_loss=abCE_loss(iters_per_epoch=iter_per_epoch,epochs=epochs,num_classes=2)
    rampup_ends=int(ramp_up*epochs)
    cons_w_unsup=consistency_weight(final_w=unsupervised_w,iters_per_epoch=iter_per_epoch,rampup_ends=rampup_ends)
    model=CCT(num_classes=2,sup_loss=sup_loss,cons_w_unsup=cons_w_unsup,weakly_loss_w=0.4,)
    trainer=Trainer(model=model,resume=None,epochs=epochs,
                    supervised_loader=supervised_loader,
                    unsupervised_loader=unsupervised_loader,
                    iter_per_epoch=iter_per_epoch,
                    val_loader=test_loader)

main()
import matplotlib.pyplot as plt
%matplotlib inline
image=cv2.imread('/kaggle/input/new-hulianwang/train_data/sofa/masks/sofa_8.png',0)
print(image.shape)
plt.imshow(image,cmap='gray')
