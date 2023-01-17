import sys

sys.path.insert(0, "../input/timm-efficientdet-pytorch")

sys.path.insert(0, "../input/omegaconf")

sys.path.insert(0, "../input/weightedboxesfusion")

sys.path.insert(0, "../input/faaltuu/")



import ensemble_boxes

import torch

from effdet import get_efficientdet_config, EfficientDet, DetBenchEval

from effdet.efficientdet import HeadNet

from sklearn.model_selection import StratifiedKFold

import torchvision

from  torchvision.models.utils import load_state_dict_from_url

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.backbone_utils import BackboneWithFPN

from torchvision.ops import misc as misc_nn_ops

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

from collections import OrderedDict

from torch import nn

import warnings

from torch.jit.annotations import Tuple, List, Dict, Optional

from timm.models.resnest import resnest101e,resnest269e

import numpy as np

import pandas as pd

from glob import glob

from torch.utils.data import Dataset,DataLoader

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

import cv2

import gc
all_path = glob('../input/global-wheat-detection/test/*')

DATA_ROOT_PATH = '../input/global-wheat-detection/test'
def get_valid_transforms():

    return A.Compose([

            A.Resize(height=1024, width=1024, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)
class DatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
dataset = DatasetRetriever(

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),

    transforms=get_valid_transforms()

)



def collate_fn(batch):

    return tuple(zip(*batch))



data_loader = DataLoader(

    dataset,

    batch_size=4,

    shuffle=False,

    num_workers=2,

    drop_last=False,

    collate_fn=collate_fn

)
class CrossEntropyLabelSmooth(nn.Module):

    """Cross entropy loss with label smoothing regularizer.



    Reference:

    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    Equation: y = (1 - epsilon) * y + epsilon / K.



    Args:

        num_classes (int): number of classes.

        epsilon (float): weight.

    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):

        super(CrossEntropyLabelSmooth, self).__init__()

        self.num_classes = num_classes

        self.epsilon = epsilon

        self.use_gpu = use_gpu

        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self, inputs, targets):

        """

        Args:

            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)

            targets: ground truth labels with shape (num_classes)

        """

        log_probs = self.logsoftmax(inputs)

        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        if self.use_gpu: targets = targets.cuda()

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        loss = (- targets * log_probs).mean(0).sum()

        return loss
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):

    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]

    """

    Computes the loss for Faster R-CNN.

    Arguments:

        class_logits (Tensor)

        box_regression (Tensor)

        labels (list[BoxList])

        regression_targets (Tensor)

    Returns:

        classification_loss (Tensor)

        box_loss (Tensor)

    """



    labels = torch.cat(labels, dim=0)

    regression_targets = torch.cat(regression_targets, dim=0)

    labal_smooth_loss = CrossEntropyLabelSmooth(2)

    classification_loss = labal_smooth_loss(class_logits, labels)



    # get indices that correspond to the regression targets for

    # the corresponding ground truth labels, to be used with

    # advanced indexing

    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)

    labels_pos = labels[sampled_pos_inds_subset]

    N, num_classes = class_logits.shape

    box_regression = box_regression.reshape(N, -1, 4)



    box_loss = det_utils.smooth_l1_loss(

        box_regression[sampled_pos_inds_subset, labels_pos],

        regression_targets[sampled_pos_inds_subset],

        beta=1 / 9,

        size_average=False,

    )

    box_loss = box_loss / labels.numel()



    return classification_loss, box_loss
def fpn_backbone_269(pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3):

    # backbone = resnet.__dict__['resnet18'](pretrained=pretrained,norm_layer=norm_layer)

    print(f'\nPretarined is {pretrained}')

    backbone = resnest269e(pretrained=pretrained)

    # select layers that wont be frozen

    assert trainable_layers <= 5 and trainable_layers >= 0

    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # freeze layers only if pretrained backbone is used

    for name, parameter in backbone.named_parameters():

        if all([not name.startswith(layer) for layer in layers_to_train]):

            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8

    in_channels_list = [

        in_channels_stage2,

        in_channels_stage2 * 2,

        in_channels_stage2 * 4,

        in_channels_stage2 * 8,

    ]

    out_channels = 256

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)



class WheatDetector_269(nn.Module):

    def __init__(self, **kwargs):

        super(WheatDetector_269, self).__init__()

        self.backbone = fpn_backbone_269(pretrained=False)

        self.base = FasterRCNN(self.backbone, num_classes = 2, **kwargs)

        self.base.roi_heads.fastrcnn_loss = self.fastrcnn_loss



    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):

        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]

        """

        Computes the loss for Faster R-CNN.

        Arguments:

            class_logits (Tensor)

            box_regression (Tensor)

            labels (list[BoxList])

            regression_targets (Tensor)

        Returns:

            classification_loss (Tensor)

            box_loss (Tensor)

        """



        labels = torch.cat(labels, dim=0)

        regression_targets = torch.cat(regression_targets, dim=0)

        labal_smooth_loss = CrossEntropyLabelSmooth(2)

        classification_loss = labal_smooth_loss(class_logits, labels)



        # get indices that correspond to the regression targets for

        # the corresponding ground truth labels, to be used with

        # advanced indexing

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)

        labels_pos = labels[sampled_pos_inds_subset]

        N, num_classes = class_logits.shape

        box_regression = box_regression.reshape(N, -1, 4)



        box_loss = det_utils.smooth_l1_loss(

            box_regression[sampled_pos_inds_subset, labels_pos],

            regression_targets[sampled_pos_inds_subset],

            beta=1 / 9,

            size_average=False,

        )

        box_loss = box_loss / labels.numel()



        return classification_loss, box_loss



    def forward(self, images, targets=None):

        return self.base(images, targets)
def fpn_backbone_101(pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3):

    # backbone = resnet.__dict__['resnet18'](pretrained=pretrained,norm_layer=norm_layer)

    print(f'\nPretarined is {pretrained}')

    backbone = resnest101e(pretrained=pretrained)

    # select layers that wont be frozen

    assert trainable_layers <= 5 and trainable_layers >= 0

    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # freeze layers only if pretrained backbone is used

    for name, parameter in backbone.named_parameters():

        if all([not name.startswith(layer) for layer in layers_to_train]):

            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8

    in_channels_list = [

        in_channels_stage2,

        in_channels_stage2 * 2,

        in_channels_stage2 * 4,

        in_channels_stage2 * 8,

    ]

    out_channels = 256

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)



class WheatDetector_101(nn.Module):

    def __init__(self, **kwargs):

        super(WheatDetector_101, self).__init__()

        self.backbone = fpn_backbone_101(pretrained=False)

        self.base = FasterRCNN(self.backbone, num_classes = 2, **kwargs)

        self.base.roi_heads.fastrcnn_loss = self.fastrcnn_loss



    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):

        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]

        """

        Computes the loss for Faster R-CNN.

        Arguments:

            class_logits (Tensor)

            box_regression (Tensor)

            labels (list[BoxList])

            regression_targets (Tensor)

        Returns:

            classification_loss (Tensor)

            box_loss (Tensor)

        """



        labels = torch.cat(labels, dim=0)

        regression_targets = torch.cat(regression_targets, dim=0)

        labal_smooth_loss = CrossEntropyLabelSmooth(2)

        classification_loss = labal_smooth_loss(class_logits, labels)



        # get indices that correspond to the regression targets for

        # the corresponding ground truth labels, to be used with

        # advanced indexing

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)

        labels_pos = labels[sampled_pos_inds_subset]

        N, num_classes = class_logits.shape

        box_regression = box_regression.reshape(N, -1, 4)



        box_loss = det_utils.smooth_l1_loss(

            box_regression[sampled_pos_inds_subset, labels_pos],

            regression_targets[sampled_pos_inds_subset],

            beta=1 / 9,

            size_average=False,

        )

        box_loss = box_loss / labels.numel()



        return classification_loss, box_loss



    def forward(self, images, targets=None):

        return self.base(images, targets)
def load_net_269(checkpoint_path):

    model = WheatDetector_269()



    # Load the trained weights

    checkpoint=torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])





    del checkpoint

    gc.collect()



    model.eval();

    return model.cuda()



def load_net_101(checkpoint_path):

    model = WheatDetector_101()



    # Load the trained weights

    checkpoint=torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])





    del checkpoint

    gc.collect()



    model.eval();

    return model.cuda()



models = [

    load_net_101('../input/resnest101/fold0-best-checkpoint.bin'),

    load_net_101('../input/resnest101/fold1-best-checkpoint.bin'),

    load_net_101('../input/resnest101/fold2-best-checkpoint.bin'),

    load_net_101('../input/resnest101/fold3-best-checkpoint.bin'),

    load_net_101('../input/resnest101/fold4-best-checkpoint.bin'),

    load_net_269('../input/resnest269/fold0-best-checkpoint.bin'),

    load_net_269('../input/resnest269/fold1-best-checkpoint.bin'),

    load_net_269('../input/resnest269/fold2-best-checkpoint.bin'),

    load_net_269('../input/resnest269/fold3-best-checkpoint.bin'),

    load_net_269('../input/resnest269/fold4-best-checkpoint.bin')

]
class BaseWheatTTA:

    """ author: @shonenkov """

    image_size = 1024



    def augment(self, image):

        raise NotImplementedError

    

    def batch_augment(self, images):

        raise NotImplementedError

    

    def deaugment_boxes(self, boxes):

        raise NotImplementedError



class TTAHorizontalFlip(BaseWheatTTA):

    """ author: @shonenkov """



    def augment(self, image):

        return image.flip(1)

    

    def batch_augment(self, images):

        return images.flip(2)

    

    def deaugment_boxes(self, boxes):

        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]

        return boxes



class TTAVerticalFlip(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return image.flip(2)

    

    def batch_augment(self, images):

        return images.flip(3)

    

    def deaugment_boxes(self, boxes):

        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]

        return boxes

    

class TTARotate90(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return torch.rot90(image, 1, (1, 2))



    def batch_augment(self, images):

        return torch.rot90(images, 1, (2, 3))

    

    def deaugment_boxes(self, boxes):

        res_boxes = boxes.copy()

        res_boxes[:, [0,2]] = self.image_size - boxes[:, [3,1]] 

        res_boxes[:, [1,3]] = boxes[:, [0,2]]

        return res_boxes



class TTACompose(BaseWheatTTA):

    """ author: @shonenkov """

    def __init__(self, transforms):

        self.transforms = transforms

        

    def augment(self, image):

        for transform in self.transforms:

            image = transform.augment(image)

        return image

    

    def batch_augment(self, images):

        for transform in self.transforms:

            images = transform.batch_augment(images)

        return images

    

    def prepare_boxes(self, boxes):

        result_boxes = boxes.copy()

        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)

        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)

        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)

        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)

        return result_boxes

    

    def deaugment_boxes(self, boxes):

        for transform in self.transforms[::-1]:

            boxes = transform.deaugment_boxes(boxes)

        return self.prepare_boxes(boxes)
def process_det(index, outputs, score_threshold=0.5):

    boxes = outputs[index]['boxes'].data.cpu().numpy()   

    scores = outputs[index]['scores'].data.cpu().numpy()

    boxes = (boxes).clip(min=0, max=1023).astype(int)

    indexes = np.where(scores>score_threshold)

    boxes = boxes[indexes]

    scores = scores[indexes]

    return boxes, scores
from itertools import product



tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None], 

                               [TTAVerticalFlip(), None],

                               [TTARotate90(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
def make_tta_predictions(images,net, score_threshold=0.1):

    with torch.no_grad():

        images = torch.stack(images).float().cuda()

        predictions = []

        for tta_transform in tta_transforms:

            result = []

            outputs = net(tta_transform.batch_augment(images.clone()))



            for i, image in enumerate(images):

                boxes = outputs[i]['boxes'].data.cpu().numpy()   

                scores = outputs[i]['scores'].data.cpu().numpy()

                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]

                boxes = tta_transform.deaugment_boxes(boxes.copy())

                result.append({

                    'boxes': boxes,

                    'scores': scores[indexes],

                })

            predictions.append(result)

    return predictions



def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.5, skip_box_thr=0.43, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist() for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
fold1 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[0])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold1[image_id] = [boxes, scores, labels]

print('\nCompleted')

        

fold2 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[1])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold2[image_id] = [boxes, scores, labels]

print('\nCompleted')



        

fold3 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[2])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold3[image_id] = [boxes, scores, labels]

print('\nCompleted')



fold4 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[3])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold4[image_id] = [boxes, scores, labels]

print('\nCompleted')

fold5 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[4])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold5[image_id] = [boxes, scores, labels]

print('\nCompleted')



        

fold6 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[5])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold6[image_id] = [boxes, scores, labels]

print('\nCompleted')



        

fold7 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[6])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold7[image_id] = [boxes, scores, labels]

print('\nCompleted')

       

fold8 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[7])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold8[image_id] = [boxes, scores, labels]

print('\nCompleted')



fold9 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[8])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold9[image_id] = [boxes, scores, labels]

print('\nCompleted')



fold10 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[9])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold10[image_id] = [boxes, scores, labels]

print('\nCompleted')

def run_last_wbf(model1,model2,model3,model4,

                 model5,model6,model7,model8,model9,model10,

                 iou_thr=0.5,skip_box_thr=0.43):

    

    box1,scores1,labels1 = model1

    box2,scores2,labels2 = model2

    box3,scores3,labels3 = model3

    box4,scores4,labels4 = model4

    

    

    box1 = box1/1023

    box2 = box2/1023

    box3 = box3/1023

    box4 = box4/1023

     

    box5,scores5,labels5 = model5

    box6,scores6,labels6 = model6

    box7,scores7,labels7 = model7

    box8,scores8,labels8 = model8

    box9,scores9,labels9 = model9

    box10,scores10,labels10 = model10

    

    

    box5 = box5/1023

    box6 = box6/1023

    box7 = box7/1023

    box8 = box8/1023

    box9 = box9/1023

    box10 = box10/1023

    

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion([box1,box2,box3,box4,box5,box6,box7,box8,box9,box10], 

                                                  [scores1,scores2,scores3,scores4,scores5,scores6,scores7,scores8,scores9,scores10],

                                                [labels1,labels2,labels3,labels4,labels5,labels6,labels7,labels8,labels9,labels10],

                                                  weights=None,iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    return boxes,scores,labels
w = {}

for row in range(len(all_path)):

    image_id = all_path[row].split("/")[-1].split(".")[0]

    boxes,scores,labels = run_last_wbf(fold1[image_id],fold2[image_id],fold3[image_id],fold4[image_id],

                               fold5[image_id],fold6[image_id],fold7[image_id],fold8[image_id],fold9[image_id],fold10[image_id])

    boxes = (boxes*1023)

    indexes = np.where(scores > 0.50)[0]

    w[image_id] = [boxes,scores,labels]
def get_valid_transforms():

    return A.Compose([

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)
class DatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
dataset = DatasetRetriever(

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),

    transforms=get_valid_transforms()

)



def collate_fn(batch):

    return tuple(zip(*batch))



data_loader = DataLoader(

    dataset,

    batch_size=4,

    shuffle=False,

    num_workers=2,

    drop_last=False,

    collate_fn=collate_fn

)
class CrossEntropyLabelSmooth(nn.Module):

    """Cross entropy loss with label smoothing regularizer.



    Reference:

    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    Equation: y = (1 - epsilon) * y + epsilon / K.



    Args:

        num_classes (int): number of classes.

        epsilon (float): weight.

    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):

        super(CrossEntropyLabelSmooth, self).__init__()

        self.num_classes = num_classes

        self.epsilon = epsilon

        self.use_gpu = use_gpu

        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self, inputs, targets):

        """

        Args:

            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)

            targets: ground truth labels with shape (num_classes)

        """

        log_probs = self.logsoftmax(inputs)

        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        if self.use_gpu: targets = targets.cuda()

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        loss = (- targets * log_probs).mean(0).sum()

        return loss

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):

    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]

    """

    Computes the loss for Faster R-CNN.

    Arguments:

        class_logits (Tensor)

        box_regression (Tensor)

        labels (list[BoxList])

        regression_targets (Tensor)

    Returns:

        classification_loss (Tensor)

        box_loss (Tensor)

    """



    labels = torch.cat(labels, dim=0)

    regression_targets = torch.cat(regression_targets, dim=0)

    labal_smooth_loss = CrossEntropyLabelSmooth(2)

    classification_loss = labal_smooth_loss(class_logits, labels)



    # get indices that correspond to the regression targets for

    # the corresponding ground truth labels, to be used with

    # advanced indexing

    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)

    labels_pos = labels[sampled_pos_inds_subset]

    N, num_classes = class_logits.shape

    box_regression = box_regression.reshape(N, -1, 4)



    box_loss = det_utils.smooth_l1_loss(

        box_regression[sampled_pos_inds_subset, labels_pos],

        regression_targets[sampled_pos_inds_subset],

        beta=1 / 9,

        size_average=False,

    )

    box_loss = box_loss / labels.numel()



    return classification_loss, box_loss

def fpn_backbone(pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3):

    # backbone = resnet.__dict__['resnet18'](pretrained=pretrained,norm_layer=norm_layer)

    print(f'\nPretarined is {pretrained}')

    backbone = resnest101e(pretrained=pretrained)

    # select layers that wont be frozen

    assert trainable_layers <= 5 and trainable_layers >= 0

    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # freeze layers only if pretrained backbone is used

    for name, parameter in backbone.named_parameters():

        if all([not name.startswith(layer) for layer in layers_to_train]):

            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8

    in_channels_list = [

        in_channels_stage2,

        in_channels_stage2 * 2,

        in_channels_stage2 * 4,

        in_channels_stage2 * 8,

    ]

    out_channels = 256

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

class WheatDetector(nn.Module):

    def __init__(self, **kwargs):

        super(WheatDetector, self).__init__()

        self.backbone = fpn_backbone(pretrained=False)

        self.base = FasterRCNN(self.backbone, num_classes = 2, **kwargs)

        self.base.roi_heads.fastrcnn_loss = self.fastrcnn_loss



    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):

        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]

        """

        Computes the loss for Faster R-CNN.

        Arguments:

            class_logits (Tensor)

            box_regression (Tensor)

            labels (list[BoxList])

            regression_targets (Tensor)

        Returns:

            classification_loss (Tensor)

            box_loss (Tensor)

        """



        labels = torch.cat(labels, dim=0)

        regression_targets = torch.cat(regression_targets, dim=0)

        labal_smooth_loss = CrossEntropyLabelSmooth(2)

        classification_loss = labal_smooth_loss(class_logits, labels)



        # get indices that correspond to the regression targets for

        # the corresponding ground truth labels, to be used with

        # advanced indexing

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)

        labels_pos = labels[sampled_pos_inds_subset]

        N, num_classes = class_logits.shape

        box_regression = box_regression.reshape(N, -1, 4)



        box_loss = det_utils.smooth_l1_loss(

            box_regression[sampled_pos_inds_subset, labels_pos],

            regression_targets[sampled_pos_inds_subset],

            beta=1 / 9,

            size_average=False,

        )

        box_loss = box_loss / labels.numel()



        return classification_loss, box_loss



    def forward(self, images, targets=None):

        return self.base(images, targets)

def load_net(checkpoint_path):

    model = WheatDetector()



    # Load the trained weights

    checkpoint=torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])





    del checkpoint

    gc.collect()



    model.eval();

    return model.cuda()

models = [

    load_net('../input/fold-1-resnest101-512/best-checkpoint.bin'),

    load_net('../input/fold2resnest101512/best-checkpoint.bin'),

    load_net('../input/fold3resnest101512/best-checkpoint.bin'),

    load_net('../input/resnest101fold4/best-checkpoint.bin'),

    load_net('../input/resnest101fold3/best-checkpoint.bin')

]
class BaseWheatTTA:

    """ author: @shonenkov """

    image_size = 512



    def augment(self, image):

        raise NotImplementedError

    

    def batch_augment(self, images):

        raise NotImplementedError

    

    def deaugment_boxes(self, boxes):

        raise NotImplementedError



class TTAHorizontalFlip(BaseWheatTTA):

    """ author: @shonenkov """



    def augment(self, image):

        return image.flip(1)

    

    def batch_augment(self, images):

        return images.flip(2)

    

    def deaugment_boxes(self, boxes):

        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]

        return boxes



class TTAVerticalFlip(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return image.flip(2)

    

    def batch_augment(self, images):

        return images.flip(3)

    

    def deaugment_boxes(self, boxes):

        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]

        return boxes

    

class TTARotate90(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return torch.rot90(image, 1, (1, 2))



    def batch_augment(self, images):

        return torch.rot90(images, 1, (2, 3))

    

    def deaugment_boxes(self, boxes):

        res_boxes = boxes.copy()

        res_boxes[:, [0,2]] = self.image_size - boxes[:, [3,1]] 

        res_boxes[:, [1,3]] = boxes[:, [0,2]]

        return res_boxes



class TTACompose(BaseWheatTTA):

    """ author: @shonenkov """

    def __init__(self, transforms):

        self.transforms = transforms

        

    def augment(self, image):

        for transform in self.transforms:

            image = transform.augment(image)

        return image

    

    def batch_augment(self, images):

        for transform in self.transforms:

            images = transform.batch_augment(images)

        return images

    

    def prepare_boxes(self, boxes):

        result_boxes = boxes.copy()

        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)

        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)

        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)

        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)

        return result_boxes

    

    def deaugment_boxes(self, boxes):

        for transform in self.transforms[::-1]:

            boxes = transform.deaugment_boxes(boxes)

        return self.prepare_boxes(boxes)

from itertools import product



tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None], 

                               [TTAVerticalFlip(), None],

                               [TTARotate90(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
def make_tta_predictions(images,net, score_threshold=0.1):

    with torch.no_grad():

        images = torch.stack(images).float().cuda()

        predictions = []

        for tta_transform in tta_transforms:

            result = []

            outputs = net(tta_transform.batch_augment(images.clone()))



            for i, image in enumerate(images):

                boxes = outputs[i]['boxes'].data.cpu().numpy()   

                scores = outputs[i]['scores'].data.cpu().numpy()

                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]

                boxes = tta_transform.deaugment_boxes(boxes.copy())

                result.append({

                    'boxes': boxes,

                    'scores': scores[indexes],

                })

            predictions.append(result)

    return predictions



def run_wbf(predictions, image_index, image_size=512, iou_thr=0.5, skip_box_thr=0.43, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist() for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*1023

    return boxes, scores, labels
fold0 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[0])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold0[image_id] = [boxes, scores, labels]

        

print('\nCompleted')





fold1 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[1])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold1[image_id] = [boxes, scores, labels]

        

print('\nCompleted')

        

fold2 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[2])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold2[image_id] = [boxes, scores, labels]

        

print('\nCompleted')





fold3 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[3])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold3[image_id] = [boxes, scores, labels]

        

print('\nCompleted')

        

fold4 = {}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images,models[4])

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        image_id = image_ids[i]

        fold4[image_id] = [boxes, scores, labels]

        

print('\nCompleted')
def run_last_wbf(model1,model2,model3,model4,model5,

                 iou_thr=0.5,skip_box_thr=0.43):

    

    box1,scores1,labels1 = model1

    box2,scores2,labels2 = model2

    box3,scores3,labels3 = model3

    box4,scores4,labels4 = model4

    box5,scores5,labels5 = model5

    

    

    box1 = box1/1023

    box2 = box2/1023

    box3 = box3/1023

    box4 = box4/1023

    box5 = box5/1023



    

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion([box1,box2,box3,box4,box5], 

                                                  [scores1,scores2,scores3,scores4,scores5],

                                                [labels1,labels2,labels3,labels4,labels5],

                                                  weights=None,iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    return boxes,scores,labels

x = {}

for row in range(len(all_path)):

    image_id = all_path[row].split("/")[-1].split(".")[0]

    boxes,scores,labels = run_last_wbf(fold1[image_id],fold2[image_id],fold0[image_id],fold3[image_id],fold4[image_id])

    boxes = (boxes*1023)

    x[image_id] = [boxes,scores,labels]
def load_net(checkpoint_path):

    config = get_efficientdet_config('tf_efficientdet_d5')

    net = EfficientDet(config, pretrained_backbone=False)



    config.num_classes = 1

    config.image_size=512

    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))



    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['model_state_dict'])



    del checkpoint

    gc.collect()



    net = DetBenchEval(net, config)

    net.eval();

    return net.cuda()

models = [

    load_net('../input/efficientdetd51/0.bin'),

    load_net('../input/efficientdetd51/1.bin'),

    load_net('../input/efficientdetd51/2.bin'),

    load_net('../input/efficientdetd51/3.bin'),

    load_net('../input/efficientdetd51/4.bin'),

    load_net('../input/efficientdetd51/5.bin'),

    load_net('../input/efficientdetd51/6.bin'),

    load_net('../input/efficientdetd51/7.bin'),

]
DATA_ROOT_PATH = '../input/global-wheat-detection/test/'

class TestDatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]



def get_valid_transforms():

    return A.Compose(

        [

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(p=1.0),

        ], 

        p=1.0, 

    )

dataset = TestDatasetRetriever(

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),

    transforms=get_valid_transforms()

)



def collate_fn(batch):

    return tuple(zip(*batch))



data_loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)



def make_predictions(images, score_threshold=0.35):

    images = torch.stack(images).cuda().float()

    predictions = []

    for net in models:

        with torch.no_grad():

            det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())

            result = []

            for i in range(images.shape[0]):

                boxes = det[i].detach().cpu().numpy()[:,:4]    

                scores = det[i].detach().cpu().numpy()[:,4]

                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]

                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]

                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

                result.append({

                    'boxes': boxes[indexes],

                    'scores': scores[indexes],

                    })

            predictions.append(result)

    return predictions
def run_wbf(predictions, image_index, image_size=512, iou_thr=0.432, skip_box_thr=0.397, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
y={}

for images, image_ids in data_loader:

    predictions = make_predictions(images)

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i,)

        boxes = (boxes*2)

        y[image_ids[i]] = [boxes,scores,labels]
models = [

    load_net('../input/efficientdetd52/0.bin'),

    load_net('../input/efficientdetd52/1.bin'),

    load_net('../input/efficientdetd52/2.bin'),

    load_net('../input/efficientdetd52/3.bin'),

    load_net('../input/efficientdetd52/4.bin'),

    load_net('../input/kaggleeffnet/plabel_model/last-checkpoint1.bin'),

    load_net('../input/kaggleeffnet/plabel_model/best-checkpoint-004epoch.bin')

    ]
z={}

for images, image_ids in data_loader:

    predictions = make_predictions(images,0.4337)

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i,iou_thr=0.4637,skip_box_thr=0.12)

        boxes = (boxes*2)

        z[image_ids[i]] = [boxes,scores,labels]
def get_valid_transforms():

    return A.Compose(

        [

            A.Resize(height=1024, width=1024, p=1.0),

            ToTensorV2(p=1.0),

        ], 

        p=1.0, 

    )

dataset = TestDatasetRetriever(

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),

    transforms=get_valid_transforms()

)



def collate_fn(batch):

    return tuple(zip(*batch))



data_loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)

def make_tta_predictions(images, score_threshold=0.25):

    with torch.no_grad():

        images = torch.stack(images).float().cuda()

        predictions = []

        for tta_transform in tta_transforms:

            result = []

            det = net(tta_transform.batch_augment(images.clone()), torch.tensor([1]*images.shape[0]).float().cuda())

            

            for i in range(images.shape[0]):

                boxes = det[i].detach().cpu().numpy()[:,:4]    

                scores = det[i].detach().cpu().numpy()[:,4]

                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]

                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]

                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

                boxes = tta_transform.deaugment_boxes(boxes.copy())

                result.append({

                    'boxes': boxes,

                    'scores': scores[indexes],

                })

            predictions.append(result)

    return predictions



def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.432, skip_box_thr=0.397, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
class BaseWheatTTA:

    """ author: @shonenkov """

    image_size = 1024



    def augment(self, image):

        raise NotImplementedError

    

    def batch_augment(self, images):

        raise NotImplementedError

    

    def deaugment_boxes(self, boxes):

        raise NotImplementedError



class TTAHorizontalFlip(BaseWheatTTA):

    """ author: @shonenkov """



    def augment(self, image):

        return image.flip(1)

    

    def batch_augment(self, images):

        return images.flip(2)

    

    def deaugment_boxes(self, boxes):

        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]

        return boxes



class TTAVerticalFlip(BaseWheatTTA):

    """ author: @shonenkov """

    def augment(self, image):

        return image.flip(2)

    

    def batch_augment(self, images):

        return images.flip(3)

    

    def deaugment_boxes(self, boxes):

        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]

        return boxes

    

class TTARotate90(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return torch.rot90(image, 1, (1, 2))



    def batch_augment(self, images):

        return torch.rot90(images, 1, (2, 3))

    

    def deaugment_boxes(self, boxes):

        res_boxes = boxes.copy()

        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]

        res_boxes[:, [1,3]] = boxes[:, [2,0]]

        return res_boxes



class TTACompose(BaseWheatTTA):

    """ author: @shonenkov """

    def __init__(self, transforms):

        self.transforms = transforms

        

    def augment(self, image):

        for transform in self.transforms:

            image = transform.augment(image)

        return image

    

    def batch_augment(self, images):

        for transform in self.transforms:

            images = transform.batch_augment(images)

        return images

    

    def prepare_boxes(self, boxes):

        result_boxes = boxes.copy()

        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)

        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)

        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)

        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)

        return result_boxes

    

    def deaugment_boxes(self, boxes):

        for transform in self.transforms[::-1]:

            boxes = transform.deaugment_boxes(boxes)

        return self.prepare_boxes(boxes)

    

from itertools import product



tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None], 

                               [TTAVerticalFlip(), None],

                               [TTARotate90(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
def load_net(checkpoint_path):

    config = get_efficientdet_config('tf_efficientdet_d5')

    net = EfficientDet(config, pretrained_backbone=False)



    config.num_classes = 1

    config.image_size=1024

    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))



    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['model_state_dict'])



    del checkpoint

    gc.collect()



    net = DetBenchEval(net, config)

    net.eval();

    return net.cuda()



net = load_net('../input/effdet1024/best-checkpoint-1024.bin')
l={}

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images)

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        l[image_ids[i]] = [boxes, scores, labels]
def run_last_wbf(model1,model2,

                 iou_thr=0.4,skip_box_thr=0.1):

    

    box1,scores1,labels1 = model1

    box2,scores2,labels2 = model2

    

    box1 = box1/1023

    box2 = box2/1023

    

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion([box1,box2], [scores1,scores2],[labels1,labels2],

                                                                      weights=None,iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    return boxes,scores,labels
all_path = glob('../input/global-wheat-detection/test/*')

a = {}



for row in range(len(all_path)):

    image_id = all_path[row].split("/")[-1].split(".")[0]

    boxes,scores,labels = run_last_wbf(y[image_id],z[image_id])

    boxes = (boxes*1023)

    a[image_id] = [boxes,scores,labels]
def run_last_wbf(model1,model2,

                 iou_thr=0.4,skip_box_thr=0.1):

    

    box1,scores1,labels1 = model1

    box2,scores2,labels2 = model2

    

    box1 = box1/1023

    box2 = box2/1023

    

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion([box1,box2], [scores1,scores2],[labels1,labels2],

                                                                      weights=None,iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    return boxes,scores,labels




g = {}

for row in range(len(all_path)):

    image_id = all_path[row].split("/")[-1].split(".")[0]

    boxes,scores,labels = run_last_wbf(x[image_id],a[image_id])

    boxes = (boxes*1023)

    g[image_id] = [boxes,scores,labels]
def run_last_wbf(model1,model2,model3,

                 iou_thr=0.4,skip_box_thr=0.1):

    

    box1,scores1,labels1 = model1

    box2,scores2,labels2 = model2

    box3,scores3,labels3 = model3

    

    box1 = box1/1023

    box2 = box2/1023

    box3 = box3/1023

    

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion([box1,box2,box3], [scores1,scores2,scores3],[labels1,labels2,labels3],

                                                                      weights=None,iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    return boxes,scores,labels
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)



results = []



for row in range(len(all_path)):

    image_id = all_path[row].split("/")[-1].split(".")[0]

    boxes,scores,labels = run_last_wbf(w[image_id],g[image_id],l[image_id])

    boxes = (boxes*1023).astype(np.int32).clip(min=0, max=1023)

    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    result = {'image_id': image_id,'PredictionString': format_prediction_string(boxes, scores)}

    results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv',index=False)

test_df