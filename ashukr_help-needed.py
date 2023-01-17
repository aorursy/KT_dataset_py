!pip install pycocotools
import cv2
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



      

def get_instance_segmentation_model(num_classes):

    # load an instance segmentation model pre-trained on COCO

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)



    # get the number of input features for the classifier

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    # now get the number of input features for the mask classifier

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256

    # and replace the mask predictor with a new one

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,

                                                       hidden_layer,

                                                       num_classes)



    return model
import os

import numpy as np

import torch

import torch.utils.data

from PIL import Image

import pandas as pd

from tqdm import tqdm

import collections

from torchvision import transforms



class SevDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, df_path, transforms=None):

        #self.root = root

        self.image_dir = image_dir

        self.transforms = transforms

        self.df = pd.read_csv(df_path)

        self.df =  self.df[self.df['EncodedPixels'].notnull()]

        self.df = self.process(self.df)

    def process(self, df):

        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

        df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])



        #print(df.shape)

        return df

    def __getitem__(self, idx):

        # load images ad masks

        img_path = os.path.join(self.image_dir,self.df.iloc[idx]["ImageId"])

        img = Image.open(img_path)

        image = cv2.imread(img_path)

        rle = self.df.iloc[idx]["EncodedPixels"]

        shape = (1600,256)

        mask = self.rle2mask(rle, 256,1600)

        mask = np.array(mask)

        obj_ids = np.unique(mask)

        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)



        box=[]

        pos = np.where(mask)

        xmin = np.min(pos[1])

        xmax = np.max(pos[1])

        ymin = np.min(pos[0])

        ymax = np.max(pos[0])

        box.append([0, 0, 256, 256])

        boxes = torch.as_tensor(box, dtype=torch.float32)

        labels = torch.ones((1,), dtype=torch.int64)

        #index_lab = int(self.df.iloc[idx]['ClassId'])

        #labels[0] = index_lab

        masks = torch.as_tensor(mask, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        area = (boxes[:,3] - boxes[:, 1])*(boxes[:, 2]-boxes[:,0])

        iscrowd = torch.zeros((1,), dtype = torch.int64)

        target={}

        target["boxes"] = boxes

        target["labels"] = labels

        target["masks"]  = masks

        target["image_id"] = image_id

        target["area"] = area

        target["iscrowd"] = iscrowd

        

        if self.transforms is not None:

            img, target = self.transforms(img, target)



        return img, target

      

    def rle2mask(self,rle, width, height):

        mask= np.zeros(width* height)

        array = np.asarray([int(x) for x in rle.split()])

        starts = array[0::2]

        lengths = array[1::2]



        current_position = 0

        for index, start in enumerate(starts):

            current_position += start

            mask[current_position:current_position+lengths[index]] = 1

            current_position += lengths[index]



        return mask.reshape(width, height)



    def __len__(self):

        return (self.df.shape[0])
dataset_train = SevDataset("../input/severstal-steel-defect-detection/train_images/", "../input/severstal-steel-defect-detection/train.csv")
import math

import sys

import time

import torch



import torchvision.models.detection.mask_rcnn



#from coco_utils_py import get_coco_api_from_dataset

from coco_eval import CocoEvaluator

import utils_py





def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):

    model.train()

    metric_logger = utils_py.MetricLogger(delimiter="  ")

    metric_logger.add_meter('lr', utils_py.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)



    lr_scheduler = None

    if epoch == 0:

        warmup_factor = 1. / 1000

        warmup_iters = min(1000, len(data_loader) - 1)



        lr_scheduler = utils_py.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)



    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        loss_dict = model(images, targets)



        losses = sum(loss for loss in loss_dict.values())



        # reduce losses over all GPUs for logging purposes

        loss_dict_reduced = utils_py.reduce_dict(loss_dict)

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())



        loss_value = losses_reduced.item()



        if not math.isfinite(loss_value):

            print("Loss is {}, stopping training".format(loss_value))

            print(loss_dict_reduced)

            sys.exit(1)



        optimizer.zero_grad()

        losses.backward()

        optimizer.step()



        if lr_scheduler is not None:

            lr_scheduler.step()



        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])





def _get_iou_types(model):

    model_without_ddp = model

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):

        model_without_ddp = model.module

    iou_types = ["bbox"]

    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):

        iou_types.append("segm")

    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):

        iou_types.append("keypoints")

    return iou_types





@torch.no_grad()

def evaluate(model, data_loader, device):

    n_threads = torch.get_num_threads()

    # FIXME remove this and make paste_masks_in_image run on the GPU

    torch.set_num_threads(1)

    cpu_device = torch.device("cpu")

    model.eval()

    metric_logger = utils_py.MetricLogger(delimiter="  ")

    header = 'Test:'



    coco = get_coco_api_from_dataset(data_loader.dataset)

    iou_types = _get_iou_types(model)

    coco_evaluator = CocoEvaluator(coco, iou_types)



    for image, targets in metric_logger.log_every(data_loader, 100, header):

        image = list(img.to(device) for img in image)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        torch.cuda.synchronize()

        model_time = time.time()

        outputs = model(image)



        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        model_time = time.time() - model_time



        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()

        coco_evaluator.update(res)

        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)



    # gather the stats from all processes

    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)

    coco_evaluator.synchronize_between_processes()



    # accumulate predictions from all images

    coco_evaluator.accumulate()

    coco_evaluator.summarize()

    torch.set_num_threads(n_threads)

    return coco_evaluator
import transforms_py as T
import utils_py
def get_transform(train):

    transforms = []

    # converts the image, a PIL image, into a PyTorch Tensor

    transforms.append(T.ToTensor())

    if train:

        # during training, randomly flip the training images

        # and ground-truth for data augmentation

        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)
# use our dataset and defined transformations

# dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))

# dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

dataset_train = SevDataset("../input/severstal-steel-defect-detection/train_images/", "../input/severstal-steel-defect-detection/train.csv", get_transform(train=True))

dataset_test = SevDataset("../input/severstal-steel-defect-detection/train_images/", "../input/severstal-steel-defect-detection/train.csv", get_transform(train=False))

# split the dataset in train and test set

torch.manual_seed(1)

indices = torch.randperm(len(dataset_train)).tolist()

dataset = torch.utils.data.Subset(dataset_train, indices[:-2350])

dataset_test = torch.utils.data.Subset(dataset_test, indices[-2350:])



# define training and validation data loaders

data_loader = torch.utils.data.DataLoader(

    dataset, batch_size=2, shuffle=True, num_workers=4,

    collate_fn=utils_py.collate_fn)



data_loader_test = torch.utils.data.DataLoader(

    dataset_test, batch_size=1, shuffle=False, num_workers=4,

    collate_fn=utils_py.collate_fn)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# our dataset has two classes only - background and person

num_classes = 2



# get the model using our helper function

model = get_instance_segmentation_model(num_classes)

# move model to the right device

model.to(device)



# construct an optimizer

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005,

                            momentum=0.9, weight_decay=0.0005)



# and a learning rate scheduler which decreases the learning rate by

# 10x every 3 epochs

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,

                                               step_size=3,

                                               gamma=0.1)
# let's train it for 10 epochs

num_epochs = 1



for epoch in range(num_epochs):

    # train for one epoch, printing every 10 iterations

    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

    # update the learning rate

    lr_scheduler.step()

    # evaluate on the test dataset

    evaluate(model, data_loader_test, device=device)