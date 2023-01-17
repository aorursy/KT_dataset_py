import os

from PIL import Image

import numpy as np

import pandas as pd

import torch

from tqdm import tqdm_notebook as tqdm
TEMP_DIR = './tmp/'

BASE_DIR = '../input/openimage2019segmentationsubset2000/subset2000/'

TEST_DIR = '../input/open-images-2019-instance-segmentation/test/'



with open(BASE_DIR+"classes-segmentation.txt") as f:

    CLASSES = [c.strip() for c in f.readlines()]

CLASSES = ["__background__"] + CLASSES

NUM_CLASS = len(CLASSES)

CLOP_SIZE = 480

BASE_SIZE = 520

NUM_CROP = 1
BATCH_SIZE = 2

NUM_WORKERS = 3

NUM_EPOCHS = 2

NUM_GPUS = 1

# Use all data: 0-f, z is subset2k

USING_SPLITS = ["z"] #["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"]
if not os.path.isdir(TEMP_DIR):

    os.mkdir(TEMP_DIR)

    os.mkdir(TEMP_DIR+"join-masks")

    os.mkdir(TEMP_DIR+"output-images")
def _mask_filepart_classname(name):

    if name.startswith("m"):

        return "/m/" + name[1:]

    return name
import math

def _create_resize_image(img, ismask=False, tosize=None):

    long_side_size = BASE_SIZE * 2

    if img.height < img.width:

        scale = img.width / long_side_size

        size = (long_side_size, max(BASE_SIZE,math.ceil(img.height / scale)))

    else:

        scale = img.height / long_side_size

        size = (max(BASE_SIZE,math.ceil(img.width / scale)), long_side_size)

    return img.resize(size if tosize is None else tosize, Image.NEAREST if ismask else Image.BILINEAR)
import pickle

def _make_openimage2019_mask(split_name):

    img_paths = []

    mask_paths = []

    img_folder = os.path.join(BASE_DIR, 'train-images-'+split_name)

    mask_folder = os.path.join(BASE_DIR, 'mask-images-'+split_name)

    join_folder = os.path.join(TEMP_DIR, 'join-masks')

    img_folder_list = sorted(list(os.listdir(img_folder)))

    image_mask = {}

    for filename in os.listdir(mask_folder):

        basename, _ = os.path.splitext(filename)

        maskname = basename.split("_")

        if filename.endswith(".png"):

            imgpath = os.path.join(img_folder, filename)

            imagename = maskname[0] + '.jpg'

            imagepath = os.path.join(img_folder, imagename)

            if os.path.isfile(imagepath):

                if imagepath not in image_mask:

                    image_mask[imagename] = [filename]

                else:

                    image_mask[imagename].append(filename)

            else:

                print('cannot find the image:', imagepath)



    for imagename, masknames in tqdm(image_mask.items()):

        for nc in range(NUM_CROP):

            imgpath = os.path.join(img_folder, imagename)

            basename, _ = os.path.splitext(imagename)

            joinpath = os.path.join(join_folder, basename+"-"+str(nc)+".pkl")

            if os.path.isfile(joinpath):

                continue



            img_rs = _create_resize_image(Image.open(imgpath)).convert('RGB')

            

            crop_x = np.random.randint(img_rs.width-CLOP_SIZE)

            crop_y = np.random.randint(img_rs.height-CLOP_SIZE)

            img = img_rs.crop((crop_x, crop_y, crop_x+CLOP_SIZE, crop_y+CLOP_SIZE))



            boxes = []

            masks = []

            labels = []



            for filename in masknames:

                basename, _ = os.path.splitext(filename)

                maskname = basename.split("_")

                maskpath = os.path.join(mask_folder, filename)

                maskflag = _create_resize_image(Image.open(maskpath), ismask=True, tosize=(img_rs.width,img_rs.height))

                maskflag = maskflag.crop((crop_x, crop_y, crop_x+CLOP_SIZE, crop_y+CLOP_SIZE))

                maskflag = np.array(maskflag.convert('1'))

                maskclass = _mask_filepart_classname(maskname[1])

                if np.sum(maskflag) > 0 and maskclass in CLASSES:

                    labels.append(CLASSES.index(maskclass))

                    pos = np.where(maskflag)

                    xmin = np.min(pos[1])

                    xmax = np.max(pos[1])

                    ymin = np.min(pos[0])

                    ymax = np.max(pos[0])

                    boxes.append([xmin, ymin, xmax, ymax])

                    masks.append(maskflag)



            if len(boxes) > 0:



                boxes = np.array(boxes)

                masks = np.array(masks)

                labels = np.array(labels)



                idx = 0

                if imagename in img_folder_list:

                    idx = img_folder_list.index(imagename)

                image_id = [idx]



                if boxes.shape[0] == 0:

                    area = 0

                else:

                    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])



                iscrowd = np.zeros((len(boxes),), dtype=np.int64)



                target = {}

                target["boxes"] = boxes

                target["labels"] = labels

                target["masks"] = masks

                target["image_id"] = image_id

                target["area"] = area

                target["iscrowd"] = iscrowd



                imgf = np.array(img, dtype=np.float32).transpose(2,0,1)



                with open(joinpath, 'wb') as f:

                    pickle.dump((imgf,target), f)
for z in USING_SPLITS:

    _make_openimage2019_mask(z)
join_folder = os.path.join(TEMP_DIR, 'join-masks')

join_files = sorted([f for f in os.listdir(join_folder) if f.endswith(".pkl")])
class MyDataset(object):

    def __init__(self):

        self.join_folder = os.path.join(TEMP_DIR, 'join-masks')

        self.files = sorted([f for f in os.listdir(self.join_folder) if f.endswith(".pkl")])



    def __getitem__(self, idx):

        mask_path = os.path.join(self.join_folder, self.files[idx])

        with open(mask_path, 'rb') as f:

            imgf,target = pickle.load(f)

        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)

        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)

        target["masks"] = torch.as_tensor(target["masks"], dtype=torch.uint8)

        target["image_id"] = torch.as_tensor(target["image_id"], dtype=torch.int32)

        target["area"] = torch.as_tensor(target["area"], dtype=torch.float32)

        target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.int64)



        imgf = torch.as_tensor(imgf, dtype=torch.float32)

 

        return imgf, target["boxes"], target["labels"], target["masks"], target["image_id"], target["area"], target["iscrowd"]



    def __len__(self):

        return len(self.files)
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



rcnnmodel = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=NUM_CLASS, pretrained_backbone=True)

rcnnmodel.eval()

model = torch.nn.DataParallel(rcnnmodel)

model.cuda()
import torchvision.transforms as T



def get_transform(train):

    transforms = []

    transforms.append(T.ToTensor())

    if train:

        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)
import math

import sys

import time

import torch



def train_one_epoch(model, optimizer, data_loader, device, epoch):

    model.train()



    lr_scheduler = None

    if epoch == 0:

        warmup_factor = 1. / 1000

        warmup_iters = min(1000, len(data_loader) - 1)



    prog = tqdm(data_loader, total=len(data_loader))

    for imgfs, boxes, labels, masks, image_id, area, iscrowd in prog:

        images = []

        targets = []



        for i in range(imgfs.shape[0]):

            images.append(imgfs[i].cuda())

            target = {}

            target["boxes"] = boxes[i].cuda()

            target["labels"] = labels[i].cuda()

            target["masks"] = masks[i].cuda()

            target["image_id"] = image_id[i].cuda()

            target["area"] = area[i].cuda()

            target["iscrowd"] = iscrowd[i].cuda()

            targets.append(target)



        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())



        prog.set_description("loss:%03f"%losses)

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

device = torch.device('cuda')



dataset = MyDataset()



data_loader = torch.utils.data.DataLoader(

    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)



params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params)



for epoch in range(NUM_EPOCHS):

    train_one_epoch(model, optimizer, data_loader, device, epoch)

    torch.save(rcnnmodel.state_dict(), "checkpoint-%d"%epoch)
torch.save(rcnnmodel.state_dict(), "final_model")
import shutil

shutil.rmtree(TEMP_DIR)