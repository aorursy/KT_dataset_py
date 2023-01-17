DIR_INPUT = '/kaggle/input/pennfudanped'
import os

import numpy as np

from matplotlib import pyplot as plt

import cv2

from PIL import Image

import torch

import torch.utils.data

from torchvision.transforms import functional as F

import torchvision.transforms as transforms
class PennFudanDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):

        self.root = root

        # 转换器

        self.transforms = transforms

        # load all image files, sorting them to

        # ensure that they are aligned

        # 获取所有文件路径，并排好序

        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))

        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))



    def __getitem__(self, idx):

        # load images ad masks

        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])

        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])



        img = Image.open(img_path).convert("RGB")

        ###### 这句自己加的  #####

        img = np.array(img)



        # note that we haven't converted the mask to RGB,

        # because each color corresponds to a different instance

        # with 0 being background

        mask = Image.open(mask_path)

        # 转换成numpy的二维数组

        mask = np.array(mask)



        # instances are encoded as different colors

        # unique()的作用大概是求mask二维数组中有多少个相异的值，这里应该是背景+行人数

        obj_ids = np.unique(mask)

        # first id is the background, so remove it

        # 0是背景，所以从1开始

        obj_ids = obj_ids[1:]



        # split the color-encoded mask into a set

        # of binary masks

        # 这个好像是单独获取出每个行人的mask？

        masks = mask == obj_ids[:, None, None]



        # get bounding box coordinates for each mask

        # 对于每个行人的mask，求出矩形边框

        num_objs = len(obj_ids)

        boxes = []

        for i in range(num_objs):

            pos = np.where(masks[i])  # np.where(condition)满足condition，返回索引

            xmin = np.min(pos[1])

            xmax = np.max(pos[1])

            ymin = np.min(pos[0])

            ymax = np.max(pos[0])

            boxes.append([xmin, ymin, xmax, ymax])



        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class

        labels = torch.ones((num_objs,), dtype=torch.int64)

        # masks = torch.as_tensor(masks, dtype=torch.uint8)



        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)



        target = {}

        target["boxes"] = boxes

        target["labels"] = labels

        target["image_id"] = image_id

        target["area"] = area

        target["iscrowd"] = iscrowd



        # 这个transforms究竟是怎么回事？

        if self.transforms is not None:

            img = self.transforms(img,target)

        else: # pytorch的图像要求是：tensor类型、(通道，长，宽)、取值[0,1],用这个to_tensor函数刚刚好

            img = F.to_tensor(img)





        return img, target



    def __len__(self):

        return len(self.imgs)
# use our dataset and defined transformations

dataset = PennFudanDataset(f'{DIR_INPUT}/PennFudanPed')

dataset_test = PennFudanDataset(f'{DIR_INPUT}/PennFudanPed')



# split the dataset in train and test set

torch.manual_seed(1)

indices = torch.randperm(len(dataset)).tolist()

dataset = torch.utils.data.Subset(dataset, indices[:-50])

dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
print(dataset.__len__(),dataset_test.__len__())
# >>>a = [1,2,3]

# >>> b = [4,5,6]

# >>> c = [4,5,6,7,8]

# >>> zipped = zip(a,b)     # 打包为元组的列表

# [(1, 4), (2, 5), (3, 6)]

# >>> zip(a,c)              # 元素个数与最短的列表一致

# [(1, 4), (2, 5), (3, 6)]

# >>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式

# [(1, 2, 3), (4, 5, 6)]

def collate_fn(batch):

  return tuple(zip(*batch)) #这里的意思估计是对batch解压，再转成元组，长度为二。元组的元组

    

# define training and validation data loaders

data_loader = torch.utils.data.DataLoader(

    dataset, batch_size=2, shuffle=True, num_workers=4,

    collate_fn=collate_fn)



data_loader_test = torch.utils.data.DataLoader(

    dataset_test, batch_size=16, shuffle=False, num_workers=4,

    collate_fn=collate_fn)
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



# our dataset has two classes only - background and person

num_classes = 2

# get the model using our helper function



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)





# move model to the right device

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)



# 大疑问：损失函数是什么？

# construct an optimizer

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)



# and a learning rate scheduler which decreases the learning rate by

# 10x every 3 epochs

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
num_epochs = 10



for epoch in range(num_epochs): 

    sum_loss = 0.0

    cal_freq = 10

    itr = 1

    for images, targets in data_loader: # 我猜就是一下子给batch_size个数据，迭代size/batch_size次



        # 为什么要做这两步工作？ 为了.to(device)，转移到cuda:0上

        # 获取图像本体和信息

        images = list(image.to(device) for image in images)

        # 是一个list，每个元素是一个dict，对应一张image

        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]



        # 计算损失值

        # 疑问1：损失函数是什么？在哪里设定？

        # 疑问2：输入图像的大小？在哪里设定？

        loss_dict = model(images, targets)



        # 统计损失值之和，并梯度下降

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad() # ？？？

        losses.backward()  #反向传播

        optimizer.step() # ？？？



        # 用于输出训练过程 

        sum_loss += losses.item() # 损失值累加

        if itr % cal_freq == 0:

            print("Iteration %d loss:%.03f"%(itr,sum_loss/cal_freq))

            sum_loss = 0

        itr += 1



    # update the learning rate

    if lr_scheduler is not None:

        lr_scheduler.step()

    print('--------------Epoch %d finished'%epoch) 
# pick one image from the test set

images, targets = next(iter(data_loader_test))
img = images[12]

image = np.array(img.mul(255).permute(1, 2, 0)).astype(np.uint8)

# put the model in evaluation mode

model.eval()

with torch.no_grad():

    prediction = model([img.to(device)])[0]

    for box in prediction['boxes']:

        cv2.rectangle(image,(box[0], box[1]),(box[2], box[3]),(220, 0, 0), 3)

Image.fromarray(image)