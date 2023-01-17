import os

import numpy as np

from matplotlib import pyplot as plt

import cv2

from PIL import Image

import torch

import torch.utils.data

import torchvision.transforms as T

from torchvision.transforms import functional as F

import random 

import base64

import json

from torch.utils.data import DataLoader, random_split,Subset

import pickle
root_path = '../input/airport-dataset/AirportDataset/开舱门'
class HatchDataset(torch.utils.data.Dataset):

  def __init__(self, root, transforms=None):

    self.dataset = []

    self.transforms=transforms

    dir_list = list(sorted(os.listdir(root)))

    for dir_name in dir_list:

      dir_path = os.path.join(root, dir_name)

      # print(dir_path)

      for file_name in list(sorted(os.listdir(dir_path))):

        if(file_name.endswith('.json')):

          file_path = os.path.join(dir_path,file_name)

          with open(file_path, 'rb') as f:

            one_data = pickle.load(f)

            one_data['image']  = cv2.imdecode(one_data['image'], cv2.IMREAD_COLOR) 

            one_data['image'] = cv2.cvtColor(one_data['image'], cv2.COLOR_BGR2RGB)

            

            ##  获得翻转副本

            f_image = one_data['image'].copy()

            f_target = {k:v.copy() for k,v in one_data['target'].items()}

            f_image,f_target  = self.flip(f_image,f_target)

            f_data = {'image':f_image,'target':f_target}

           

            ### 加入

            self.dataset.append(one_data)

            self.dataset.append(f_data)

          

  

  def __getitem__(self, index):

    item = self.dataset[index]

    ## 这个要放在这里转化，因为所在空间是翻了20倍，内存吃不消

    

    image = item['image'].copy()

    target = {k:v.copy() for k,v in item['target'].items()} # 要搞个副本，避免原始数据遭到修改

    ## 对数据的随机处理要放在深复制后面，避免原始数据被修改

    image = F.to_tensor(image)

    target = {

        'boxes':torch.as_tensor(target['boxes'], dtype=torch.float32),

        'labels':torch.as_tensor([True]*len(target['labels']),dtype=torch.int64)

    }

    

    return image,target





  ##  翻转图片

  def flip(self,image, target):

    boxes = target['boxes']

    width = image.shape[1]

    image = F.hflip(Image.fromarray(image) )

    boxes[:, 0::2] = width - boxes[:, 2::-2]

    return image, target



  def __len__(self):

    return len(self.dataset)
# dataset = HatchDataset(root_path)

# n_val = int(len(dataset) * 0.2)

# n_train = len(dataset) - n_val

# dataset_train, dataset_test = random_split(dataset, [n_train, n_val])

# print(len(dataset),dataset_train.__len__(),dataset_test.__len__())
dataset = HatchDataset(root_path)

n_train = int(len(dataset) * 0.8)

dataset_train = Subset(dataset,range(n_train))

dataset_test = Subset(dataset,range(n_train,len(dataset))) 

print(len(dataset),dataset_train.__len__(),dataset_test.__len__())
def collate_fn(batch):

  return tuple(zip(*batch)) #这里的意思估计是对batch解压，再转成元组，长度为二。元组的元组

  

train_loader = torch.utils.data.DataLoader(

    dataset_train, batch_size=1, shuffle=True, num_workers=4, # batch_size不能大，用pin_memory也会内存爆炸

    collate_fn=collate_fn)



test_loader = torch.utils.data.DataLoader(

    dataset_test, batch_size=1, shuffle=False, num_workers=4,pin_memory=True,

    collate_fn=collate_fn)
def validate():

  loss = 0

  cnt = 0

  with torch.no_grad():

    for images, targets in test_loader:

      # images, targets = images.to(device), targets.to(device)

      images = list(image.to(device) for image in images)

      targets = [{k:v.to(device) for k,v in t.items()} for t in targets]



      loss_dict = model(images, targets)

      loss += sum(loss for loss in loss_dict.values())

      cnt += len(images)

  loss = loss/cnt

  print('--Loss of the network on the test images: %.3f\n' % loss)

  return loss
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



num_classes = 2



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)



params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.00005,momentum=0.9,weight_decay=0.0005)



lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.8)
num_epochs = 20

train_list = []

valid_list = []

for epoch in range(num_epochs): 

  itr_loss = 0.0

  epo_loss = 0.0

  accumulation_steps = 16

  itr = 0

  optimizer.zero_grad()

  for i,(images,targets) in enumerate(train_loader):

    images = list(image.to(device) for image in images)

    targets = [{key:val.to(device) for key,val in target.items()} for target in targets]



    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())

    losses.backward()  

    

    itr_loss += losses.item() 

    epo_loss += losses.item()

    itr = itr+len(images)

    if (itr%accumulation_steps) == 0: 

      optimizer.step() 

      optimizer.zero_grad()

      print("Iteration %d train_loss:%.03f"%(itr,itr_loss/accumulation_steps))

      itr_loss = 0   



  optimizer.step() 

  optimizer.zero_grad()

  if lr_scheduler is not None:

      lr_scheduler.step()

  epo_loss = epo_loss/itr

  train_list.append(epo_loss)

  print('--Epoch %d finished, train_loss:%.3f lr=%f'%(epoch,epo_loss,lr_scheduler.get_last_lr()[0])) 

  ret = validate()

  valid_list.append(ret)
epochs = range(1+1, len(train_list) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, train_list[1:], 'r', label='Training loss')

# b is for "solid blue line"x

plt.plot(epochs, valid_list[1:], 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
class NMSTool:

    

    def __init__(self,

             nms_iou_threshold=0.5

        ):

        self.nms_iou_threshold = nms_iou_threshold

    

    

    # NMS算法

    # bboxes维度为[N,4]，scores维度为[N,], 均为tensor

    def nms(self,prediction):

        

        boxes = prediction['boxes']

        scores = prediction['scores']

        labels = prediction['labels']

        

        if(len(scores)==0):

            return []



        x1 = boxes[:,0]

        y1 = boxes[:,1]

        x2 = boxes[:,2]

        y2 = boxes[:,3]

        areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积

        _, order = scores.sort(0, descending=True)    # 降序排列



        keep = []

        while order.numel() > 0:       # torch.numel()返回张量元素个数

            if order.numel() == 1:     # 保留框只剩一个

                i = order.item()

                keep.append(i)

                break

            else:

                i = order[0].item()    # 保留scores最大的那个框box[i]

                keep.append(i)



            # 计算box[i]与其余各框的IOU(思路很好)

            xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]

            yy1 = y1[order[1:]].clamp(min=y1[i])

            xx2 = x2[order[1:]].clamp(max=x2[i])

            yy2 = y2[order[1:]].clamp(max=y2[i])

            inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]



            iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]

            idx = (iou <= self.nms_iou_threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]

            if idx.numel() == 0:

                break

            order = order[idx+1]  # 修补索引之间的差值

        

        prediction['boxes'] = prediction['boxes'][keep]

        prediction['scores'] = prediction['scores'][keep]

        prediction['labels'] = prediction['labels'][keep]

        return prediction
nms_tool = NMSTool(0.01)
model.eval()

score_threshold = 0.3

with torch.no_grad():

  for images, targets in test_loader:

    image = images[0]

    img = image.clone()

    image = (image*255).permute([1,2,0]).numpy().astype(np.uint8).copy()

    prediction = model([img.to(device)])[0]

    

    prediction = nms_tool.nms(prediction)

    # print(len(prediction['boxes']))

    for i in range(len(prediction['boxes'])):

      box = prediction['boxes'][i]

      score = prediction['scores'][i]

      if(score<score_threshold):

        continue

      cv2.rectangle(image,(box[0], box[1]),(box[2], box[3]),(220, 0, 0), 2)



    plt.figure(figsize=(8,8),dpi=150)

    plt.subplot(111)

    plt.imshow(image)

    plt.show()
torch.save(model, 'open_hatch.pkl')