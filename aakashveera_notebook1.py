import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import cv2

import warnings

warnings.filterwarnings("ignore")
import torch

import torchvision

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

from torch.utils.data import DataLoader, Dataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN
train_df = pd.read_csv("../input/global-wheat-detection/train.csv")

train_df.drop(['width','source','height'],axis=1,inplace=True)

train_df.head()
image_ids = train_df['image_id'].unique()



images = image_ids[:10]
fig,a =  plt.subplots(2,5,figsize=(20,10))

row = 0

col = 0



for n,i in enumerate(images):

    img = cv2.imread("../input/global-wheat-detection/train/"+i+".jpg")

    a[row][col].imshow(img)

    

    for j in train_df[train_df['image_id'] == i]['bbox']:

        dim = j[1:-1].split()

        dim = [int(float(x[:-1])) for x in dim]

        x,y,width,height = dim[0],dim[1],dim[2],dim[3]

        

        rect = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')

        a[row][col].add_patch(rect)

        

    col+=1    

    if n==4:

        col = 0

        row = 1

    
train_df['x'] = train_df['bbox'].apply(lambda x: int(float(x[1:-1].split(',')[0])))

train_df['y'] = train_df['bbox'].apply(lambda x: int(float(x[1:-1].split(',')[1])))

train_df['width'] = train_df['bbox'].apply(lambda x: int(float(x[1:-1].split(',')[2])))

train_df['height'] = train_df['bbox'].apply(lambda x: int(float(x[1:-1].split(',')[3])))
train_df.drop('bbox',axis=1,inplace=True)
train_df
transform =  A.Compose([

        ToTensorV2(p=1.0)

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
transform =  A.Compose([

            A.OneOf([

                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 

                                     val_shift_limit=0.2, p=0.8),

                A.RandomBrightnessContrast(brightness_limit=0.2, 

                                           contrast_limit=0.2, p=0.8),],p=0.5),

            A.ToGray(p=0.1),

            A.Cutout(num_holes=4, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),

            ToTensorV2(p=1.0)

        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
class WheatDataset(Dataset):



    def __init__(self, dataframe, transforms):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]



        image = cv2.imread("../input/global-wheat-detection/train/"+image_id+".jpg", cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        boxes = records[['x', 'y', 'width', 'height']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        area = torch.as_tensor(area, dtype=torch.float32)



        labels = torch.ones((records.shape[0],), dtype=torch.int64)       

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['iscrowd'] = iscrowd

        target['area'] = area

        target['image_id'] = torch.tensor([index])



        sample = {

                'image': image,

                'bboxes': target['boxes'],

                'labels': labels

            }

        sample = self.transforms(**sample)

        image = sample['image']



        target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
train_dataset = WheatDataset(train_df, transform)
def collate_fn(batch):

    return tuple(zip(*batch))



train_data_loader = DataLoader(

    train_dataset,

    batch_size=16,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.roi_heads.box_predictor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
class Averager:

    def __init__(self):

        self.current_total = 0.0

        self.iterations = 0.0



    def send(self, value):

        self.current_total += value

        self.iterations += 1



    @property

    def value(self):

        if self.iterations == 0:

            return 0

        else:

            return 1.0 * self.current_total / self.iterations



    def reset(self):

        self.current_total = 0.0

        self.iterations = 0.0
loss_hist = Averager()



for epoch in range(30):

    loss_hist.reset()

    

    for n,(images, targets, image_ids) in enumerate(train_data_loader):

        n+=1

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        loss_dict = model(images, targets)



        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()



        loss_hist.send(loss_value)



        optimizer.zero_grad()

        losses.backward()

        optimizer.step()



            

    print(f"Epoch : "+str(epoch)+" loss: " +str(loss_hist.value))   
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pt')