import torch

import torchvision

import pandas as pd

import os

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
class wheatdataset(torch.utils.data.Dataset):

    def __init__(self,root,folder='train',transforms=None):

        self.transforms=[]

        if transforms!=None:

            self.transforms.append(transforms)

        self.root=root

        self.folder=folder

        box_data=pd.read_csv(os.path.join(root, "train.csv"))

        self.box_data=pd.concat([box_data,box_data.bbox.str.split('[').str.get(1).str.split(']').str.get(0).str.split(',',expand=True)],axis=1)

        self.imgs=list(os.listdir(os.path.join(root, self.folder)))

    def __len__(self):

        return len(self.imgs)

        

    def __getitem__(self,idx):

        img_path=os.path.join(os.path.join(self.root,self.folder),self.imgs[idx])

        img = Image.open(img_path).convert("RGB")

        df=self.box_data[self.box_data['image_id']==self.imgs[idx].split('.')[0]]

        if df.shape[0]!=0:

            df[2]=df[0].astype(float)+df[2].astype(float)

            df[3]=df[1].astype(float)+df[3].astype(float)

            boxes=df[[0,1,2,3]].astype(float).values

            labels=np.ones(len(boxes))

        else:

            boxes=np.asarray([[0,0,0,0]])

            labels=np.ones(len(boxes))

        for i in self.transforms:

            img=i(img)

            

        targets={}

        targets['boxes']=torch.from_numpy(boxes).double()

        targets['labels']=torch.from_numpy(labels).type(torch.int64)

        #targets['id']=self.imgs[idx].split('.')[0]

        return img.double(),targets
root='../input/global-wheat-detection'

dataset=wheatdataset(root,'train',transforms=torchvision.transforms.ToTensor())
dataset[0]
torch.manual_seed(1)

indices = torch.randperm(len(dataset)).tolist()

dataset_train = torch.utils.data.Subset(dataset, indices[:-2500])

dataset_test = torch.utils.data.Subset(dataset, indices[-2500:])

data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True,collate_fn=lambda x:list(zip(*x)))

data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False,collate_fn=lambda x:list(zip(*x)))
images,labels=next(iter(data_loader_train))

from matplotlib import patches

def view(images,labels,k,std=1,mean=0):

    figure = plt.figure(figsize=(30,30))

    images=list(images)

    labels=list(labels)

    for i in range(k):

        out=torchvision.utils.make_grid(images[i])

        inp=out.cpu().numpy().transpose((1,2,0))

        inp=np.array(std)*inp+np.array(mean)

        inp=np.clip(inp,0,1)  

        ax = figure.add_subplot(2,2, i + 1)

        ax.imshow(images[i].cpu().numpy().transpose((1,2,0)))

        l=labels[i]['boxes'].cpu().numpy()

        l[:,2]=l[:,2]-l[:,0]

        l[:,3]=l[:,3]-l[:,1]

        for j in range(len(l)):

            ax.add_patch(patches.Rectangle((l[j][0],l[j][1]),l[j][2],l[j][3],linewidth=2,edgecolor='w',facecolor='none')) 



view(images,labels,4)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2  # 1 class (person) + background

in_features = model.roi_heads.box_predictor.cls_score.in_features



model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model=model.to(device)





params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.01)
model.train()

from tqdm.notebook import tqdm

for epoch in tqdm(range(1)):

    for images,targets in tqdm(data_loader_train):

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model=model.double()

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        losses.backward()



        optimizer.zero_grad()

        optimizer.step()

        

    print("Loss = {:.4f} ".format(losses.item()))



torch.save(model.state_dict(), './model.pth')
model.load_state_dict(torch.load('./model.pth'))
images,targets=next(iter(data_loader_test))

images = list(image.to(device) for image in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



model.eval()

output=model(images)



with torch.no_grad():

    view(images,output,2)