import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torchvision

import matplotlib.pyplot as plt

import matplotlib

import matplotlib.image as im

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from PIL import Image

import albumentations as A



"""

Peak at data - note only 3.3k unique images not alot which means will have to augment the images with albumentations library

"""

train_df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")



print(len(train_df["image_id"].unique()))



train_df.head()
"""

Peak at image

"""



imgs_df=train_df["image_id"].unique()



image_id = imgs_df[45]



img = im.imread("/kaggle/input/global-wheat-detection/train/" + image_id +".jpg")



boxes = list(train_df["bbox"][ train_df["image_id"]==image_id ].values)



box=[]

for i,l in enumerate(boxes): 

    b=[float(num) for num in l[1:-1].split(",")] 

    #boxes[i]=[b[0],b[1],b[0]+b[2],b[1]+b[3]]

    #box.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])

    box.append(b)



def print_im(image,bboxes):

    

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(image)

    for c in bboxes:

        rect = matplotlib.patches.Rectangle((c[0],c[1]),c[2],c[3],linewidth=1,edgecolor='r',facecolor='none')

        ax.add_patch(rect)

    plt.show()

    

###----------------------



#format sets the format for the bounding boxes

transform = A.Compose([

    #A.RandomCrop(width=450, height=450),

    A.Resize(512, 512),

    A.VerticalFlip(p=1),

    #A.HorizontalFlip(p=1),

    A.RandomBrightnessContrast(p=0.2),

], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

#bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))



transformed = transform( image=img, bboxes=box, class_labels=["wheat"]*len(box) )

transformed_image = transformed['image']

transformed_bboxes = transformed['bboxes']



#print_im(img,box)

print_im(transformed_image,transformed_bboxes)
class ImageDataset(Dataset):

    def __init__(self, root="/kaggle/input/global-wheat-detection/", tt="train",transforms_tt=True):

        

        df=pd.read_csv("{}{}.csv".format(root,tt))

        

        self.root = root

        self.transforms = transforms

        self.imgs = df["image_id"].unique()

        self.df=df

        self.tt=tt

        self.transform=None

        if transforms_tt is True:

            self.transform=A.Compose( [  A.Resize(512, 512),

                                         A.VerticalFlip(p=0.25),

                                         A.HorizontalFlip(p=0.25),

                                         A.RandomBrightnessContrast(p=0.2)],

                                         #ToTensorV2], 

                                      bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))



    def __getitem__(self, idx):

        # load images ad masks

        image_id = self.imgs[idx]

        

        _image = im.imread("/kaggle/input/global-wheat-detection/{}/{}.jpg".format(self.tt,image_id))

        #image = Image.open( "/kaggle/input/global-wheat-detection/{}/{}.jpg".format(self.tt,image_id) )

        

        records = self.df["bbox"][self.df['image_id'] == image_id].values

        boxes=[]

        for i,l in enumerate(records): 

            b=[float(num) for num in l[1:-1].split(",")] 

            boxes.append ( b )

        

        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}

        target["labels"] = torch.ones((records.shape[0],), dtype=torch.int64)

        #target["area"] = area

        target["iscrowd"] = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        if self.transforms is not None:

            transformed = self.transform( image=_image, bboxes=boxes, class_labels=["wheat"]*len(boxes) )

            img = transformed['image']

            img=torchvision.transforms.functional.to_tensor(img)

            transformed_bboxes = transformed['bboxes']

            bboxes=[]

            for b in transformed_bboxes:

                bboxes.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])

            target["boxes"]=torch.as_tensor(bboxes, dtype=torch.float32)

            

        if self.transform is None:

            bboxes=[]

            for b in boxes:

                bboxes.append([b[0],b[1],b[0]+b[2],b[1]+b[3]])

            target["boxes"]=bboxes

            img=torchvision.transforms.functional.to_tensor(_image)

        

        del records

        del _image

        

        return img, target



    def __len__(self):

        return len(self.imgs)
dataset = ImageDataset()

data_loader = DataLoader(dataset,batch_size=50,collate_fn=lambda batch: list(zip(*batch)) )
"""

Check.

"""

dataset = ImageDataset()

data_loader = DataLoader(dataset,batch_size=50,collate_fn=lambda batch: list(zip(*batch)) )



images, targets= next(iter(data_loader))



idx=45



img= images[idx].permute(1,2,0).numpy()



print(img.shape)



fig, ax = plt.subplots(figsize=(10, 10))



ax.imshow(img)



boxes=targets[idx]["boxes"].numpy()



for c in boxes:

    rect = matplotlib.patches.Rectangle((c[0],c[1]),c[2]-c[0],c[3]-c[1],linewidth=1,edgecolor='r',facecolor='none')

    ax.add_patch(rect)





plt.show()

        
"""

Download and set up model

"""     

model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, pretrained_backbone=True)



num_classes = 2  # 1 class (wheat) + background

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from torch import optim



dataset = ImageDataset()

data_loader = DataLoader(dataset,batch_size=10,collate_fn=lambda batch: list(zip(*batch)) )



EPOCHS=16



model = model.to(device)

model.train()





params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.0008, momentum=0.9, weight_decay=0.0005)

#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,cooldown=20,factor=0.65,min_lr=0.00001,verbose=True)



print("Begin training")

lossAvg,lossPer=[],[]

for epoch in range(EPOCHS):

    total_loss,count=0,0

    for batch in data_loader:

        #check if targets is a list

        images,targets=batch

        

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        

        if count%10==0:

            #scheduler.step(losses)

            print("loss: {}".format( losses.item() ))

            lossPer.append(losses.item())

        count+=1

        total_loss+=losses.item()

    g=total_loss/count 

    lossAvg.append( g )

    print("END EPOCH #{} avg: {}".format(epoch,total_loss/count))

            

            
# print(lossA[:-10])



# plt.plot(lossA[60:])

# plt.show()

# plt.plot(lossA[600:])

# plt.show()

# plt.plot(lossA[:-600])

# plt.show()
# test_imgs=[]

# for file in os.listdir("/kaggle/input/global-wheat-detection/test/"):

#     test_imgs.append(file)



# img = im.imread("/kaggle/input/global-wheat-detection/test/{}".format(test_imgs[0]))

# img = list(img)

# torch.onnx.export(model,[img], "dml-frcnn-trained.onnx")
test_imgs=[]

for file in os.listdir("/kaggle/input/global-wheat-detection/test/"):

    test_imgs.append(file)



EPOCHS=1



model = model.to(device)

model.eval()





print("Begin testing")



predsA=[]

for image_id in test_imgs:

    

    img = im.imread("/kaggle/input/global-wheat-detection/test/{}".format(image_id))

    print( img.shape )

    img=torchvision.transforms.functional.to_tensor(img).to(device)

    

    preds = model([img])[0]["boxes"]

    

    predsA.append( preds.detach().cpu().numpy() )

    







torch.save(model.state_dict(), 'fasterRCNN_101.pth')

import matplotlib





fig, ax = plt.subplots(10,figsize=(60,60))



for i,boxes in enumerate(predsA):

    img = im.imread("/kaggle/input/global-wheat-detection/test/{}".format(test_imgs[i]))

    ax[i].imshow(img)

    for c in boxes:

        rect = matplotlib.patches.Rectangle((c[0],c[1]),c[2]-c[0],c[3]-c[1],linewidth=1,edgecolor='r',facecolor='none')

        ax[i].add_patch(rect)



plt.show()



"""

submission is space delimited

"""

def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)





detection_threshold = 0.5

results = []



predsA=[]

for image_id in test_imgs:

    

    img = im.imread("/kaggle/input/global-wheat-detection/test/{}".format(image_id))

    print( img.shape )

    img=torchvision.transforms.functional.to_tensor(img).to(device)

    

    outputs = model([img])



    boxes = outputs[0]['boxes'].data.cpu().numpy()

    scores = outputs[0]['scores'].data.cpu().numpy()



    boxes = boxes[scores >= detection_threshold].astype(np.int32)

    scores = scores[scores >= detection_threshold]



    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]



    result = {

        'image_id': image_id,

        'PredictionString': format_prediction_string(boxes, scores)

    }





    results.append(result)

        

        

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])





print( test_df.head() )





test_df.to_csv('submission.csv', index=False)

print( test_df.head() )
from IPython.display import FileLink

FileLink(r'fasterRCNN_101.pth')