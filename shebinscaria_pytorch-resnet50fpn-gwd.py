import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import PIL

from PIL import Image



import cv2



import torch

import torchvision

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision import transforms



import torch.utils.data

from torch.utils.data import Dataset, DataLoader

import random
os.listdir("/kaggle/input/global-wheat-detection")



train_dir = "/kaggle/input/global-wheat-detection/train"

test_dir = "/kaggle/input/global-wheat-detection/test"



df_train=pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
print(df_train.head())

print(df_train.shape)
print(len(df_train['image_id'].unique()))

print((df_train.shape[0])/len(df_train['image_id'].unique()))
print("Image_id v/s # of bounding boxes")

print(df_train['image_id'].value_counts())
print("Height")

print(df_train['height'].value_counts())

print("Width")

df_train['width'].value_counts()
list_image_ids_df = list(df_train['image_id'].unique())

list_image_ids_dir = os.listdir("/kaggle/input/global-wheat-detection/train")

#list_image_ids_df.sort() == list_image_ids_dir.sort()

#sorted(list_image_ids_df,key = lambda x: x,reverse=False)
len(list_image_ids_dir)- len(list_image_ids_df)
#There is a difference of 49 images i.e. there are 49 images in training directory for which there are no bounding boxes indicating no wheatheads detected
for col in df_train.columns:

    if sum(df_train[col].isnull())==1:

        print(col+" has null values")

    else:

        print(col+" no null values")
df_train['x0'] = df_train['bbox'].map(lambda x: x[1:-1].split(",")[0]).astype(float)

df_train['y0'] = df_train['bbox'].map(lambda x: x[1:-1].split(",")[1]).astype(float)

df_train['w'] = df_train['bbox'].map(lambda x: x[1:-1].split(",")[2]).astype(float)

df_train['h'] = df_train['bbox'].map(lambda x: x[1:-1].split(",")[3]).astype(float)

df_train['x1'] = df_train['x0'] + df_train['w']

df_train['y1'] = df_train['y0'] + df_train['h']
df_train.head()
for col in df_train.columns:

    if sum(df_train[col].isnull())==1:

        print(col+" has null values")

    else:

        print(col+" no null values")
df_train.dtypes
cols_to_be_selected = ['image_id','x0','y0','x1','y1']

df1_train = df_train[cols_to_be_selected]
val_percentage = 0.2

num_val_images = int(len(df1_train['image_id'].unique())*val_percentage)

num_train_images = len(df1_train['image_id'].unique()) - num_val_images

list_val_imageid = list(df1_train['image_id'].unique())[(-1)*num_val_images:]

list_train_imageid = list(df1_train['image_id'].unique())[:num_train_images]
print("Number of validation images: ",num_val_images)

print("Number of training images: ",num_train_images)

print(num_val_images + num_train_images)
df2_val = df1_train.loc[df1_train['image_id'].isin(list_val_imageid),:]

df2_train = df1_train.loc[df1_train['image_id'].isin(list_train_imageid),:]
def get_transform():

    list_transforms = []

    # converts the input image, which should be a PIL image, into a PyTorch Tensor

    list_transforms.append(transforms.ToTensor())

    

    #keeping space for augmentations in future

    

    return transforms.Compose(list_transforms)
class GlobalWheatDetectionDataset(torch.utils.data.Dataset):

    # first lets start with __init__ and initialize any objects

    def __init__(self,input_df,input_dir,transforms=None):

        

        self.df=input_df

        

        self.list_images = list(self.df['image_id'].unique())

        

        self.image_dir=input_dir

        

        self.transforms = transforms

    

    # next lets define __getitem__

    # very important to note what it returns for EACH image:

    # I. image - a PIL image of size (H,W) for ResNet50 FPN image should be scaled

    # II. target - a dictionary containing the following fields

    # A. boxes as FloatTensor of dimensions - N,4 where N = # of bounding boxs within an image 

    # and 4 columns include [x0,y0,x1,y1]

    # B. labels as Int64Tensor of dimension - N

    # C. area as Int64Tensor of dimension - N

    # D. iscrowd as UInt8Tensor of dimension - N

    # III. image_id 

    

    def __getitem__(self,idx):

        

        # II. target

        # Preparation for (A) boxes

        # FloatTensor of dimensions - N,4 where N = # of bounding boxs within an image and 4 columns include [x0,y0,x1,y1]

        

        cols_to_be_selected =['x0','y0','x1','y1']

        img_id = self.list_images[idx]

        bboxes_array = np.array(self.df.loc[self.df['image_id']==img_id,cols_to_be_selected])

        boxes = torch.tensor(bboxes_array, dtype=torch.float32)

        

        # Preparation for (B) labels

        # Int64Tensor of dimension - N

        num_boxes = self.df.loc[self.df['image_id']==img_id].shape[0]

        labels = torch.ones(num_boxes, dtype=torch.int64)

        

        # Preparation for (C) area

        # dimension - N, int64tensor

        area = torch.tensor(np.array((self.df['x1']-self.df['x0'])*(self.df['y1']-self.df['y0'])), dtype=torch.int64)

        

        # Preparation for (D) iscrowd

        # dimension - N, Uint8tensor

        iscrowd = torch.zeros(num_boxes, dtype=torch.uint8)

        

        # Combining everything

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['area'] = area

        target['iscrowd'] = iscrowd

        

        # I. Input image

        # Specifications: A.RGB format B. scaled (0,1) C. size (H,W) D. PIL format

        

        img = cv2.imread(self.image_dir+"/"+img_id+".jpg")

        img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img_scaled = img_RGB/255.0

        img_final = img_scaled

        

        if self.transforms is not None:

            img_final = self.transforms(img_final)

        

        # III. image_id

        

        

        return img_final, target, img_id

    

    # next lets define __len__    

    def __len__(self):

        

        return len(self.df['image_id'].unique())
train_dataset = GlobalWheatDetectionDataset(df2_train,train_dir,get_transform())

val_dataset = GlobalWheatDetectionDataset(df2_val,train_dir,get_transform())
def collate_fn(batch):

    return tuple(zip(*batch))
train_dataloader = DataLoader(train_dataset, batch_size=16,shuffle=False, num_workers=4,collate_fn=collate_fn)

#val_dataloader = DataLoader(val_dataset, batch_size=8,shuffle=False, num_workers=4,collate_fn=collate_fn)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)
#Showing a sample

image,targets,image_id = train_dataset[0]

# Converting the A.)images to cuda or device 

image = image.to(device)

# To see why this is needed especially in a GPU environment, try running [print(img.device) for img in images]



# Converting the B.)images to cuda or device
boxes = targets['boxes'].cpu().numpy().astype(np.int32)

sample = image.permute(1,2,0).cpu().numpy().astype(np.float32)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (255, 0, 0), 3)

    

ax.set_axis_off()

ax.imshow(sample)
# load a model; pre-trained on COCO

#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
def get_instance_objectdetection_model(num_classes,path_weight):

    # load an instance segmentation model pre-trained on COCO

    create_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False)



    # get the number of input features for the classifier

    in_features = create_model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one

    create_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    create_model.load_state_dict(torch.load(path_weight,map_location=torch.device('cpu')))



    return create_model
path_weight= "/kaggle/input/fasterrcnn/fasterrcnn_resnet50_fpn_best.pth"
num_classes = 2

# Why 2 classes - background and wheat-heads

model = get_instance_objectdetection_model(num_classes,path_weight)
torch.cuda.empty_cache()
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
mode="validation"
if mode=="training":

    model.train()

    model.to(device)



    num_epochs = 5



    itr = 1



    for epoch in range(num_epochs):

        #loss_hist.reset()

        loss_sum = 0

        num_iterations = 0

        for images, targets, image_ids in train_dataloader:



            images = list(image.to(device,dtype=torch.float) for image in images)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



            loss_dict = model(images, targets)   ##Return the loss



            losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()



            #loss_hist.send(loss_value)  #Average out the loss

            loss_sum = loss_sum + loss_value

            num_iterations = num_iterations + 1



            optimizer.zero_grad()

            losses.backward()

            optimizer.step()



            if itr % 5 == 0:

                print(f"Iteration #{itr} loss: {loss_value}")



            itr += 1



        # update the learning rate

        if lr_scheduler is not None:

            lr_scheduler.step()



        if num_iterations>0:

            loss_avg_value = loss_sum/num_iterations



        print("Epoch"+ "#"+str(epoch)+" loss: "+str(loss_avg_value))
os.listdir("/kaggle/input/gwd-customtrained-fasterrcnn-resnet-50-fpn-01")
#torch.save(model.state_dict(), '/kaggle/working/customtrained_fasterrcnn_resnet50_fpn.pth')
path_trained_weight = "/kaggle/input/gwd-customtrained-fasterrcnn-resnet-50-fpn-01/customtrained_fasterrcnn_resnet50_fpn.pth"

num_classes=2

trained_model = get_instance_objectdetection_model(num_classes,path_trained_weight)
# os.chdir("/kaggle/working/")
# from IPython.display import FileLink

# FileLink(r'customtrained_fasterrcnn_resnet50_fpn.pth')
torch.cuda.empty_cache()
val_dataloader = DataLoader(val_dataset, batch_size=8,shuffle=False, num_workers=2,collate_fn=collate_fn)
trained_model.eval()

trained_model.to(device)





images, targets, image_ids = next(iter(val_dataloader))



images = list(image.to(device,dtype=torch.float) for image in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)

sample = images[2].permute(1,2,0).cpu().numpy()



fig,ax = plt.subplots(1,1,figsize=(16,8))



for box in boxes:

    cv2.rectangle(sample, (box[0],box[1]),(box[2],box[3]),(255,0,0),3)

    ax.set_axis_off()

    ax.imshow(sample)
class GlobalWheatDetectionTestDataset(torch.utils.data.Dataset):

    # first lets start with __init__ and initialize any objects

    def __init__(self,input_df,input_dir,transforms=None):

        

        self.df=input_df

        

        self.list_images = list(self.df['image_id'].unique())

        

        self.image_dir=input_dir

        

        self.transforms = transforms

    

    # next lets define __getitem__

    # very important to note what it returns for EACH image:

    # I. image - a PIL image of size (H,W) for ResNet50 FPN image should be scaled

    

    # II. image_id 

    

    def __getitem__(self,idx):

        

        # II. image_id

        img_id = self.list_images[idx]

        # I. Input image

        # Specifications: A.RGB format B. scaled (0,1) C. size (H,W) D. PIL format

        

        img = cv2.imread(self.image_dir+"/"+img_id+".jpg")

        img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img_scaled = img_RGB/255.0

        img_final = img_scaled

        

        if self.transforms is not None:

            img_final = self.transforms(img_final)

        

        

        

        

        return img_final, img_id

    

    # next lets define __len__    

    def __len__(self):

        

        return len(self.df['image_id'].unique())
df_test=pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")
df_test.head()
# df_test['x0'] = df_test['bbox'].map(lambda x: x[1:-1].split(",")[0]).astype(float)

# df_test['y0'] = df_test['bbox'].map(lambda x: x[1:-1].split(",")[1]).astype(float)

# df_test['w'] = df_test['bbox'].map(lambda x: x[1:-1].split(",")[2]).astype(float)

# df_test['h'] = df_test['bbox'].map(lambda x: x[1:-1].split(",")[3]).astype(float)

# df_test['x1'] = df_test['x0'] + df_test['w']

# df_test['y1'] = df_test['y0'] + df_test['h']
test_dataset = GlobalWheatDetectionTestDataset(df_test,test_dir,get_transform())
test_dataloader = DataLoader(test_dataset, batch_size=8,shuffle=False, num_workers=2,collate_fn=collate_fn)
detection_threshold = 0.45
def format_prediction_string(boxes, scores): ## Define the formate for storing prediction results

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
device = torch.device('cuda')
## Lets make the prediction

results=[]

trained_model.eval()

images = []

outputs =[]

for images_, image_ids in test_dataloader:    



    images = list(image.to(device,dtype=torch.float) for image in images_)

    outputs = trained_model(images)



    for i, image in enumerate(images):



        boxes = outputs[i]['boxes'].data.cpu().numpy()    ##Formate of the output's box is [Xmin,Ymin,Xmax,Ymax]

        scores = outputs[i]['scores'].data.cpu().numpy()

        

        boxes = boxes[scores >= detection_threshold].astype(np.int32) #Compare the score of output with the threshold and

        scores = scores[scores >= detection_threshold]                    #slelect only those boxes whose score is greater

                                                                          # than threshold value

        image_id = image_ids[i]

        

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]         

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]         #Convert the box formate to [Xmin,Ymin,W,H]

        

        

            

        result = {                                     #Store the image id and boxes and scores in result dict.

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }



        

        results.append(result)              #Append the result dict to Results list



test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()
#os.chdir("/kaggle/working")
test_df.to_csv('submission.csv', index=False)
# from IPython.display import FileLink

# FileLink(r'submission_1.csv')
sample = images[1].permute(1,2,0).cpu().numpy()

boxes = outputs[1]['boxes'].data.cpu().numpy()

scores = outputs[1]['scores'].data.cpu().numpy()



boxes = boxes[scores >= detection_threshold].astype(np.int32)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 2)

    

ax.set_axis_off()

ax.imshow(sample)