

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import torch

import os

import time

import shutil

import torch.nn as nn

from skimage import io

import torchvision

import cv2

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler

from albumentations.pytorch import ToTensor

from torchvision import utils

from albumentations import (HorizontalFlip, Flip,ShiftScaleRotate, VerticalFlip, Normalize,

                            Compose, GaussNoise)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



csv_path = '/kaggle/input/global-wheat-detection/train.csv'

test_dir = '/kaggle/input/global-wheat-detection/test'

train_dir = '/kaggle/input/global-wheat-detection/train'



# Load the trained weights

weights = '/kaggle/input/fasterrcnn-using-pytorch-baseline/bestmodel.pt'
def get_transforms(phase):

            list_transforms = []

                

            list_transforms.extend(

                    [

            ToTensor(),

                    ])

            list_trfms = Compose(list_transforms)

            return list_trfms
class Wheatset(Dataset):

    def __init__(self,image_dir,phase):

        super().__init__()

   

        self.image_dir = image_dir

        self.images = os.listdir(image_dir)

        self.transforms = get_transforms(phase)

        

        

    def __len__(self):

        return len(self.images)

    

    def __getitem__(self,idx):

        image = self.images[idx]

        image_arr = io.imread(os.path.join(self.image_dir,image))

        image_id = str(image.split('.')[0])

        

        if self.transforms:

            sample = {

                'image': image_arr,

            }

            sample = self.transforms(**sample)

            image = sample['image']

               

        return image, image_id

def collate_fn(batch):

    return tuple(zip(*batch))





test_dataset = Wheatset(test_dir,phase='validation')



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
# load a model; pre-trained on COCO

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    

num_classes = 2  # 1 class (wheat) + background



# get number of input features for the classifier

in_features = model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)





checkpoint = torch.load(weights,map_location=device)

model.load_state_dict(checkpoint['state_dict'])

model.eval()

model.to(device)
detection_threshold = 0.5

results = []

for images, image_ids in test_data_loader:



    images = list(image.to(device) for image in images)

    outputs = model(images)



    

    for i, image in enumerate(images):



        boxes = outputs[i]['boxes'].data.cpu().numpy()

        scores = outputs[i]['scores'].data.cpu().numpy()

        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        scores = scores[scores >= detection_threshold]

        image_id = image_ids[i]

               

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        

        for box in boxes:

            result = {

                'image_id': 'psudo_' + image_id,

                'width': 1024,

                'height': 1024,

                'source': 'psudo',

                'bbox': [box[0],box[1],box[2],box[3]]}

        

            results.append(result)

    

    


pseudo = pd.DataFrame(results, columns=['image_id', 'width', 'height','source','bbox'])

print(f'shape before preprocessing: {pseudo.shape}')

pseudo.head()

def process_bbox(df,phase):

    ids = []

    values = []

    imd = np.unique(df['image_id'])

#     imd = [i.split('_')[1] for i in imd]

    if phase=='train':

        df['bbox'] = df['bbox'].apply(lambda x: eval(x))

    for image_id in os.listdir(train_dir):

        image_id = image_id.split('.')[0]

        if image_id not in imd :

            ids.append(image_id)

            values.append(str([-1,-1,-1,-1]))

    new_df = {'image_id':ids, 'bbox':values}

    new_df = pd.DataFrame(new_df)

    df = df[['image_id','bbox']]

    df.append(new_df)

    df = df.sample(frac=1).reset_index(drop=True)

    df['x'] = df['bbox'].apply(lambda x: x[0])

    df['y'] = df['bbox'].apply(lambda x: x[1])

    df['w'] = df['bbox'].apply(lambda x: x[2])

    df['h'] = df['bbox'].apply(lambda x: x[3])



    df.drop(columns=['bbox'],inplace=True)

    return df
df_test = process_bbox(pseudo,phase='validation')

print(f'shape of test frame is {df_test.shape}')

df_test.head()

df_train = pd.read_csv(csv_path)

df_train = process_bbox(df_train,phase='train')

print(f'shape of train frame is {df_train.shape}')

df_train.head()
# concatinating both frames

frames = [df_train, df_test]

final_df = pd.concat(frames).reset_index(drop=True)

print(f'shape of final train frame is {final_df.shape}')

final_df.tail()
# we will shuffle dataframe

final_df = final_df.sample(frac=1).reset_index(drop=True)

final_df.tail()
image_ids = final_df['image_id'].unique()

train_ids = image_ids[0:int(0.8*len(image_ids))]

val_ids = image_ids[int(0.8*len(image_ids)):]

print(f'Total images {len(image_ids)}')

print(f'No of train images {len(train_ids)}')

print(f'No of validation images {len(val_ids)}')


train_df = final_df[final_df['image_id'].isin(train_ids)]

val_df = final_df[final_df['image_id'].isin(val_ids)]

def get_transforms(phase):

            list_transforms = []

            if phase == 'train':

                list_transforms.extend([

                       Flip(p=0.5)

                         ])

            list_transforms.extend(

                    [

            ToTensor(),

                    ])

            list_trfms = Compose(list_transforms,

                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

            return list_trfms















class Wheatset(Dataset):

    def __init__(self,data_frame,train_dir,test_dir,phase='train'):

        super().__init__()

        self.df = data_frame

        self.train_dir = train_dir

        self.test_dir = test_dir

        self.images = data_frame['image_id'].unique()

        self.transforms = get_transforms(phase)

        

        

    def __len__(self):

        return len(self.images)

    

    def __getitem__(self,idx):

        image_id = self.images[idx]

        

        if 'psudo' in image_id:

            image = image_id.split('_')[1] + '.jpg'

            image_arr = cv2.imread(os.path.join(self.test_dir,image), cv2.IMREAD_COLOR)

            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32)

        else:

            image = image_id + '.jpg'

            image_arr = cv2.imread(os.path.join(self.train_dir,image), cv2.IMREAD_COLOR)

            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32)

        

        image_arr /= 255.0

#         image_id = str(image.split('.')[0])

        point = self.df[self.df['image_id'] == image_id]

        boxes = point[['x', 'y', 'w', 'h']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        area = torch.as_tensor(area, dtype=torch.float32)

        

        # there is only one class

        labels = torch.ones((point.shape[0],), dtype=torch.int64)

        

        # suppose all instances are not crowd

        iscrowd = torch.zeros((point.shape[0],), dtype=torch.int64)

        

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['image_id'] = torch.tensor(idx)

        target['area'] = area

        target['iscrowd'] = iscrowd

        

        if self.transforms:

            sample = {

                'image': image_arr,

                'bboxes': target['boxes'],

                'labels': target['labels']

            }

            sample = self.transforms(**sample)

            image = sample['image']

            

        target['boxes'] = torch.stack(tuple(map(torch.tensor, 

                                                zip(*sample['bboxes'])))).permute(1, 0)

        

        return image, target, image_id

            

train_data = Wheatset(train_df,train_dir,test_dir,phase='train')

val_data = Wheatset(val_df,train_dir,test_dir,phase='validation')



print(f'Length of train data {len(train_data)}')

print(f'Length of validation data {len(val_data)}')
# batching

def collate_fn(batch):

    return tuple(zip(*batch))



train_data_loader = DataLoader(

    train_data,

    batch_size=8,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)





valid_data_loader = DataLoader(

    val_data,

    batch_size=8,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)
def image_convert(image):

    image = image.clone().cpu().numpy()

    image = image.transpose((1,2,0))



    image = (image * 255).astype(np.uint8)

    return image



def plot_img(data,idx):

    out = data.__getitem__(idx)

    image = image_convert(out[0])

    image = np.ascontiguousarray(image)

    bb = out[1]['boxes'].numpy()

    for i in bb:

        cv2.rectangle(image, (i[0],i[1]), (i[2],i[3]), (0,255,0), thickness=2)

    plt.figure(figsize=(10,10))

    plt.imshow(image)

plot_img(train_data,129)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (wheat) + background



# get number of input features for the classifier

in_features = model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)





images, targets, ids = next(iter(train_data_loader))

images = list(image.to(device) for image in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

# optimizer = torch.optim.Adam(params, lr=0.001)

optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.00001)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

def save_ckp(state, is_best, checkpoint_path, best_model_path):

    """

    state: checkpoint we want to save

    is_best: is this the best checkpoint; min validation loss

    checkpoint_path: path to save checkpoint

    best_model_path: path to save best model

    """

    # save checkpoint data to the path given, checkpoint_path

    torch.save(state, checkpoint_path)

    # if it is a best model, min validation loss

    if is_best:

        # copy that checkpoint file to best path given, best_model_path

        shutil.copyfile(checkpoint_path, best_model_path)

        

def load_ckp(checkpoint_fpath, model, optimizer):

    """

    checkpoint_path: path to save checkpoint

    model: model that we want to load checkpoint parameters into       

    optimizer: optimizer we defined in previous training

    """

    # load check point

    checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dict from checkpoint to model

    model.load_state_dict(checkpoint['state_dict'])

    # initialize optimizer from checkpoint to optimizer

    optimizer.load_state_dict(checkpoint['optimizer'])

    # initialize valid_loss_min from checkpoint to valid_loss_min

    valid_loss_min = checkpoint['valid_loss_min']

    # return model, optimizer, epoch value, min validation loss 

    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
num_epochs = 3

train_loss_min = 0.9

total_train_loss = []





checkpoint_path = '/kaggle/working/chkpoint_'

best_model_path = '/kaggle/working/psudomodel.pt'



for epoch in range(num_epochs):

    print(f'Epoch :{epoch + 1}')

    start_time = time.time()

    train_loss = []

    model.train()

    for images, targets, image_ids in train_data_loader:

        

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        loss_dict = model(images, targets)



        losses = sum(loss for loss in loss_dict.values())

        train_loss.append(losses.item())        

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        

#     if lr_scheduler is not None:

#         lr_scheduler.step()

    #train_loss/len(train_data_loader.dataset)

    epoch_train_loss = np.mean(train_loss)

    total_train_loss.append(epoch_train_loss)

    print(f'Epoch train loss is {epoch_train_loss}')

    



    

    # create checkpoint variable and add important data

    checkpoint = {

            'epoch': epoch + 1,

            'train_loss_min': epoch_train_loss,

            'state_dict': model.state_dict(),

            'optimizer': optimizer.state_dict(),

        }

    

    # save checkpoint

    save_ckp(checkpoint, False, checkpoint_path, best_model_path)

    ## TODO: save the model if validation loss has decreased

    if epoch_train_loss <= train_loss_min:

            print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,epoch_train_loss))

            # save checkpoint as best model

            save_ckp(checkpoint, True, checkpoint_path, best_model_path)

            train_loss_min = epoch_train_loss

    

    time_elapsed = time.time() - start_time

    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
plt.title('train loss')

plt.plot(total_train_loss)

plt.xlabel('epochs')

plt.ylabel('loss')

plt.show()