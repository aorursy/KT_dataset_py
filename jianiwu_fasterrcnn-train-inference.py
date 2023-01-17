import pandas as pd

import numpy as np

import cv2

import os

import re



from PIL import Image



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator



from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler



from matplotlib import pyplot as plt



Test = True
# from tensorboardX import SummaryWriter

# writer = SummaryWriter('log')
# DIR_INPUT = '/kaggle/input/global-wheat-detection'

# DIR_TRAIN = f'{DIR_INPUT}/train'

# DIR_TEST = f'{DIR_INPUT}/test'

# train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

# train_df.shape
# train_df['x'] = -1

# train_df['y'] = -1

# train_df['w'] = -1

# train_df['h'] = -1



# def expand_bbox(x):

#     r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))

#     if len(r) == 0:

#         r = [-1, -1, -1, -1]

#     return r



# train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

# train_df.drop(columns=['bbox'], inplace=True)

# train_df['x'] = train_df['x'].astype(np.float)

# train_df['y'] = train_df['y'].astype(np.float)

# train_df['w'] = train_df['w'].astype(np.float)

# train_df['h'] = train_df['h'].astype(np.float)



# image_ids = train_df['image_id'].unique()

# valid_ids = image_ids[-665:]

# train_ids = image_ids[:-665]



# valid_df = train_df[train_df['image_id'].isin(valid_ids)]

# train_df = train_df[train_df['image_id'].isin(train_ids)]
# train_df
# skpike_train_images = '../input/wheat-data-spide/images/train'

# skpike_train_labels = '../input/wheat-data-spide/labels/train'

# we_images = os.listdir(skpike_train_images)

# we_labels = os.listdir(skpike_train_labels)

# we_images = np.sort(we_images)

# we_labels = np.sort(we_labels)

# spike_df_train = None

# for (lab, img) in zip(we_labels, we_images):

#     df = pd.read_csv(skpike_train_labels + '/' + lab, sep=' ', header=None)

#     df.columns = ['class', 'x', 'y', 'w', 'h']

#     df['x'] = (1024 * df['x'])

#     df['y'] = np.ceil(1024 * df['y'])

#     df['w'] = np.floor(1024 * df['w'])

#     df['h'] = np.floor(1024 * df['h'])

#     df['x'] = np.ceil(df['x'] - df['w']/2 - 1)

#     df['y'] = np.ceil(df['y'] - df['h']/2 - 1)

    

#     df['x'] = df['x'].clip(0.1, 1023)

#     df['y'] = df['y'].clip(0.1, 1023)

#     keep_idx = df['w'] > 1

#     df = df[keep_idx]

#     keep_idx = df['h'] > 1

#     df = df[keep_idx]

    

    

    

#     df['image_id'] = img.split('.')[0]

#     df['base_path'] = '../input/spike-dataset/images/train/'

#     df['width'] = 1024

#     df['height'] = 1024

#     df['source'] = 'spike'

#     df = df.drop(['class'], axis=1)

#     df = df[['image_id', 'width', 'height', 'source', 'x', 'y', 'w', 'h', 'base_path']]

    

    

#     #print ( lab, img)

#     if spike_df_train is None:

#         spike_df_train = df.copy()

#     else:

#         spike_df_train = pd.concat((spike_df_train, df))

        

# spike_df_train.head()    
# skpike_valid_images = '../input/wheat-data-spide/images/valid'

# skpike_valid_labels = '../input/wheat-data-spide/labels/valid'

# we_images = os.listdir(skpike_valid_images)

# we_labels = os.listdir(skpike_valid_labels)

# we_images = np.sort(we_images)

# we_labels = np.sort(we_labels)

# spike_df_valid = None

# for (lab, img) in zip(we_labels, we_images):

#     df = pd.read_csv(skpike_valid_labels + '/' + lab, sep=' ', header=None)

#     df.columns = ['class', 'x', 'y', 'w', 'h']

#     df['x'] = (1024 * df['x'])

#     df['y'] = np.ceil(1024 * df['y'])

#     df['w'] = np.floor(1024 * df['w'])

#     df['h'] = np.floor(1024 * df['h'])

#     df['x'] = np.ceil(df['x'] - df['w']/2 - 1)

#     df['y'] = np.ceil(df['y'] - df['h']/2 - 1)

    

#     df['x'] = df['x'].clip(0.1, 1023)

#     df['y'] = df['y'].clip(0.1, 1023)

#     keep_idx = df['w'] > 1

#     df = df[keep_idx]

#     keep_idx = df['h'] > 1

#     df = df[keep_idx]

    

    

    

#     df['image_id'] = img.split('.')[0]

#     df['base_path'] = '../input/spike-dataset/images/valid/'

#     df['width'] = 1024

#     df['height'] = 1024

#     df['source'] = 'spike'

#     df = df.drop(['class'], axis=1)

#     df = df[['image_id', 'width', 'height', 'source', 'x', 'y', 'w', 'h', 'base_path']]

    

    

#     #print ( lab, img)

#     if spike_df_valid is None:

#         spike_df_valid = df.copy()

#     else:

#         spike_df_valid = pd.concat((spike_df_valid, df))

        

# spike_df_valid.head()    
# '''

# 预览图像

# '''

# image = cv2.imread('../input/wheat-data-spide/images/train/Spike_0001_0_0.jpg')

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

# image /= 255.0

# fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# ax.imshow(image)
class WheatDataset(Dataset):



    def __init__(self, dataframe, image_dir, transforms=None):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]



        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        boxes = records[['x', 'y', 'w', 'h']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        area = torch.as_tensor(area, dtype=torch.float32)



        # there is only one class

        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        

        # suppose all instances are not crowd

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        # target['masks'] = None

        target['image_id'] = torch.tensor([index])

        target['area'] = area

        target['iscrowd'] = iscrowd



        if self.transforms:

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
# Albumentations

def get_train_transform():

    return A.Compose([

        A.Flip(0.5),

        ToTensorV2(p=1.0)

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



def get_valid_transform():

    return A.Compose([

        ToTensorV2(p=1.0)

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



# anchor_sizes = ((8,), (16,), (32,), (64,), (128,))

# aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

# rpn_anchor_generator = AnchorGenerator(

#     anchor_sizes, aspect_ratios

# )



# load a model; pre-trained on COCO

if Test:

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False)

#                                                                  rpn_anchor_generator=rpn_anchor_generator)

else:

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # 1 class (wheat) + background



# get number of input features for the classifier

in_features = model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
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

# def collate_fn(batch):

#     return tuple(zip(*batch))



# train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())

# valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())

# # skpike_train_images为图片路径

# spike_train_dataset = WheatDataset(spike_df_train, skpike_train_images, get_train_transform())

# spike_valid_dataset = WheatDataset(spike_df_valid, skpike_valid_images, get_valid_transform())



# # split the dataset in train and test set

# indices = torch.randperm(len(train_dataset)).tolist()



# train_data_loader = DataLoader(

#     train_dataset,

#     batch_size=8,

#     shuffle=False,

#     num_workers=4,

#     collate_fn=collate_fn

# )



# valid_data_loader = DataLoader(

#     valid_dataset,

#     batch_size=8,

#     shuffle=False,

#     num_workers=4,

#     collate_fn=collate_fn

# )



# spike_train_data_loader = DataLoader(

#     spike_train_dataset,

#     batch_size=8,

#     shuffle=False,

#     num_workers=4,

#     collate_fn=collate_fn

# )



# spike_valid_data_loader = DataLoader(

#     spike_valid_dataset,

#     batch_size=8,

#     shuffle=False,

#     num_workers=4,

#     collate_fn=collate_fn

# )
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# [i for i in model.state_dict()]

# param_list = [(name,param.requires_grad) for name, param in model.named_parameters()]

# param_list
# i = 0

# for name, param in model.named_parameters():

# #     23最好，69最差

#     if i>=23:

#         param.requires_grad = True

#     else:

#         param.requires_grad = False

#     i+=1
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

lr_scheduler = None



num_epochs = 4
# loss_hist = Averager()

# itr = 1

# best_loss = 100000



# for epoch in range(num_epochs):

#     loss_hist.reset()

    

#     for images, targets, image_ids in train_data_loader:

        

#         images = list(image.to(device) for image in images)

#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



#         loss_dict = model(images, targets)



#         losses = sum(loss for loss in loss_dict.values())

#         loss_value = losses.item()



#         loss_hist.send(loss_value)



#         optimizer.zero_grad()

#         losses.backward()

#         optimizer.step()



    

#         if itr % 50 == 0:

#             print(f"Iteration #{itr} loss: {loss_value}")



#         itr += 1

        

#     for images, targets, image_ids in spike_train_data_loader:

        

#         images = list(image.to(device) for image in images)

#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



#         loss_dict = model(images, targets)



#         losses = sum(loss for loss in loss_dict.values())

#         loss_value = losses.item()



#         loss_hist.send(loss_value)



#         optimizer.zero_grad()

#         losses.backward()

#         optimizer.step()



    

#         if itr % 50 == 0:

#             print(f"spike Iteration #{itr} loss: {loss_value}")



#         itr += 1

        

#     if best_loss > loss_hist.value:

#         best_loss = loss_hist.value

#         torch.save(model.state_dict(), 'fasterrcnn_V4_epoch4.pth')    

# #     torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')

    

#     # update the learning rate

#     if lr_scheduler is not None:

#         lr_scheduler.step()



#     print(f"Epoch #{epoch} loss: {loss_hist.value}")   
def collate_fn(batch):

    return tuple(zip(*batch))





class WheatTestDataset(Dataset):



    def __init__(self, dataframe, image_dir, transforms=None):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]



        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        if self.transforms:

            sample = {

                'image': image,

            }

            sample = self.transforms(**sample)

            image = sample['image']



        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]

DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TEST = f'{DIR_INPUT}/test'

# DIR_WEIGHTS = '/kaggle/input/global-wheat-detection-public'

# WEIGHTS_FILE = f'{DIR_WEIGHTS}/fasterrcnn_resnet50_fpn.pth'

# WEIGHTS_FILE = '../input/weithght-v1-ep4/fasterrcnn_resnet50_fpn_epoch30.pth'

WEIGHTS_FILE = '../input/weithght-v1-ep4/fasterrcnn_V4_epoch4.pth'

test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')



# model加载最好的参数

model.load_state_dict(torch.load(WEIGHTS_FILE,map_location=torch.device(device)))

model.eval()



# Albumentations

def get_test_transform():

    return A.Compose([

        # A.Resize(512, 512),

        ToTensorV2(p=1.0)

    ])



test_dataset = WheatTestDataset(test_df, DIR_TEST, get_test_transform())



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)



def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)



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

        

        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }



        

        results.append(result)

        

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()

test_df.to_csv('submission.csv', index=False)