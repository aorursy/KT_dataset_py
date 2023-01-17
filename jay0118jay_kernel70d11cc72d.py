# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from PIL import Image

import numpy as np

import torchvision

import random

import torchvision.transforms as transforms

from tqdm import tqdm

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import glob

from torch.utils.data import Dataset, DataLoader



DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'



# DIR_WEIGHTS = '/kaggle/input/global-wheat-detection-public'

DIR_WEIGHTS = '/kaggle/input/kernel70d11cc72d'



# WEIGHTS_FILE = f'{DIR_WEIGHTS}/fasterrcnn_resnet50_fpn_best.pth'

WEIGHTS_FILE = f'{DIR_WEIGHTS}/model_states.pt'
import math

import torch

from torch.optim.optimizer import Optimizer, required



class RAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):

        if not 0.0 <= lr:

            raise ValueError("Invalid learning rate: {}".format(lr))

        if not 0.0 <= eps:

            raise ValueError("Invalid epsilon value: {}".format(eps))

        if not 0.0 <= betas[0] < 1.0:

            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        if not 0.0 <= betas[1] < 1.0:

            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        

        self.degenerated_to_sgd = degenerated_to_sgd

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):

            for param in params:

                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):

                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])

        super(RAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                buffered = group['buffer'][int(state['step'] % 10)]

                if state['step'] == buffered[0]:

                    N_sma, step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    elif self.degenerated_to_sgd:

                        step_size = 1.0 / (1 - beta1 ** state['step'])

                    else:

                        step_size = -1

                    buffered[2] = step_size



                # more conservative since it's an approximated value

                if N_sma >= 5:

                    if group['weight_decay'] != 0:

                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)

                    p.data.copy_(p_data_fp32)

                elif step_size > 0:

                    if group['weight_decay'] != 0:

                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                    p.data.copy_(p_data_fp32)



        return loss
train_mat = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')

train_mat.head()

test_df = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')

print(train_mat['source'].unique())

color_list = dict()

for src in train_mat['source'].unique():

    color_list[src] = (random.random(), random.random(), random.random())

print(color_list)

print(color_list['usask_1'])

a = 'usask_1'

print(color_list[a])
def red_box(box_df, plt, m_output=False):

    for i in range(len(box_df)):

        if isinstance(box_df, torch.Tensor):

            if m_output:

                box_df[i][2] -= box_df[i][0]

                box_df[i][3] -= box_df[i][1]

            rect = patches.Rectangle(

                (float(box_df[i][0]), float(box_df[i][1])),

                float(box_df[i][2]),

                float(box_df[i][3]),

                linewidth=2,

                edgecolor='red',

                facecolor='none'

            )

        else:

            # Read image, not model output

            one_bbox = box_df['bbox'].iloc[i].replace("[", "").replace("]", "").replace(" ", "").split(",")

            source = box_df['source'].iloc[i]

            rect = patches.Rectangle(

                (float(one_bbox[0]), float(one_bbox[1])),

                float(one_bbox[2]),

                float(one_bbox[3]),

                linewidth=2,

                edgecolor=color_list[source],

                facecolor='none'

            )



        plt.add_patch(rect)
one = train_mat.loc[train_mat['image_id'] == '0a3cb453f']

print(len(one))

print(one.head())

print(isinstance(one, pd.core.frame.DataFrame))

print(type(one))

print(one['bbox'].iloc[0].replace("[", "").replace("]","").replace(" ","").split(','))
one_image = Image.open('/kaggle/input/global-wheat-detection/train/0a3cb453f.jpg')

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(1, 1, 1)

ax.imshow(one_image)

transform = transforms.ToTensor()

img_ten = transform(one_image)

print(img_ten.max())

print(img_ten.size())

red_box(one, ax) # Input dataframe and fig subplot(which is plotting image)
class wheatDataset(object):

    # original bbox = [x_min, y_min, width, height] --> wheatDataset return boxs(list) = [[x_min, y_min, x_max, y_max]]

    def __init__(self, root_dir, dataframe, transforms, train=True):

        self.img_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.root_dir = root_dir

        self.transforms = transforms

        self.len = len(self.img_ids)

        self.train = train

    

    def __getitem__(self, index):

        image_id = self.img_ids[index]

        a, b, c = torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item()

        

        if self.train:

            bbox = self.df[self.df['image_id'] == image_id]['bbox']

            box_l = []

            for i in range(len(bbox)):

                if a > 0.5 and b > 0.5: # horizontal flip and vertical

                    one_bbox_t = list(map(float, bbox.iloc[i].replace("[", "").replace("]", "").replace(" ", "").split(",")))

                    one_bbox = [1024 - one_bbox_t[0]-one_bbox_t[2], 1024-one_bbox_t[1]-one_bbox_t[3], one_bbox_t[2], one_bbox_t[3]]

                    one_bbox[2] += one_bbox[0]

                    one_bbox[3] += one_bbox[1]

                    box_l.append(one_bbox)

                elif a > 0.5 and b < 0.5: # horizontal flip

                    one_bbox_t = list(map(float, bbox.iloc[i].replace("[", "").replace("]", "").replace(" ", "").split(",")))

                    one_bbox = [1024 - one_bbox_t[0]-one_bbox_t[2], one_bbox_t[1], one_bbox_t[2], one_bbox_t[3]]

                    one_bbox[2] += one_bbox[0]

                    one_bbox[3] += one_bbox[1]

                    box_l.append(one_bbox)

                elif b > 0.5 and a < 0.5: # vertical flip

                    one_bbox_t = list(map(float, bbox.iloc[i].replace("[", "").replace("]", "").replace(" ", "").split(",")))

                    one_bbox = [one_bbox_t[0], 1024-one_bbox_t[1]-one_bbox_t[3], one_bbox_t[2], one_bbox_t[3]]

                    one_bbox[2] += one_bbox[0]

                    one_bbox[3] += one_bbox[1]

                    box_l.append(one_bbox)

                elif a < 0.5 and b < 0.5: # ordinary

                    one_bbox = list(map(float, bbox.iloc[i].replace("[", "").replace("]", "").replace(" ", "").split(",")))

                    one_bbox[2] += one_bbox[0]

                    one_bbox[3] += one_bbox[1]

                    box_l.append(one_bbox)

            boxs = torch.tensor(box_l)

        else:

            boxs = None



        image = Image.open("/kaggle/input/global-wheat-detection" + self.root_dir + '/' + image_id + '.jpg')

        

        if a > 0.5:

            trans1 = transforms.RandomHorizontalFlip(p=1.0)

            image = trans1(image)

        

        if b > 0.5:

            trans1 = transforms.RandomVerticalFlip(p=1.0)

            image = trans1(image)

        

        if c > 0.5:

            trans1 = transforms.ColorJitter(contrast=(0.5, 1.5), brightness=(0.5, 1.5), saturation=(0.5, 1.5))

            image = trans1(image)

        

        if self.transforms:

            image = self.transforms(image).to(device)

        

        return image, image_id, boxs



    def __len__(self):

        return self.len



    

    

class WheatDataset(Dataset):  # For test phase

    def __init__(self, dataframe, image_dir, transforms):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms



    def __len__(self) -> int:

        return len(self.image_ids)



    def __getitem__(self, idx: int):

        image_id = self.image_ids[idx]

        image = Image.open(f'{self.image_dir}/{image_id}.jpg')

        if self.transforms:

            image = self.transforms(image).to(device)



        records = self.df[self.df['image_id'] == image_id]



        return image, image_id
# Assign dataset and dataloader



def collate_fn(batch):

    return tuple(zip(*batch))



trainset = wheatDataset(root_dir='/train', dataframe=train_mat, transforms=transform)

set_length = len(trainset)

trainset, validset = torch.utils.data.random_split(trainset, [set_length-int(set_length/10), int(set_length/10)])



trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=16, shuffle=False, collate_fn=collate_fn)

validloader = torch.utils.data.DataLoader(dataset=validset, batch_size=16, shuffle=True, collate_fn=collate_fn)
# Test dataset and dataloader



device = 'cuda'

dataiter = iter(trainloader)

images, labels, box = dataiter.next()

images = list(image for image in images)

print('d', labels)

print(images[0].size())

print(box[0].size())

print(len(box))

print(box[0][:6,:])





fig = plt.figure(figsize=(7, 7))

ax = fig.add_subplot(1, 1, 1)

with torch.no_grad():

    ax.imshow(images[0].permute(1, 2, 0).cpu())

    red_box(box[0], ax, m_output=True)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)



num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device("cuda")))



optimizer = RAdam([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)


num_epoch = 5



model.train()

model.to(device)



for epoch in range(num_epoch):

    total_loss = 0.0

    model.train()

    for i, (images, labels, boxes) in tqdm(enumerate(trainloader)):

        targets = []

        for j in range(len(images)):

            d = {}

            d['boxes'] = boxes[j].to(device)

            d['labels'] = torch.ones(boxes[j].size(0), dtype=torch.int64).to(device)

            targets.append(d)

        

        images = list(image for image in images)

        output = model(images, targets)

        losses = sum(loss for loss in output.values())



        optimizer.zero_grad()

        losses.backward()

        optimizer.step()



        total_loss += losses.item()

    

    model.eval()

    for i, (images, labels, boxes) in tqdm(enumerate(validloader)):

        targets = []

        for j in range(len(images)):

            d = {}

            d['boxes'] = boxes[j].to(device)

            d['labels'] = torch.ones(boxes[j].size(0), dtype=torch.int64).to(device)

            targets.append(d)

        

        images = list(image for image in images)

        output = model(images, targets)

        losses = sum(loss for loss in output.values())

    

    if epoch % 1 == 0:

        print(f"{epoch}/{num_epoch} : {total_loss:.4f}")
# Test trainset! not testset

detection_threshold = 0.5



trainset = wheatDataset(root_dir='/train', dataframe=train_mat, transforms=transform)

trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=4, shuffle=True, collate_fn=collate_fn)

device = 'cuda:0'

dataiter = iter(trainloader)

images, labels, box = dataiter.next()

images = list(image for image in images)



model.eval()

output = model(images)

# print(output[0].values())



fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(2, 2, 1)

ax2 = fig.add_subplot(2, 2, 2)

ax3 = fig.add_subplot(2, 2, 3)

ax4 = fig.add_subplot(2, 2, 4)

ax = [ax1, ax2, ax3, ax4]



for i, a in enumerate(ax):

    # one = train_mat.loc[train_mat['image_id'] == labels[i]]

    a.imshow(Image.open("/kaggle/input/global-wheat-detection/train/" + labels[i] + '.jpg'))

    boxes = output[i]['boxes'].data.cpu().numpy()

    scores = output[i]['scores'].data.cpu().numpy()

    boxes = torch.from_numpy(boxes[scores >= detection_threshold].astype(np.int32))

    red_box(boxes, a, m_output=True)
sample_submission = pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")

sample_submission

sample_submission.to_csv("/kaggle/working/sample_submission.csv", index=False)
test_dataset = WheatDataset(test_df, os.path.join("/kaggle/input/global-wheat-detection", "test"), transforms=transform)



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

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

os.chdir("/kaggle/working")

!pwd

!ls
torch.save(model.state_dict(), "model_states.pt")



test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

print(test_df)

test_df.to_csv('submission.csv', index=False)

print("saved")
"""

detection_threshold = 0.3



test_img_list = glob.glob("/kaggle/input/global-wheat-detection/test/*.jpg")

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(1, 1, 1)

answer_list = []



testset = wheatDataset(root_dir='/test', dataframe=test_df, transforms=transform, train=False)

testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False, collate_fn=collate_fn)



for i, (images, labels, _) in tqdm(enumerate(testloader)):

    targets = []

    pdstring = ""



    output = model(images)

    for j in range(len(boxes)):

        if j == len(boxes)-1:

            pdstring += str(scores[j].item()) + " " + str(boxes[j][0].item()) + " " + str(boxes[j][1].item()) + " " + str(boxes[j][2].item()-boxes[j][0].item()) + " " + str(boxes[j][3].item() - boxes[j][1].item())

        else:

            pdstring += str(scores[j].item()) + " " + str(boxes[j][0].item()) + " " + str(boxes[j][1].item()) + " " + str(boxes[j][2].item()-boxes[j][0].item()) + " " + str(boxes[j][3].item() - boxes[j][1].item()) + " "

    answer_list.append(pdstring)

   

   

   

   fucnalksdnf;kasjdnfkj;sadnfkjsnadkfasdkhfasldkfhsadjhf;lasjfd 

   

   

   

"""

"""

for i in range(10):

    pdstring = ""

    img = transforms.ToTensor()(Image.open(f"/kaggle/input/global-wheat-detection/test/{sample_submission.iloc[i, 0]}.jpg"))

    model.eval()

    output = model(img.unsqueeze(0).cuda())

    

    ax.imshow(img.permute(1, 2, 0).cpu())

    boxes = output[0]['boxes'].data.cpu().numpy()

    scores = output[0]['scores'].data.cpu().numpy()

    

    boxes = torch.from_numpy(boxes[scores >= detection_threshold].astype(np.int32))

    scores = torch.from_numpy(scores[scores >= detection_threshold])

    

    for j in range(len(boxes)):

        if j == len(boxes)-1:

            print('end')

            pdstring += str(scores[j].item()) + " " + str(boxes[j][0].item()) + " " + str(boxes[j][1].item()) + " " + str(boxes[j][2].item()-boxes[j][0].item()) + " " + str(boxes[j][3].item() - boxes[j][1].item())

        else:

            pdstring += str(scores[j].item()) + " " + str(boxes[j][0].item()) + " " + str(boxes[j][1].item()) + " " + str(boxes[j][2].item()-boxes[j][0].item()) + " " + str(boxes[j][3].item() - boxes[j][1].item()) + " "

    print(i, f"{sample_submission.iloc[i, 0]}", pdstring)

    answer_list.append(pdstring)

    red_box(boxes, ax, m_output=True)

"""
"""

print(sample_submission)

sample_submission.iloc[:, 1] = answer_list

os.chdir("/kaggle/working/")

print(sample_submission)

sample_submission.to_csv("submission.csv", index=False)

print("saved")

"""