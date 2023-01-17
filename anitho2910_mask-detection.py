# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

import torch

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from torch.utils.data.sampler import SubsetRandomSampler

import json

import cv2

import matplotlib.pyplot as plt

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch.optim as optim

%matplotlib inline
path = '/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask'

os.listdir(path)
categories = ['background']

with open(path + '/meta.json') as f:

    data = json.load(f)

    for i in data['classes']:

        categories.append(i['title'])
categories
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device
dataset_path = path + '/Medical Mask'

os.listdir(dataset_path)
import random



from torchvision.transforms import functional as F





class Compose(object):

    def __init__(self, transforms):

        self.transforms = transforms



    def __call__(self, image, target):

        for t in self.transforms:

            image, target = t(image, target)

        return image, target





class RandomHorizontalFlip(object):

    def __init__(self, prob):

        self.prob = prob



    def __call__(self, image, target):

        if random.random() < self.prob:

            height, width = image.shape[-2:]

            image = image.flip(-1)

            bbox = target["boxes"]

            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]

            target["boxes"] = bbox

        return image, target





class ToTensor(object):

    def __call__(self, image, target):

        image = F.to_tensor(image)

        return image, target
class MaskDataset(Dataset):

    

    def __init__(self, path, categories, transform = None):

        self.root = path

        self.images = list(sorted(os.listdir(path + '/images')))

        self.annotations = list(sorted(os.listdir(path + '/annotations')))

        self.transforms = transform

        self.categories = categories

        

    def __len__(self):

        return len(self.annotations)

    

    def __getitem__(self, idx):



        name_of_annnotation = self.annotations[idx]

        

        with open(self.root + '/annotations/'+name_of_annnotation) as f:

            data = json.load(f)

            

            

        bounding_boxes = []

        labels = []

        areas = []

        image_name = data['FileName']

        img = Image.open(self.root + '/images/' + image_name).convert("RGB")

        number_of_annotation = data['NumOfAnno']

        anno = data['Annotations']

        for i in anno:

            #print(i['BoundingBox'])

            x1,y1,x2,y2 = i['BoundingBox']

            bounding_boxes.append([x1,y1,x2,y2])

            labels.append(self.categories.index(i['classname']))

            area = (x2 - x1) * (y2 - y1)

            areas.append(area)

            

        target = {}

        target['boxes'] = torch.as_tensor(bounding_boxes, dtype = torch.float32)

        target['labels'] = torch.as_tensor(labels, dtype = torch.int64)

        target['image_id'] = torch.as_tensor(int(image_name[:image_name.find('.')]), dtype = torch.int64)

        target['area'] = torch.as_tensor(areas)

        target['iscrowd'] = torch.zeros((number_of_annotation,), dtype = torch.int64)

        if self.transforms is not None:

            img, target = self.transforms(img, target)

            

        return img, target
def collate_fn(batch):

    return tuple(zip(*batch))
def get_transform(train):

    transforms = []

    transforms.append(ToTensor())

    if train:

        transforms.append(RandomHorizontalFlip(0.5))

    return Compose(transforms)
batch_size = 2

shuffle = True

validation_split = 0.2

random_seed = 42
dataset = MaskDataset(dataset_path,categories, get_transform(True))

length = len(dataset)

indices = list(range(length))

split_idx = int(np.floor(length * validation_split))

if shuffle:

    np.random.seed(random_seed)

    np.random.shuffle(indices)

    

train_idx, valid_idx = indices[split_idx:], indices[:split_idx]
train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



train_data = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler, collate_fn = collate_fn)

valid_data = DataLoader(dataset, batch_size = batch_size, sampler = valid_sampler, collate_fn = collate_fn)
imgs, targets = next(iter(train_data))
def show_image(img, target):

    img = img.clone().numpy()

    img = img.transpose((1,2,0))

    img = np.ascontiguousarray(img)

    box_coords = target['boxes']

    label = target['labels']

    font = cv2.FONT_HERSHEY_SIMPLEX 

    fontscale = 0.6

    for idx, i in enumerate(box_coords):



        x1,y1,x2,y2 = map(int, i)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)

        cv2.putText(img, categories[int(label[idx])],(max(0, x1 - 5), max(0, y1 - 5)), 

                    font, fontscale, (0,0,0), 1)

    plt.figure(figsize = (15,15))

    plt.imshow(img)
show_image(imgs[0], targets[0])
model = fasterrcnn_resnet50_fpn(pretrained = True)
num_classes = len(categories)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
epoch = 5

lr = 0.0001

model.to(device)

params = [param for param in model.parameters() if param.requires_grad]

optimizer = optim.Adam(params, lr = lr)
# Uncomment following code to train the model



"""for i in range(epoch):

    running_loss = 0

    idx = 1

    print("*"*50)

    print("Epoch : {}/{}".format(i + 1, epoch))

    for images,targets in train_data:

      if idx % 200 == 0:

        print(f"Iteration {idx}/1731")

      optimizer.zero_grad()

      images = [image.to(device) for image in images]

      targets = [{key: value.to(device) for key,value in target.items()} for target in targets]

      

      output = model(images, targets)

      

      losses = sum(loss for loss in output.values())

      running_loss += losses.item()

      

      losses.backward()

      optimizer.step()

      idx += 1



    print("Loss after epoch {} is {:.4f}".format(i + 1, running_loss/len(train_data))) """   
path_to_model = '/kaggle/input/model-weights-trained-on-colab'

os.listdir(path_to_model)
# For CPU

#model = torch.load(path_to_model+'/complete-model.pth', map_location=torch.device('cpu'))

# For GPU

model = torch.load(path_to_model+'/complete-model.pth')
model
test_csv = pd.read_csv('/kaggle/input/face-mask-detection-dataset/submission.csv')

test_csv.head()
length = len(test_csv['name'])

i = 0

model.eval()

while i < length:

    name_of_file = test_csv['name'][i]

    img = Image.open(dataset_path+'/images/'+name_of_file).convert("RGB")

    img = F.to_tensor(img)

    count = len(test_csv[test_csv['name'] == name_of_file])

    print(f"Doing inference on image {name_of_file}")

    with torch.no_grad():

        predictions = model([img.to(device)])

        for j in range(count):

            print(f"Getting {j} inference on {name_of_file} ")

            try:

                preds = list(map(int, predictions[0]['boxes'][j].to('cpu').numpy()))

                label = categories[predictions[0]['labels'][j].item()]

                test_csv.loc[i + j, 'x1'], test_csv.loc[i + j, 'x2'], test_csv.loc[i + j,'y1'], test_csv.loc[i + j,'y2'] = preds 

                test_csv.loc[i + j, 'classname'] = label

            except:

                continue

    i += count
test_csv.tail()
test_csv.to_csv('submission.csv', index = False)