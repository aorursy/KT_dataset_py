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
import os
import sys
import tarfile
import torch
import collections
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as F


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
classes = ["background", 'crack', 'bullseye', 'scratch']
# Defining Dataset Class
class VOCDetection(VisionDataset):
    def __init__(self,root,year='2012',image_set='train',download=False,transform=None,target_transform=None,transforms=None):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)

        image_dir = os.path.join("../input/internvoc/intern", 'images')
        annotation_dir = os.path.join("../input/internvoc/intern", 'annotations')

        self.images = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.annotations = [os.path.join(annotation_dir, x ) for x in os.listdir(annotation_dir)]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        raw_target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        
        target = {}
        boxes = []
        labels = []
        try:
            for that in raw_target["annotation"]["object"]:
                boxes.append(list(map(float, list(that["bndbox"].values()))))
                labels.append(classes.index(that["name"]))
            target["boxes"] = torch.tensor(boxes)
            target["labels"] = torch.tensor(labels)
        except TypeError:
            boxes.append(list(map(float, list(raw_target["annotation"]["object"]["bndbox"].values()))))
            labels.append(classes.index(raw_target["annotation"]["object"]["name"]))
            target["boxes"] = torch.tensor(boxes)
            target["labels"] = torch.tensor(labels)
        target["image_id"] = torch.tensor([int(raw_target["annotation"]["filename"].split(".")[0])])
        
    
        if self.transforms :
            img, target = self.transforms(img, target)
            
        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)
datadict = VOCDetection("../input/internvoc/intern", year="2007", download=False, transforms=False)
!git clone https://github.com/adityak2920/FasterRCNN
!pip install pycocotools
import sys
sys.path.append("FasterRCNN/src/")

import transforms as T
import utils
from engine import train_one_epoch, evaluate

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
def faster_rcnn_model(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    
    return model
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = VOCDetection("../input/internvoc/intern", year="2007", download=False, transforms=get_transform(train=True))
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices)


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8,
    collate_fn=utils.collate_fn)


num_classes = 4  #  classes + background 
model = faster_rcnn_model(num_classes, device)

#defining paremeters for training
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                          momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=3,
                                             gamma=0.1)


num_epochs = 10

for epoch in range(num_epochs):
  # train for one epoch, printing every 10 iterations
  train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
  # update the learning rate
  lr_scheduler.step()
  # evaluate on the test dataset
#       evaluate(model, data_loader_test, device=device)

print("That's it!")

model.eval()
evalt = T.ToTensor()
annot = model([F.to_tensor(Image.open("../input/internvoc/intern/images/46.jpg")).cuda()])
annot
x1 = annot[0]["boxes"].cpu().detach().numpy()[0][0]
y1 = annot[0]["boxes"].cpu().detach().numpy()[0][1]
x2 = annot[0]["boxes"].cpu().detach().numpy()[0][2]
y2 = annot[0]["boxes"].cpu().detach().numpy()[0][3]
classes[int(annot[0]["labels"])]
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

img = np.array(Image.open("../input/internvoc/intern/images/46.jpg"), dtype=np.uint8)
# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(img)

width = x2 - x1
height = y2 - y1
rect = patches.Rectangle((x1,y1),width,height,linewidth=1,edgecolor='r',facecolor='none', label = classes[int(annot[0]["labels"])])
ax.add_patch(rect)

plt.show()
