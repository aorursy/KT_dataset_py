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
from IPython.display import Image
Image("/kaggle/input/global-wheat-detection/train/00ea5e5ee.jpg", width=500)
import pandas as pd
import numpy as np
import cv2
import os
import re

import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader, Dataset

from matplotlib import pyplot as plt
# Set default figure size
plt.rcParams['figure.figsize'] = (10.0, 10.0)
# Define File Path Constants
INPUT_DIR = os.path.abspath('/kaggle/input/global-wheat-detection')
TRAIN_DIR = os.path.join(INPUT_DIR, "train")

# Load and Show Training Labels
pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
def read_image_from_path(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_image_from_train_folder(image_id):
    path = os.path.join(TRAIN_DIR, image_id + ".jpg")
    return read_image_from_path(path)
# Test loader functions
sample_image_id = "b6ab77fd7"
plt.imshow(read_image_from_train_folder(sample_image_id))
_ = plt.title(sample_image_id)
# Functions for parsing bounding box string into x1, y1, x2, y2
def parse_bbox_text(string_input):
    input_without_brackets = re.sub("\[|\]", "", string_input)
    input_as_list = np.array(input_without_brackets.split(","))
    return input_as_list.astype(np.float) 

def xywh_to_x1y1x2y2(x,y,w,h):
    return np.array([x,y,x+w,y+h])
# Parse training bounding box labels
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
bbox_series = train_df.bbox.apply(parse_bbox_text)

xywh_df = pd.DataFrame(bbox_series.to_list(), columns=["x", "y", "w", "h"])

x2_df = pd.DataFrame(xywh_df.x + xywh_df.w, columns=["x2"])
y2_df = pd.DataFrame(xywh_df.y + xywh_df.h, columns=["y2"])

# Update training dataframe with parsed labels
train_df = train_df.join([xywh_df, x2_df, y2_df])
train_df.head()
# Convenience function for drawing a list of bounding box coordinates on and image
def draw_boxes_on_image(boxes, image, color=(255,0,0)):    
    for box in boxes:
        cv2.rectangle(image,
                      (int(box[0]), int(box[1]) ),
                      (int(box[2]), int(box[3]) ),
                      color, 3)
    return image
# Sample a random training instance and draw the labelled bounding boxes
sample_image_id =  train_df.image_id.sample().item()

sample_image = read_image_from_train_folder(sample_image_id)
sample_bounding_boxes = train_df[train_df.image_id == sample_image_id][["x", "y","x2","y2"]]

plt.imshow(draw_boxes_on_image(sample_bounding_boxes.to_numpy(), sample_image, color=(0,200,200)))
_ = plt.title(sample_image_id)
# Download a pre-trained bounding box detector
model = fasterrcnn_resnet50_fpn(pretrained=True)
model
# Replace the pre-trained bounding box detector head with
# a new one that predicts our desired 2 classes {BACKGROUND, WHEAT}
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=2)

# Verify the model architecture
model.roi_heads
# Determine device to run on. GPU is highly recommended
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def move_batch_to_device(images, targets):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets
## Split data into training and validation subsets
unique_image_ids = train_df['image_id'].unique()

n_validation = int(0.2 * len(unique_image_ids))
valid_ids = unique_image_ids[-n_validation:]
train_ids = unique_image_ids[:-n_validation]

validation_df = train_df[train_df['image_id'].isin(valid_ids)]
training_df = train_df[train_df['image_id'].isin(train_ids)]

print("%i training samples\n%i validation samples" % (len(training_df.image_id.unique()),len(validation_df.image_id.unique())) )
class WheatDataset(Dataset):

    def __init__(self, dataframe):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe

    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        image = read_image_from_train_folder(image_id).astype(np.float32)
        # Scale to [0,1] range expected by the pre-trained model
        image /= 255.0
        # Convert the shape from [h,w,c] to [c,h,w] as expected by pytorch
        image = torch.from_numpy(image).permute(2,0,1)
        
        records = self.df[self.df['image_id'] == image_id]
        
        boxes = records[['x', 'y', 'x2', 'y2']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        n_boxes = boxes.shape[0]
        
        # there is only one foreground class, WHEAT
        labels = torch.ones((n_boxes,), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        
        return image, target
# Create pytorch data loaders for training and validation

train_dataset = WheatDataset(training_df)
valid_dataset = WheatDataset(validation_df)

# A function to bring images with different
# number of bounding boxes into the same batch
def collate_fn(batch):
    return tuple(zip(*batch))

is_training_on_cpu = device == torch.device('cpu')
batch_size = 4 if is_training_on_cpu else 16

train_data_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
# Test the data loader
batch_of_images, batch_of_targets = next(iter(train_data_loader))

sample_boxes = batch_of_targets[0]['boxes'].cpu().numpy().astype(np.int32)
sample_image = batch_of_images[0].permute(1,2,0).cpu().numpy() # convert back from pytorch format

plt.imshow(draw_boxes_on_image(sample_boxes, sample_image, color=(0,200,200)))
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
num_epochs = 1 if is_training_on_cpu else 3

# Prepare the model for training
model = model.to(device)
model.train()
    
for epoch in range(num_epochs):
    print("Epoch %i/%i " % (epoch + 1, num_epochs) )
    average_loss = 0
    for batch_id, (images, targets) in enumerate(train_data_loader):
        # Prepare the batch data
        images, targets = move_batch_to_device(images, targets)

        # Calculate losses
        loss_dict = model(images, targets)
        batch_loss = sum(loss for loss in loss_dict.values()) / len(loss_dict)
        
        # Refresh accumulated optimiser state and minimise losses
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        # Record stats
        loss_value = batch_loss.item()
        average_loss = average_loss + (loss_value - average_loss) / (batch_id + 1)
        print("Mini-batch: %i/%i Loss: %.4f" % ( batch_id + 1, len(train_data_loader), average_loss), end='\r')
        if batch_id % 100 == 0:
            print("Mini-batch: %i/%i Loss: %.4f" % ( batch_id + 1, len(train_data_loader), average_loss))
#Проверяем обучение 
# Подготовить модель для вывода
model.eval()


def make_validation_iter():
    valid_data_iter = iter(valid_data_loader)
    for images, targets in valid_data_iter:
        images, targets = move_batch_to_device(images, targets)

        cpu_device = torch.device("cpu")
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        for image, output, target in zip(images, outputs, targets): 
            predicted_boxes = output['boxes'].cpu().detach().numpy().astype(np.int32)
            ground_truth_boxes = target['boxes'].cpu().numpy().astype(np.int32)
            image = image.permute(1,2,0).cpu().numpy()
            yield image, ground_truth_boxes, predicted_boxes

validation_iter = make_validation_iter()
image, ground_truth_boxes, predicted_boxes = next(validation_iter)
image = draw_boxes_on_image(predicted_boxes, image, (255,0,0))
image = draw_boxes_on_image(ground_truth_boxes, image , (0,255,0))
plt.imshow(image)
torch.save(model.state_dict(), 'fasterrcnn_gwd_finetuned.pth')