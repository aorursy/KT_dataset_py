import pandas as pd

import numpy as np

import os

import re



from PIL import Image

from PIL import ImageDraw



import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN



from torch.utils.data import DataLoader, Dataset



from matplotlib import pyplot as plt



weights_dir = '../input/wheat-head-detection'



saved_model_path = os.path.join(weights_dir,'fasterrcnn_resnet50_fpn.pth')

saved_model_path
# Creating the test dataset

from torchvision.transforms import functional as F



class WheatDatasetTest(Dataset):

    def __init__(self, data_dir, dataframe, transforms=True):

        super().__init__()

        self.data = data_dir

        self.df = dataframe

        self.transforms = transforms

        # load all the images and sort them

        self.images = sorted(os.listdir(os.path.join(data_dir)))

        

    def __getitem__(self, idx:int):

        # getting the image id for the given image

        image_id = self.images[idx].split('.')[0]

        

        # get the bounding box info for the given image

        boxes_info = (self.df[self.df['image_id'] == image_id])



        # load the image

        path = os.path.join(self.data, image_id) + '.jpg'

        image = Image.open(path).convert("RGB")

        

        if self.transforms:

            image= F.to_tensor(image)   # normalizes and returns `torch.tensor`

            

        return image, image_id

    

    def __len__(self):

        return len(self.images)
# Define the root directory

root_dir = '../input/global-wheat-detection'



# create test dataframe

test_df = pd.read_csv(os.path.join(root_dir, 'sample_submission.csv'))

test_df
# location of test images

test_data_dir = os.path.join(root_dir, 'test')

test_data_dir
# merge all the images in a batch

def collate_fn(batch):

    return tuple(zip(*batch))



# Use the dataset class and create train and test dataloaders

dataset_test = WheatDatasetTest(data_dir=test_data_dir, dataframe=test_df)



# creating dataloader

dataloader_test = DataLoader(dataset_test, batch_size=10, shuffle=False, 

                                   num_workers=4, collate_fn=collate_fn)
def get_model(num_classes):

    # load the object detection model pre-trained on COCO

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, 

                                                                 pretrained_backbone=False)

    

    # get the input features in the classifier

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    

    # replace the input features of pretrained head with the num_classes

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    

    return model
if torch.cuda.is_available():

    print(torch.cuda.get_device_name())



device = 'cuda' if torch.cuda.is_available() else 'cpu'
# get the model and load the saved model

# our dataset has only two classes: wheat heads and background

num_classes = 2



# get the model using helper function

model = get_model(num_classes)



# Load the trained weights

model.load_state_dict(torch.load(saved_model_path))

model.eval()



# move the model to device

model.to(device)
def format_prediction_string(boxes, scores):

    # source: https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
detection_threshold = 0.45

results = []



for images, image_ids in dataloader_test:



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

test_df
# Showing inference on model output

def show_inferences(images, targets, num_images):

    ncols = 2

    nrows = min(num_images//ncols, 2)

    

    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 14))

    

    ax = ax.flatten()  

    for idx in range(num_images):

        image = Image.fromarray(images[idx].mul(255).permute(1,2,0).cpu().detach().byte().numpy())

        boxes = targets[idx]['boxes'].cpu().detach().numpy().astype(np.int64)



        draw = ImageDraw.Draw(image)



        for box in boxes:

            coord1 = (box[0], box[1])

            coord2 = (box[2], box[3])

            draw.rectangle([coord1, coord2], outline=(220,20,60), width=10)

        ax[idx].set_axis_off()

        ax[idx].imshow(image)
show_inferences(images, outputs, 4)
test_df.to_csv("submission.csv", index=False)