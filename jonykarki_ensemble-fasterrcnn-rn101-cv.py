import sys

sys.path.insert(0, "/kaggle/input/weightedboxesfusion")



from IPython.display import Image



import pandas as pd

import numpy as np

import cv2

import os, re

import gc

import random



import torch



import albumentations

from albumentations.pytorch.transforms import ToTensorV2



import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone



from torch.utils.data import DataLoader, Dataset



from matplotlib import pyplot as plt 

plt.rcParams['figure.figsize'] = (10.0, 10.0)
DATA_DIR = "/kaggle/input/global-wheat-detection"

MODELS_IN_DIR = "/kaggle/input"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def load_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    assert image is not None, f"IMAGE NOT FOUND AT {image_path}"

    return image
class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms



    def __len__(self) -> int:

        return len(self.image_ids)



    def __getitem__(self, idx: int):

        image_id = self.image_ids[idx]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

#         # change the shape from [h,w,c] to [c,h,w]  

#         image = torch.from_numpy(image).permute(2,0,1)



        records = self.df[self.df['image_id'] == image_id]

    

        if self.transforms:

            sample = {"image": image}

            sample = self.transforms(**sample)

            image = sample['image']



        return image, image_id
def get_test_transforms():

    return albumentations.Compose([

                ToTensorV2(p=1.0)

            ])
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)



def collate_fn(batch):

    return tuple(zip(*batch))



test_dataset = WheatDataset(pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv")), os.path.join(DATA_DIR, "test"), get_test_transforms())



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
def get_model(checkpoint_path):

    """

    https://stackoverflow.com/questions/58362892/resnet-18-as-backbone-in-faster-r-cnn

    """

    backbone = resnet_fpn_backbone('resnet101', pretrained=False)

    model = FasterRCNN(backbone, num_classes=2)

    model.load_state_dict(torch.load(checkpoint_path))

    model.to(DEVICE)

    model.eval()

    return model



def get_model_152(checkpoint_path):

    backbone = resnet_fpn_backbone('resnet152', pretrained=False)

    model = FasterRCNN(backbone, num_classes=2)

    model.load_state_dict(torch.load(checkpoint_path))

    model.to(DEVICE)

    model.eval()

    return model
models = [

    get_model(os.path.join(MODELS_IN_DIR, "frcnn101f0", "best_model.pth")),

    get_model(os.path.join(MODELS_IN_DIR, "frcnnfone", "best_model.pth")),

    get_model(os.path.join(MODELS_IN_DIR, "frcnnfoldtwo", "best_model.pth")),

    get_model(os.path.join(MODELS_IN_DIR, "frcnnfoldthree", "best_model.pth")),

    get_model(os.path.join(MODELS_IN_DIR, "frcnnfoldfour", "best_model.pth")),

    get_model_152(os.path.join(MODELS_IN_DIR, "frcnn152foldthree", "best_model.pth")),

    get_model_152(os.path.join(MODELS_IN_DIR, "frcnn152foldtwo", "best_model.pth")),

]
from ensemble_boxes import *



def make_ensemble_predictions(images):

    images = list(image.to(DEVICE) for image in images)    

    result = []

    for model in models:

        with torch.no_grad():

            outputs = model(images)

            result.append(outputs)

            del model

            gc.collect()

            torch.cuda.empty_cache()

    return result



def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.55, skip_box_thr=0.7, weights=None):

    boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]

    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
results = []



for images, image_ids in test_data_loader:

    predictions = make_ensemble_predictions(images)

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        boxes = boxes.astype(np.int32).clip(min=0, max=1023)

        image_id = image_ids[i]



        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        

        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }

        results.append(result)
results[0:2]
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()
test_df.to_csv('submission.csv', index=False)