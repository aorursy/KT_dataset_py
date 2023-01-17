import pandas as pd

import numpy as np

import cv2

import os, re



import torch

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



from torch.utils.data import DataLoader, Dataset



from matplotlib import pyplot as plt
DATA_DIR = "/kaggle/input/global-wheat-detection"

MODELS_IN_DIR = "/kaggle/input/fasterrcnnresnet101"
test_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

test_df.shape
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

def get_model():

    """

    https://stackoverflow.com/questions/58362892/resnet-18-as-backbone-in-faster-r-cnn

    """

    backbone = resnet_fpn_backbone('resnet101', pretrained=False)

    model = FasterRCNN(backbone, num_classes=2)

    return model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model()

model.load_state_dict(torch.load(os.path.join(MODELS_IN_DIR, "best_model.pth")))

model.eval()



model.to(device)

1 == 1
def get_test_transforms():

    return A.Compose([

            A.Resize(height=1024, width=1024, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)
def collate_fn(batch):

    return tuple(zip(*batch))



test_dataset = WheatDataset(test_df, os.path.join(DATA_DIR, "test"), get_test_transforms())



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

    num_workers=1,

    drop_last=False,

    collate_fn=collate_fn

)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
detection_threshold = 0.4

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
results[0:2]
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()
test_df.to_csv('submission.csv', index=False)