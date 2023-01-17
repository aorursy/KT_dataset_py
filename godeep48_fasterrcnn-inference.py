import pandas as pd

import numpy as np

from matplotlib import pyplot as plt



#from tqdm import tqdm_notebook as tqdm

from tqdm import tqdm 



import cv2

import os

import re



import random



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
INPUT_DATA = "../input/global-wheat-detection/"

TRAIN_DIR = os.path.join(INPUT_DATA, "train")

TEST_DIR = os.path.join(INPUT_DATA, "test")



#Path for the weight file.

DIR_PATH = "../input/fasterrcnn-wheat-head-detection"

WEIGHT_FILE = os.path.join(DIR_PATH, "fasterrcnn_best_resnet50.pth")
df_test = pd.read_csv(os.path.join(INPUT_DATA, "sample_submission.csv"))

print(f"Shape of test dataframe: {df_test.shape}")

df_test.head(4)
# Data Transform - Test Albumentation

def get_test_transform():

    return A.Compose([

        ToTensorV2(p=1.0)

    ])





class WheatDatasetTest(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):

        super().__init__()

        self.dataframe = dataframe

        self.image_dir = image_dir

        self.transform = transform

        self.image_ids = dataframe["image_id"].unique()

        

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        #details = self.dataframe[self.dataframe["image_id"]==image_id]

        img_path = os.path.join(TEST_DIR, image_id)+".jpg"

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        

        if self.transform:

            sample = {

                'image': image,

            }

            

            sample = self.transform(**sample)

            image = sample['image']

        

        return image, image_id

    

    def __len__(self) -> int:

        return len(self.image_ids)
def collate_fn(batch):

    return tuple(zip(*batch))



test_dataset = WheatDatasetTest(df_test, TEST_DIR, get_test_transform())

print(f"Length of test dataset: {len(test_dataset)}")



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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



num_classes = 2  # 1 class (wheat) + background



# get number of input features for the classifier

in_features = model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



# Load the trained weights

model.load_state_dict(torch.load(WEIGHT_FILE, map_location=torch.device('cpu')))

model.eval()



x = model.to(device)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
detection_threshold = 0.5

results = []

output_list = []





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

        

        output_dict = {

            'image_id': image_ids[i],

            'boxes': outputs[i]['boxes'].data.cpu().numpy(),

            'scores': outputs[i]['scores'].data.cpu().numpy()

        }

        output_list.append(output_dict)

        

        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }



        

        results.append(result)
df_test = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

df_test.head()
## Plot image prediction



def predict_images(n_num, random_selection=True):

    '''Plot N Number of Predicted Images'''

    if random_selection:

        index = random.sample(range(0, len(df_test["image_id"].unique())), n_num)

    else:

        index = range(0, n_num)

        

    plt.figure(figsize=(15,15))

    fig_no = 1

    

    for i in index:

        images, image_id = test_dataset.__getitem__(i)

        sample = images.permute(1,2,0).cpu().numpy()

        boxes = output_list[i]['boxes']

        scores = output_list[i]['scores']

        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        #Plot figure/image

        for box in boxes:

            cv2.rectangle(sample,(box[0], box[1]),(box[2], box[3]),(255,223,0), 2)

        plt.subplot(n_num/2, n_num/2, fig_no)

        plt.imshow(sample)

        fig_no+=1
predict_images(4, True)
df_test.to_csv('submission.csv', index=False)