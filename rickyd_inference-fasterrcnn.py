import sys

sys.path.insert(0, "../input/weightedboxesfusion")



import numpy as np

import pandas as pd 

import cv2

import os

import torch

import torchvision

import glob 

import ensemble_boxes



from matplotlib import pyplot as plt

from itertools import product

from torch.utils.data import DataLoader, Dataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN

from albumentations import Resize,Compose

from albumentations.pytorch.transforms import ToTensorV2





DATA_TEST_PATH = '../input/global-wheat-detection/test/'

MODEL_PATH = '../input/faster000075/fasterrcnn_fold 0 000075.pt'





device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device
def collate_fn(batch):

    return tuple(zip(*batch))



def get_test_transform():

    return Compose([

        Resize(height=512, width=512, p=1.0),

        ToTensorV2(p=1.0)

    ])



def load_checkpoint(cp_file, model, is_model=False):

    checkpoint = torch.load(cp_file)

    if is_model:

        model.load_state_dict(checkpoint)

    else:

        model.load_state_dict(checkpoint['model'])



    return model



    

def normalize(img):

    img -= img.min()

    img /= img.max()

    return img



class DatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, index: int):

        

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DATA_TEST_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image = normalize(image)

        

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

        return image, image_id



    def __len__(self) -> int:

        return len(self.image_ids)







test_ids = [path.split('/')[-1][:-4] for path in glob.glob(f'{DATA_TEST_PATH}/*.jpg')]



dataset = DatasetRetriever(image_ids=test_ids, transforms=get_test_transform())

data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False)

num_classes = 2 

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.__name__ = "fpn_resnet"

model.to(device)

model = load_checkpoint(MODEL_PATH,model,True)

model.eval()
class BaseWheatTTA:

    """ author: @shonenkov """

    image_size = 512



    def augment(self, image):

        raise NotImplementedError

    

    def batch_augment(self, images):

        raise NotImplementedError

    

    def deaugment_boxes(self, boxes):

        raise NotImplementedError



class TTAHorizontalFlip(BaseWheatTTA):

    """ author: @shonenkov """



    def augment(self, image):

        return image.flip(1)

    

    def batch_augment(self, images):

        return images.flip(2)

    

    def deaugment_boxes(self, boxes):

        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]

        return boxes



class TTAVerticalFlip(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return image.flip(2)

    

    def batch_augment(self, images):

        return images.flip(3)

    

    def deaugment_boxes(self, boxes):

        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]

        return boxes

    

class TTARotate90(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return torch.rot90(image, 1, (1, 2))



    def batch_augment(self, images):

        return torch.rot90(images, 1, (2, 3))

    

    def deaugment_boxes(self, boxes):

        res_boxes = boxes.copy()

        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]

        res_boxes[:, [1,3]] = boxes[:, [2,0]]

        return res_boxes



class TTACompose(BaseWheatTTA):

    """ author: @shonenkov """

    def __init__(self, transforms):

        self.transforms = transforms

        

    def augment(self, image):

        for transform in self.transforms:

            image = transform.augment(image)

        return image

    

    def batch_augment(self, images):

        for transform in self.transforms:

            images = transform.batch_augment(images)

        return images

    

    def prepare_boxes(self, boxes):

        result_boxes = boxes.copy()

        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)

        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)

        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)

        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)

        return result_boxes

    

    def deaugment_boxes(self, boxes):

        for transform in self.transforms[::-1]:

            boxes = transform.deaugment_boxes(boxes)

        return self.prepare_boxes(boxes)



def process_det(index, det, score_threshold=0.25):

    boxes = det[index]['boxes'].detach().cpu().numpy()[:,:4]    

    scores = det[index]['scores'].detach().cpu().numpy()



    boxes = (boxes).clip(min=0, max=511).astype(int)

    indexes = np.where(scores>score_threshold)

    boxes = boxes[indexes]

    scores = scores[indexes]

    return boxes, scores  



def make_tta_predictions(images, score_threshold=0.25):

    with torch.no_grad():

        images = torch.stack(images).float().cuda()

        predictions = []

        for tta_transform in tta_transforms:

            result = []

            det = model(tta_transform.batch_augment(images.clone()))



            for i in range(images.shape[0]):

                boxes = det[i]['boxes'].detach().cpu().numpy()[:,:4]    

                scores = det[i]['scores'].detach().cpu().numpy()

                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]



                boxes = tta_transform.deaugment_boxes(boxes.copy())

                result.append({

                    'boxes': boxes,

                    'scores': scores[indexes],

                })

            predictions.append(result)

    return predictions



def run_wbf(predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size)).tolist() for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)



tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None], 

                               [TTAVerticalFlip(), None],

                               [TTARotate90(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))



results = []



for images, image_ids in data_loader:

    predictions = make_tta_predictions(images)

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        boxes_s = (boxes*2).astype(np.int32).clip(min=0, max=1023)

        image_id = image_ids[i]

        #visualize outout

#         boxes = boxes.astype(np.int32)





#         sample = images[i].permute(1,2,0).cpu().numpy()



#         fig, ax = plt.subplots(1, 1, figsize=(16, 8))



#         for box in boxes:

            

#             cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 0, 0), 1)



#         ax.set_axis_off()

#         ax.set_title(image_id)

#         ax.imshow(sample);

        boxes_s[:, 2] = boxes_s[:, 2] - boxes_s[:, 0]

        boxes_s[:, 3] = boxes_s[:, 3] - boxes_s[:, 1]

        

        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes_s, scores)

        }

        results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('/kaggle/working/submission.csv', index=False)

test_df.head()