import glob

import os

from torch.utils.data import Dataset,DataLoader

import cv2

import numpy as np

import torchvision

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import itertools

import torch

import time

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2


class MyDataset(Dataset):

    def __init__(self, path, transform=None):

        self.image_paths = glob.glob(os.path.join(path, '*.jpg'))

        self.transform = transform



    def __getitem__(self, index):

        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)  # reading an image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # changing color space BGR --> RGB

        image /= 255.0

        if self.transform:

            data = {'image': image}

            data = self.transform(**data)

        return data['image'], self.image_paths[index]  # return image tensor , path to image



    def __len__(self):

        return len(self.image_paths)



    



# reference : XXX need to add

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



        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)  # reading an image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # changing color space BGR --> RGB

        image /= 255.0



        boxes = records[['x', 'y', 'w', 'h']].to_numpy()

        area = (boxes[:, 3]) * (boxes[:, 2])  # Calculating area of boxes W * H

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # upper coordinate X + W

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # lower coordinate Y + H

        area = torch.as_tensor(area, dtype=torch.float32)



        # there is only one class

        labels = torch.ones((records.shape[0],), dtype=torch.int64)



        # suppose all instances are not crowd

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)



        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': area, 'iscrowd': iscrowd}



        if self.transforms:

            sample = {

                'image': image,

                'bboxes': target['boxes'],

                'labels': labels

            }

            sample = self.transforms(**sample)

            image = sample['image']



            target['boxes'] = torch.tensor(sample['bboxes']).float()

            return image, target, f'{self.image_dir}/{image_id}.jpg'  # image Tensor , target with boxes , path to image



    def __len__(self):

        return self.image_ids.shape[0]



class DatasetUtils:

    @staticmethod

    def collate_fn(batch):

        return tuple(zip(*batch))



    @staticmethod

    def preprocessing_csv(data_frame):

        # seperate bbox to x,y,w,h columns

        seperator = lambda x: np.fromstring(x[1:-1], sep=',')

        bbox = np.stack(data_frame['bbox'].apply(seperator))

        for i, dim in enumerate(['x', 'y', 'w', 'h']):

            data_frame[dim] = bbox[:, i]

        data_frame.drop(columns='bbox', inplace=True)



    @staticmethod

    def splitData(all_data_records: pd.DataFrame, test_size=0.33):

        image_ids = all_data_records['image_id'].unique()

        train_ids, valid_ids = train_test_split(image_ids, test_size=test_size, random_state=42)

        valid_df = all_data_records[all_data_records['image_id'].isin(valid_ids)]

        train_df = all_data_records[all_data_records['image_id'].isin(train_ids)]

        return train_df, valid_df



class Desnet121:

    @staticmethod

    def getNet():

        # load a pre-trained model for classification and return

        # only the features

        backbone = torchvision.models.densenet121(pretrained=False).features

        backbone.out_channels = 1024

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),

                                           aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],

                                                        output_size=7,

                                                        sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model

        densenet121 = FasterRCNN(backbone,

                                 num_classes=2,

                                 rpn_anchor_generator=anchor_generator,

                                 box_roi_pool=roi_pooler)



        densenet121.__name__ = "densenet121"



        return densenet121



class FpnResenet50:

    @staticmethod

    def getNet():

        fpn_resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False)

        num_classes = 2  # 1 class (wheat) + background

        # get number of input features for the classifier

        in_features = fpn_resnet.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one

        fpn_resnet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



        fpn_resnet.__name__ = "fpn_resnet"



        return fpn_resnet



class EvalUtils:

    @staticmethod

    def _predictEnsemble(models, device, image, detection_threshold=0.9, iou_thr=0.5):

        boxes_list = []

        scores_list = []

        image = image.to(device)

        for model in models:

            model.to(device)

            model.eval()

            with torch.no_grad():

                outputs = model(image)

                boxes = outputs[0]['boxes'].data.detach().cpu().numpy()

                scores = outputs[0]['scores'].data.detach().cpu().numpy()

                boxes = boxes[scores >= detection_threshold]

                scores = scores[scores >= detection_threshold]

                boxes_list.append(boxes)

                scores_list.append(scores)



        boxes_list = list(itertools.chain.from_iterable(boxes_list))  # flat array

        boxNumpy = np.array(boxes_list)

        scores_list = list(itertools.chain.from_iterable(scores_list))  # flat array

        scoreNumpy = np.array(scores_list)

        boxesTensor = torch.tensor(boxNumpy)

        scoreTensor = torch.tensor(scoreNumpy)



        indexKeep = torchvision.ops.nms(boxesTensor, scoreTensor, iou_thr)

        boxesTensor = boxesTensor[indexKeep]

        scoreTensor = scoreTensor[indexKeep]



        return boxesTensor, scoreTensor



    @staticmethod

    def evalEnsembleModels(models, device, data_loader, detection_threshold=0.9, iou_thr=0.5):



        d = next(iter(data_loader))

        trainMode = True

        if len(d) == 2:

            trainMode = False



        results = []

        for data in data_loader:

            if trainMode:

                images, targets, image_path = data

                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                images = list(image.to(device) for image in images)

                with torch.no_grad():

                    for image, target in zip(images, targets):

                        image = image.unsqueeze(0)

                        boxes_pred, score_pred = EvalUtils._predictEnsemble(models, device, image,

                                                                            detection_threshold=detection_threshold,

                                                                            iou_thr=iou_thr)

                        gt_boxes = target['boxes'].cpu().numpy()

                        image = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

                        results.append((image, boxes_pred, score_pred, gt_boxes))

            else:

                images, image_paths = data

                images = list(image.to(device) for image in images)

                with torch.no_grad():

                    for image, image_path in zip(images, image_paths):

                        image = image.unsqueeze(0)

                        boxes_pred, score_pred = EvalUtils._predictEnsemble(models, device, image,

                                                                            detection_threshold=detection_threshold,

                                                                            iou_thr=iou_thr)

                        image = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

                        results.append((image, boxes_pred, score_pred, image_path))



        return results



    

    


class Metrics:

    @staticmethod

    def calculate_iou(gt, pr, form='pascal_voc') -> float:

        """Calculates the Intersection over Union.



        Args:

            gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box

            pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box

            form: (str) gt/pred coordinates format

                - pascal_voc: [xmin, ymin, xmax, ymax]

                - coco: [xmin, ymin, w, h]

        Returns:

            (float) Intersection over union (0.0 <= iou <= 1.0)

        """

        if form == 'coco':

            gt = gt.copy()

            pr = pr.copy()



            gt[2] = gt[0] + gt[2]

            gt[3] = gt[1] + gt[3]

            pr[2] = pr[0] + pr[2]

            pr[3] = pr[1] + pr[3]



        # Calculate overlap area

        dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1



        if dx < 0:

            return 0.09

        dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1



        if dy < 0:

            return 0.0



        overlap_area = dx * dy



        # Calculate union area

        union_area = (

                (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +

                (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -

                overlap_area

        )



        return overlap_area / union_area



    @staticmethod

    def find_best_match(gts, pred, pred_idx, threshold=0.5, form='pascal_voc', ious=None) -> int:

        """Returns the index of the 'best match' between the

        ground-truth boxes and the prediction. The 'best match'

        is the highest IoU. (0.0 IoUs are ignored).



        Args:

            gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes

            pred: (List[Union[int, float]]) Coordinates of the predicted box

            pred_idx: (int) Index of the current predicted box

            threshold: (float) Threshold

            form: (str) Format of the coordinates

            ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.



        Return:

            (int) Index of the best match GT box (-1 if no match above threshold)

        """

        best_match_iou = -np.inf

        best_match_idx = -1

        for gt_idx in range(len(gts)):



            if gts[gt_idx][0] < 0:

                # Already matched GT-box

                continue



            iou = -1 if ious is None else ious[gt_idx][pred_idx]



            if iou < 0:

                iou = Metrics.calculate_iou(gts[gt_idx], pred, form=form)



                if ious is not None:

                    ious[gt_idx][pred_idx] = iou



            if iou < threshold:

                continue



            if iou > best_match_iou:

                best_match_iou = iou

                best_match_idx = gt_idx



        return best_match_idx



    @staticmethod

    def calculate_precision(gts, preds, threshold=0.5, form='coco', ious=None) -> float:

        """Calculates precision for GT - prediction pairs at one threshold.



        Args:

            gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes

            preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,

                   sorted by confidence value (descending)

            threshold: (float) Threshold

            form: (str) Format of the coordinates

            ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.



        Return:

            (float) Precision

        """

        n = len(preds)

        tp = 0

        fp = 0



        for pred_idx in range(n):



            best_match_gt_idx = Metrics.find_best_match(gts, preds[pred_idx], pred_idx,

                                                        threshold=threshold, form=form, ious=ious)



            if best_match_gt_idx >= 0:

                # True positive: The predicted box matches a gt box with an IoU above the threshold.

                tp += 1

                # Remove the matched GT box

                gts[best_match_gt_idx] = -1

            else:

                # No match

                # False positive: indicates a predicted box had no associated gt box.

                fp += 1



        # False negative: indicates a gt box had no associated predicted box.

        fn = (gts.sum(axis=1) > 0).sum()



        return tp / (tp + fp + fn)



    @staticmethod

    def calculate_image_precision(gts, preds, thresholds=(0.5,), form='coco') -> float:

        """Calculates image precision.



        Args:

            gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes

            preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,

                   sorted by confidence value (descending)

            thresholds: (float) Different thresholds

            form: (str) Format of the coordinates



        Return:

            (float) Precision

        """

        n_threshold = len(thresholds)

        image_precision = 0.0



        ious = np.ones((len(gts), len(preds))) * -1

        # ious = None



        for threshold in thresholds:

            precision_at_threshold = Metrics.calculate_precision(gts.copy(), preds, threshold=threshold,

                                                                 form=form, ious=ious)

            image_precision += precision_at_threshold / n_threshold



        return image_precision



def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], round(j[1][0]), round(j[1][1]), round(j[1][2]), round(j[1][3])))

    return " ".join(pred_strings)







def createSubmission(models, device, data_loader, detection_threshold=0.9, iou_thr=0.5):

    SUBMISSION_PATH = '/kaggle/working'

    submission_id = 'submission'

    final_csv = []

    results = EvalUtils.evalEnsembleModels(models, device, data_loader, detection_threshold=detection_threshold, iou_thr=iou_thr)

    for (image, boxes, scores, image_path) in results:

        boxes = boxes.detach().cpu().numpy()

        if boxes.shape[0] > 0 :

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            image_id = image_path.split("/")[-1]

            result = [image_id,format_prediction_string(boxes, scores)]

            final_csv.append(result)



    cur_submission_path = os.path.join(SUBMISSION_PATH, '{}.csv'.format(submission_id))

    sample_submission = pd.DataFrame(final_csv, columns=["image_id","PredictionString"])

    sample_submission.to_csv(cur_submission_path, index=False)

    submission_df = pd.read_csv(cur_submission_path)

    return submission_df













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





class TrainUtils:

    @staticmethod

    def trainModels(models, train_dataloader, valid_dataloader, device, num_epochs=1, valid_pred_min=0.65):

        for model in models:

            print("Starting train model ", str(model.__name__))

            model.to(device)

            # construct an optimizer

            params = [p for p in model.parameters() if p.requires_grad]

            optimizer = torch.optim.SGD(params, lr=0.005,

                                        momentum=0.9, weight_decay=0.0005)



            train_hist = Averager()

            t = 0

            for epoch in range(num_epochs):

                model.train()

                train_hist.reset()

                for images, targets,image_path in train_dataloader:

                    model.train()

                    images = list(image.to(device) for image in images)

                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets)

                    losses = sum(loss for loss in loss_dict.values())

                    train_loss = losses.item()

                    train_hist.send(train_loss)

                    optimizer.zero_grad()

                    losses.backward()

                    optimizer.step()

                    t += 1



                model.eval()

                validation_image_precisions = []

                iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]

                for images, targets,image_path in valid_dataloader:

                    images = list(image.to(device) for image in images)

                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



                    with torch.no_grad():

                        outputs = model(images)



                    for i, image in enumerate(images):

                        boxes = outputs[i]['boxes'].data.cpu().numpy()

                        scores = outputs[i]['scores'].data.cpu().numpy()

                        gt_boxes = targets[i]['boxes'].cpu().numpy()

                        preds_sorted_idx = np.argsort(scores)[::-1]

                        preds_sorted = boxes[preds_sorted_idx]

                        image_precision = Metrics.calculate_image_precision(preds_sorted,

                                                                            gt_boxes,

                                                                            thresholds=iou_thresholds,

                                                                            form='coco')

                        validation_image_precisions.append(image_precision)



                valid_prec = np.mean(validation_image_precisions)

                print("Validation Precision: {0:.4f}".format(valid_prec))



                # print training/validation statistics

                print('Epoch: {} \tTraining Loss: {:.6f}'.format(

                    epoch,

                    train_loss

                ))



                if valid_prec >= valid_pred_min:

                    print('Validation precision increased({:.6f} --> {:.6f}).  Saving model ...'.format(

                        valid_pred_min,

                        valid_prec))

                    torch.save(model.state_dict(), 'faster_rrcnn_' + str(model.__name__) + '.pth')

                    valid_pred_min = valid_prec

            torch.save(model.state_dict(), 'faster_rrcnn_' + str(model.__name__) + '_' + str(time.time()) + '.pth')



class TransformsUtils:



    @staticmethod

    def get_train_transform():

        return A.Compose(

            [



                A.OneOf([

                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,

                                         val_shift_limit=0.2, p=0.9),

                    A.RandomBrightnessContrast(brightness_limit=0.2,

                                               contrast_limit=0.2, p=0.9),

                ], p=0.9),

                A.ToGray(p=0.01),

                A.GaussianBlur(p=0.3),

                A.HorizontalFlip(p=0.5),

                A.VerticalFlip(p=0.5),

                A.RandomBrightnessContrast(p=0.5),

               # A.Resize(height=512, width=512, p=1),

                ToTensorV2(p=1.0),

            ],

            p=1.0,

            bbox_params=A.BboxParams(

                format='pascal_voc',

                min_area=0,

                min_visibility=0,

                label_fields=['labels']

            )

        )



    @staticmethod

    def get_valid_transform():

        return A.Compose([

            #A.Resize(height=512, width=512, p=1),

            ToTensorV2(p=1.0)

        ])



class VisualUtils:



    # helper function for data visualization

    @staticmethod

    def visualize(**images):

        """PLot images in one row."""

        n = len(images)

        plt.figure(figsize=(16, 5))

        for i, (name, image) in enumerate(images.items()):

            plt.subplot(1, n, i + 1)

            plt.xticks([])

            plt.yticks([])

            plt.title(' '.join(name.split('_')).title())

            plt.imshow(image)

        plt.show()



    @staticmethod

    def draw_boxes(image, boxes, scores=None):

        # font

        font = cv2.FONT_HERSHEY_SIMPLEX

        # fontScale

        fontScale = 0.7

        # Blue color in BGR

        color = (255, 0, 0)

        # Line thickness of 1 px

        thickness = 2

        if scores is not None:

            for box, score in zip(boxes, scores):

                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 20, 2)

                cv2.putText(image, str(score.item())[:5], (int(box[0]), int(box[1]) - 10), font, fontScale, color,

                            thickness, cv2.LINE_AA)

        else:

            for box in boxes:

                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 20, 2)



        return image



    @staticmethod

    def visualizePrediction(image, pre_boxes, scores, get_boxes=None):



        imageCopy = image.copy()

        imageCopy2 = image.copy()



        image_pred = VisualUtils.draw_boxes(imageCopy, pre_boxes, scores)



        if get_boxes is not None:

            image_get = VisualUtils.draw_boxes(imageCopy2, get_boxes)

            VisualUtils.visualize(

                Original=image,

                Ground_of_true=image_get,

                Predict=image_pred)

        else:

            VisualUtils.visualize(

                Original=image,

                Predict=image_pred)



# all_wheat_dataset_train = pd.read_csv('../input/global-wheat-detection/train.csv')

# DatasetUtils.preprocessing_csv(all_wheat_dataset_train)

# all_wheat_dataset_train.head()



# train_df,valid_df = DatasetUtils.splitData(all_data_records=all_wheat_dataset_train)





# # take only first 1000 records

# train_df = train_df[:1000]

# valid_df = valid_df[:1000]



# train_path = "../input/global-wheat-detection/train"



# train_dataset = WheatDataset(train_df,train_path,TransformsUtils.get_train_transform())

# valid_dataset = WheatDataset(valid_df,train_path,TransformsUtils.get_train_transform())



# train_dataloader = DataLoader(train_dataset,batch_size=2,collate_fn=DatasetUtils.collate_fn,num_workers=8)

# valid_dataloader = DataLoader(valid_dataset,batch_size=2,collate_fn=DatasetUtils.collate_fn,num_workers=4)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device
resnet50 = FpnResenet50.getNet()

desnet121 = Desnet121.getNet()
#TrainUtils.trainModels([desnet121], train_dataloader, valid_dataloader, device, num_epochs=50, valid_pred_min=0.50)

#ls -l ../working/
resnet50_path = '../input/faster-rrcnn-fpn-resnet-50-epoches/faster_rrcnn_fpn_resnet.pth'

desnet121_path = '../input/faster-rrcnn-densenet121/faster_rrcnn_densenet121.pth'
resnet50.load_state_dict(torch.load(resnet50_path))

desnet121.load_state_dict(torch.load(desnet121_path))

# results = EvalUtils.evalEnsembleModels([resnet50,desnet121], device, valid_dataloader, detection_threshold=0.9, iou_thr=0.3)



# for (image, boxes_pred, score_pred, gt_boxes) in results:

#     VisualUtils.visualizePrediction(image, boxes_pred, score_pred, get_boxes=gt_boxes)



query_folder = '../input/global-wheat-detection/test'

dataset_query = MyDataset(query_folder, transform=TransformsUtils.get_valid_transform())

test_loader = DataLoader(dataset_query,batch_size=1)



results =  EvalUtils.evalEnsembleModels([resnet50,desnet121], device, test_loader, detection_threshold=0.5, iou_thr=0.3)



#%%



for (image, boxes_pred, score_pred, image_path) in results:

    VisualUtils.visualizePrediction(image, boxes_pred, score_pred, get_boxes=None)

submission = createSubmission([resnet50,desnet121], device, test_loader, detection_threshold=0.5, iou_thr=0.3)
submission