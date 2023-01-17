import torch

import numpy as np

import pandas as pd

from torch import nn

from PIL import Image

import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm
df = pd.read_csv("../input/global-wheat-detection/train.csv")

df['bbox'] = df['bbox'].apply(eval)

df = df.groupby("image_id")["bbox"].apply(list).reset_index()
def xywh_to_xyxy(boxes):

    if len(boxes):

        boxes = torch.tensor(boxes, dtype = torch.float32)

        boxes[..., [2,3]] = boxes[..., [2,3]] + boxes[..., [0,1]]

    else:

        boxes = torch.empty((0, 4))

    return boxes
image_id = df.loc[0, "image_id"]

boxes = xywh_to_xyxy(df.loc[0, "bbox"])

plt.figure(figsize = (10, 10))

plt.imshow(Image.open(f"../input/global-wheat-detection/train/{image_id}.jpg"))

for bbox in boxes:

    box = plt.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor = "white", fill = False, linewidth = 2)

    plt.gca().add_patch(box)
class Metric(nn.Module):

    def __init__(self, start = 0.5, end = 0.75, step = 0.05):

        super().__init__()

        self.thresholds = torch.arange(start = start, end = end, step = step)



    def box_area(self, boxes):

        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        

    def box_iou(self, boxes1, boxes2):

        area1 = self.box_area(boxes1)

        area2 = self.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]

        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]

        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union

        return iou



    def find_best_match(self, iou, threshold):

        return torch.tensor(-1).to(iou) if iou.max().lt(threshold) else iou.argmax()



    def precision(self, pred_boxes, gt_boxes, threshold):

        ious = self.box_iou(pred_boxes, gt_boxes)

        fp = 0.0

        for i in range(ious.size(0)):

            idx = self.find_best_match(ious[i], threshold)

            if idx.ge(0):

                ious[:, idx] = -1

            else:

                fp += 1.0

        tp = ious.min(dim = 0).values.eq(-1).sum()

        fn = ious.max(dim = 0).values.gt(-1).sum()

        return tp / (tp + fp + fn)



    def avg_precision(self, pred_boxes, gt_boxes):

        total_precision = torch.tensor(0.0).to(gt_boxes)

        for threshold in self.thresholds:

            total_precision += self.precision(pred_boxes, gt_boxes, threshold)

        return total_precision / self.thresholds.numel()



    def forward(self, outputs, targets):

        total = 0.0

        for output, target in tqdm(zip(outputs, targets), total = len(targets)):

            if len(output)*len(target) > 0:

                total += self.avg_precision(output, target).item()

        return total / len(targets)
metric = Metric(start = 0.5, end = 0.75, step = 0.05)

boxes = [xywh_to_xyxy(boxes) for boxes in df.bbox.values]

metric(boxes, boxes)
class WeightedBoxFusion(nn.Module):

    def __init__(self, iou_threshold = 0.6, skip_threshold = 0.55):

        super().__init__()

        self.iou_threshold = iou_threshold

        self.skip_threshold = skip_threshold



    def box_area(self, boxes):

        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])



    def box_iou(self, boxes1, boxes2):

        area1 = self.box_area(boxes1)

        area2 = self.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]

        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]

        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union

        return iou



    def cluster_boxes(self, boxes, iou_threshold):

        ious = self.box_iou(boxes, boxes)

        for i in range(ious.size(0)):

            init = ious[i, i].item()

            ious[i, i] = -1

            args = ious[i].ge(iou_threshold).nonzero().view(-1)

            ious[args, :] = -1

            ious[:, args] = -1

            ious[i, args] = 1

            ious[i, i] = init

        ious[ious.ne(1)] = 0

        return ious.bool()



    def weight_boxes(self, clustered_masks, boxes, scores, weights):

        weighted_boxes = []

        weighted_scores = []

        for mask in clustered_masks:

            if mask.sum() > 0:

                weighted_box = (boxes[mask] * scores[mask][..., None]).sum(dim = 0) / scores[mask].sum()

                weighted_score = scores[mask].sum() / max(weights.sum(), weights.mean() * mask.sum())

                weighted_boxes.append(weighted_box)

                weighted_scores.append(weighted_score)

        if sum(map(len, weighted_boxes)):

            weighted_boxes = torch.stack(weighted_boxes)

            weighted_scores = torch.stack(weighted_scores)

            indices = weighted_scores.argsort(descending = True)

            weighted_boxes = weighted_boxes[indices]

            weighted_scores = weighted_scores[indices]

        else:

            weighted_boxes = torch.empty((0, 4))

            weighted_scores = torch.empty((0,))

        return weighted_boxes, weighted_scores



    def filter_boxes(self, boxes_list, scores_list, weights, skip_threshold):

        assert len(boxes_list) == len(scores_list) == len(weights)

        for i, weight in enumerate(weights):

            mask = scores_list[i] > skip_threshold

            boxes_list[i] = boxes_list[i][mask]

            scores_list[i] = scores_list[i][mask]

            scores_list[i] = scores_list[i] * weight

        if sum(map(len, boxes_list)):

            boxes = torch.cat(boxes_list).float()

            scores = torch.cat(scores_list).float()

            indices = scores.argsort(descending = True)

            boxes = boxes[indices]

            scores = scores[indices]

        else:

            boxes = torch.empty((0, 4))

            scores = torch.empty((0,))

        return boxes, scores



    def forward(self, boxes_list, scores_list, weights = None):

        if weights is None: weights = np.ones(len(scores_list))

        weights = np.array(weights)

        boxes, scores = self.filter_boxes(boxes_list, scores_list, weights, self.skip_threshold)

        clustered_masks = self.cluster_boxes(boxes, self.iou_threshold)

        weighted_boxes, weighted_scores = self.weight_boxes(clustered_masks, boxes, scores, weights)

        return weighted_boxes, weighted_scores
wbf = WeightedBoxFusion(iou_threshold = 0.6, skip_threshold = 0.55)

boxes = [xywh_to_xyxy(boxes) for boxes in df.bbox.values]

scores = [torch.ones((len(boxes),)) for boxes in df.bbox.values]

# WBF over folds (Copyied same boxes 5 times)

folds_boxes = [boxes for _ in range(5)]

folds_scores = [scores for _ in range(5)]

wbf_boxes, wbf_scores = [], []

for boxes_list, scores_list in tqdm(zip(zip(*folds_boxes), zip(*folds_scores)), total = len(boxes)):

    img_boxes, img_scores = wbf(list(boxes_list), list(scores_list))

    wbf_boxes.append(img_boxes); wbf_scores.append(img_scores)
image_id = df.loc[0, "image_id"]

boxes = wbf_boxes[0]

plt.figure(figsize = (10, 10))

plt.imshow(Image.open(f"../input/global-wheat-detection/train/{image_id}.jpg"))

for bbox in boxes:

    box = plt.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor = "white", fill = False, linewidth = 2)

    plt.gca().add_patch(box)