import torch

from torch import nn

import os

import cv2

from PIL import Image
bbox = torch.tensor([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=torch.float32) # [y1, x1, y2, x2] format

labels = torch.tensor([6, 8], dtype=torch.int8)
class Utils:

    @staticmethod

    def loc_to_bbox(src_bbox, loc):

        if src_bbox.shape[0] == 0:

            return torch.zeros((0, 4), dtype=loc.dtype)



        src_height = src_bbox[:, 2] - src_bbox[:, 0]

        src_width = src_bbox[:, 3] - src_bbox[:, 1]

        src_ctr_y = src_bbox[:, 0] + 0.5 * src_height

        src_ctr_x = src_bbox[:, 1] + 0.5 * src_width



        dy = loc[:, 0::4]

        dx = loc[:, 1::4]

        dh = loc[:, 2::4]

        dw = loc[:, 3::4]



        ctr_y = dy * src_height.view(-1, 1) + src_ctr_y.view(-1, 1)

        ctr_x = dx * src_width.view(-1, 1) + src_ctr_x.view(-1, 1)

        h = xp.exp(dh) * src_height.view(-1, 1)

        w = xp.exp(dw) * src_width.view(-1, 1)



        dst_bbox = torch.zeros(loc.shape, dtype=loc.dtype)

        dst_bbox[:, 0::4] = ctr_y - 0.5 * h

        dst_bbox[:, 1::4] = ctr_x - 0.5 * w

        dst_bbox[:, 2::4] = ctr_y + 0.5 * h

        dst_bbox[:, 3::4] = ctr_x + 0.5 * w

        return dst_bbox



    @staticmethod

    def bbox_to_loc(src_bbox, dst_bbox):

        src_height = src_bbox[..., 2] - src_bbox[..., 0]

        src_width = src_bbox[..., 3] - src_bbox[..., 1]

        src_ctr_y = src_bbox[..., 0] + 0.5 * src_height

        src_ctr_x = src_bbox[..., 1] + 0.5 * src_width



        dst_height = dst_bbox[..., 2] - dst_bbox[..., 0]

        dst_width = dst_bbox[..., 3] - dst_bbox[..., 1]

        dst_ctr_y = dst_bbox[..., 0] + 0.5 * dst_height

        dst_ctr_x = dst_bbox[..., 1] + 0.5 * dst_width



        eps = torch.Tensor([1e-7])

        src_height = torch.max(src_height, eps)

        src_width = torch.max(src_width, eps)

        dy = (dst_ctr_y - src_ctr_y) / src_height

        dx = (dst_ctr_x - src_ctr_x) / src_width

        dh = torch.log(dst_height / src_height)

        dw = torch.log(dst_width / src_width)

        locs = torch.stack((dy, dx, dh, dw), dim = 2)

        return locs



    @staticmethod

    def non_maximum_supression(roi, thresh):

        y1 = roi[:, 0]

        x1 = roi[:, 1]

        y2 = roi[:, 2]

        x2 = roi[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        order = score.argsort(dim = -1, descending = True)

        keep = []

        while order.size > 0:

            i = order[0]

            keep.append(i)

            xx1 = torch.max(x1[i], x1[order[1:]])

            yy1 = torch.max(y1[i], y1[order[1:]])

            xx2 = torch.max(x2[i], x2[order[1:]])

            yy2 = torch.max(y2[i], y2[order[1:]])

            w = torch.max(0.0, xx2 - xx1 + 1)

            h = torch.max(0.0, yy2 - yy1 + 1)

            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]

            order = order[inds + 1]

        return keep



    def calculate_ious(src_bbox, dst_bbox):

        src = src_bbox.unsqueeze(dim = -2).repeat(1, 1, dst_bbox.shape[-2], 1)

        dst = dst_bbox.unsqueeze(dim = -3).repeat(1, src_bbox.shape[-2], 1, 1)

        src_area = (src[..., 2] - src[..., 0]) * (src[..., 3] - src[..., 1])

        dst_area = (dst[..., 2] - dst[..., 0]) * (dst[..., 3] - dst[..., 1])

        intersection_left = torch.max(src[..., 0], dst[..., 0])

        intersection_top = torch.max(src[..., 1], dst[..., 1])

        intersection_right = torch.min(src[..., 2], dst[..., 2])

        intersection_bottom = torch.min(src[..., 3], dst[..., 3])

        intersection_width = torch.clamp(intersection_right - intersection_left, min=0)

        intersection_height = torch.clamp(intersection_bottom - intersection_top, min=0)

        intersection_area = intersection_width * intersection_height

        return intersection_area / (src_area + dst_area - intersection_area)
class Anchors:

    def __init__(self, batch_size = 2, img_size = (800, 600), sub_sample = 16, ratios = [0.5, 1, 2], scales = [8, 16, 32]):

        self.batch_size = batch_size

        self.img_size = img_size

        self.sub_sample = sub_sample

        self.ratios = ratios

        self.scales = scales

        self.dtype = torch.float32

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generate()



    def generate(self):

        ctr_x = torch.arange(start = self.sub_sample/2, end = self.img_size[0] + 1, step = self.sub_sample, dtype = self.dtype, device = self.device)

        ctr_y = torch.arange(start = self.sub_sample/2, end = self.img_size[1] + 1, step = self.sub_sample, dtype = self.dtype, device = self.device)

        ratios = torch.tensor(self.ratios, dtype = self.dtype, device = self.device)

        scales = torch.tensor(self.scales, dtype = self.dtype, device = self.device)

        ctr_x, ctr_y, ratios, scales = torch.meshgrid(ctr_x, ctr_y, ratios, scales)

        ctr_x = ctr_x.reshape(-1)

        ctr_y = ctr_y.reshape(-1)

        ratios = ratios.reshape(-1)

        scales = scales.reshape(-1)

        widths = scales * torch.sqrt(1 / ratios) * self.sub_sample

        heights = scales * torch.sqrt(ratios) * self.sub_sample

        anchors = torch.stack((ctr_x-0.5*widths, ctr_y-0.5*heights, ctr_x+0.5*widths, ctr_y+0.5*heights), dim=1)

        self.anchors = anchors.repeat(self.batch_size, 1, 1)



    def target(self, bbox, pos_iou_threshold  = 0.7, neg_iou_threshold = 0.3, pos_ratio = 0.5, n_sample = 256):

        locs = torch.zeros((*self.anchors.shape[:-1], 4), dtype = torch.float32)

        self.labels = torch.zeros((*self.anchors.shape[:-1],), dtype = torch.int32)

        inside = self.inside(self.anchors, 0, 0, 800, 800)

        self.labels[inside] = -1

        valid_anchors = self.anchors[inside].reshape(self.batch_size, -1, 4)



        ious = Utils.calculate_ious(valid_anchors, bbox)



        gt_row_max_ious, row_argmax_ious = ious.max(axis = 1, keepdim = True)

        gt_row_argmax_ious = (ious == gt_row_max_ious).nonzero()

        self.labels[gt_row_argmax_ious[...,-1]] = 1



        col_max_ious, col_argmax_ious = ious.max(axis = 2, keepdim = True)

        self.labels[(col_max_ious < neg_iou_threshold).nonzero()[...,-1]] = 0

        self.labels[(col_max_ious >= pos_iou_threshold).nonzero()[...,-1]] = 1



        n_pos = pos_ratio * n_sample

        pos_index = (labels == 1).nonzero()

        disable_index = torch.randperm(len(pos_index))[:int(min(len(pos_index), n_pos))]

        self.labels[:,disable_index] = -1

        n_neg = n_sample - n_pos

        neg_index = (labels == 0).nonzero()

        disable_index = torch.randperm(len(neg_index))[:int(min(len(neg_index), n_neg))]

        self.labels[:,disable_index] = -1

        locs[inside] = Utils.bbox_to_loc(valid_anchors, bbox[torch.arange(2).unsqueeze(1),col_argmax_ious.squeeze(2)]).reshape(-1,4)

        return valid_anchors, locs, labels



    def inside(self, bboxes, left, top, right, bottom):

        return ((bboxes[..., 0] >= left) & (bboxes[..., 1] >= top) &

            (bboxes[..., 2] <= right) & (bboxes[..., 3] <= bottom))
a = Anchors()

# a.generate()

a.anchors

a.target(bbox.repeat(2,1,1))
class Proposal:

    def __init__(self, locs, scores):

        self.scores = scores

        anchor = Anchor()

        # Convert anchors into proposal via bbox transformations.

        self.roi = loc2bbox(An, loc)



    def _filter(self, n_pre_nms, n_post_nms, inplace = False, nms_thresh = 0.7, img_size = (800, 600), min_size = 16, scale=1):

        # Clip predicted boxes to image.

        self.roi[:, slice(0, 4, 2)] = np.clip(self.roi[:, slice(0, 4, 2)], 0, img_size[0])

        self.roi[:, slice(1, 4, 2)] = np.clip(self.roi[:, slice(1, 4, 2)], 0, img_size[1])



        # Remove predicted boxes with either height or width < threshold.

        min_size = self.min_size * scale

        hs = self.roi[:, 2] - self.roi[:, 0]

        ws = self.roi[:, 3] - self.roi[:, 1]

        keep = np.where((hs >= min_size) & (ws >= min_size))[0]

        self.roi = self.roi[keep, :]

        self.scores = self.scores[keep]



        # Sort all (proposal, score) pairs by score from highest to lowest.

        # Take top pre_nms_topN (e.g. 6000).

        order = self.score.argsort(dim = -1, descending = True)

        if n_pre_nms > 0: order = order[:n_pre_nms]

        roi = roi[order, :]



        # Apply nms (e.g. threshold = 0.7).

        # Take after_nms_topN (e.g. 300).

        keep = non_maximum_suppression(

            cp.ascontiguousarray(cp.asarray(roi)),

            thresh=self.nms_thresh)

        if n_post_nms > 0: keep = keep[:n_post_nms]

        if inplace:

            self.roi = self.roi[keep]

        else:

            return self.roi[keep]



    def roi():

        return self.roi



    def target(self, bbox, labels, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_range = .5, neg_iou_thresh_lo=0.0):

        ious = calculete_ious(self.roi, bbox)

        argmax_ious = ious.argmax(axis=1)

        max_ious = ious.max(axis=1)

        gt_roi_label = labels[argmax_ious]



        pos_index = np.where(max_iou >= pos_iou_thresh)[0]

        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))

        if pos_index.size > 0:

            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        neg_index = np.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]

        neg_roi_per_this_image = n_sample - pos_roi_per_this_image

        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))

        if  neg_index.size > 0 :

            neg_index = np.random.choice(neg_index, size = neg_roi_per_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)



        gt_roi_labels[neg_roi_per_this_image:] = 0  # negative labels --> 0

        gt_roi_labels = gt_roi_label[keep_index]

        sample_roi = self.roi[keep_index] # value of roi

        gt_roi_locs = bbox_to_loc(bbox[argmax_ious[keep_index]], sample_roi)

        return sample_roi, gt_roi_locs, gt_roi_labels
class RPN(nn.Module):

    def __init__(self, in_channels, mid_channels, n_anchor):

        super(RPN, self).__init__()

        # conv sliding layer

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        self.conv1.weight.data.normal_(0, 0.01)

        self.conv1.bias.data.zero_()



        # Regression layer

        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        self.loc.weight.data.normal_(0, 0.01)

        self.loc.bias.data.zero_()



        # classification layer

        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

        self.score.weight.data.normal_(0, 0.01)

        self.score.bias.data.zero_()



    def forward(self, out_map):

        conv1 = self.conv1(out_map)

        loc = self.loc(conv1)

        score = self.score(conv1)

        proposals = Proposal(loc, score)

        proposals._filter(12000, 6000, inplace = True)

        return proposals
class FasterRCNN(nn.Module):

    def __init__(self):

        super(FasterRCNN, self).__init__()
import numpy as np

center_ys = np.linspace(start=0, stop=256, num=2 + 2)[1:-1]

center_xs = np.linspace(start=0, stop=128, num=2 + 2)[1:-1]

ratios = np.array([[1,1],[2,1],[1,2]])

ratios = ratios[:, 0] / ratios[:, 1]

sizes = np.array([128, 256, 512])



# NOTE: it's important to let `center_ys` be the major index (i.e., move horizontally and then vertically) for consistency with 2D convolution

# giving the string 'ij' returns a meshgrid with matrix indexing, i.e., with shape (#center_ys, #center_xs, #ratios)

center_ys, center_xs, ratios, sizes = np.meshgrid(center_ys, center_xs, ratios, sizes, indexing='ij')
from PIL import Image

class TrainDataset:

    def __init__(self, base_path):

        self.images = []

        self.labels = []

        for folder in os.listdir(base_path):

            with open ("%s/%s/coor.txt"%(base_path,folder)) as f:

                coor = eval(f.read())

            for file in os.listdir("%s/%s"%(base_path,folder)):

                if file in coor:

                    self.images.append("%s/%s/%s"%(base_path,folder,file))

                    self.labels.append(coor[file])

    def __len__(self):

        return len(self.images)

    def __getitem__(self,idx):

        img = Image.open(self.images[idx])

        label = self.labels[idx][:4]

        return np.array(img),label            
t = TrainDataset('images')

t[7]

#cv2.imshow("img", t[7][0])

#cv2.waitKey(0)

#cv2.destroyAllWindows()