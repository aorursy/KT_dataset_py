import numpy as np

import cv2

import os

import re

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

import math

import torch

import torch.nn as nn

import torchvision

import torch.nn.functional as F

from scipy.spatial import distance

import os.path as osp

import glob

import gc

import timeit
# Hyper-params

MODEL_PATH = None

dataset_path = '../input/fisheye'

input_size = 640

MODEL_SCALE = 4

batch_size = 8

output_size = input_size//MODEL_SCALE

model_name = "resnet34"

TRAIN = True # True for training
def gaussian_radius(det_size, min_overlap=0.7):

    height, width = det_size



    a1  = 1

    b1  = (height + width)

    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)

    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)

    r1  = (b1 + sq1) / 2



    a2  = 4

    b2  = 2 * (height + width)

    c2  = (1 - min_overlap) * width * height

    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)

    r2  = (b2 + sq2) / 2



    a3  = 4 * min_overlap

    b3  = -2 * min_overlap * (height + width)

    c3  = (min_overlap - 1) * width * height

    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)

    r3  = (b3 + sq3) / 2

    return min(r1, r2, r3)



def draw_umich_gaussian(heatmap, center, radius, k=1):

    diameter = 2 * radius + 1

    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)



    x, y = int(center[0]), int(center[1])



    height, width = heatmap.shape[0:2]



    left, right = min(x, radius), min(width - x, radius + 1)

    top, bottom = min(y, radius), min(height - y, radius + 1)



    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]

    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug

        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap



def gaussian2D(shape, sigma=1):

    m, n = [(ss - 1.) / 2. for ss in shape]

    y, x = np.ogrid[-m:m+1,-n:n+1]



    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h



def get_affine_transform(center,

                         scale,

                         rot,

                         output_size,

                         shift=np.array([0, 0], dtype=np.float32),

                         inv=0):

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):

        scale = np.array([scale, scale], dtype=np.float32)



    scale_tmp = scale

    src_w = scale_tmp[0]

    dst_w = output_size[0]

    dst_h = output_size[1]



    rot_rad = np.pi * rot / 180

    src_dir = get_dir([0, src_w * -0.5], rot_rad)

    dst_dir = np.array([0, dst_w * -0.5], np.float32)



    src = np.zeros((3, 2), dtype=np.float32)

    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift

    src[1, :] = center + src_dir + scale_tmp * shift

    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]

    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir



    src[2:, :] = get_3rd_point(src[0, :], src[1, :])

    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])



    if inv:

        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    else:

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))



    return trans





def affine_transform(pt, t):

    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T

    new_pt = np.dot(t, new_pt)

    return new_pt[:2]



def get_3rd_point(a, b):

    direct = a - b

    return b + np.array([-direct[1], direct[0]], dtype=np.float32)





def get_dir(src_point, rot_rad):

    sn, cs = np.sin(rot_rad), np.cos(rot_rad)



    src_result = [0, 0]

    src_result[0] = src_point[0] * cs - src_point[1] * sn

    src_result[1] = src_point[0] * sn + src_point[1] * cs



    return src_result
# functions for plotting results

def showbox(img, hm, regr, thresh=0.9):

    boxes, _ = pred2box(hm, regr, thresh=thresh)

    print("preds:",boxes.shape)

    sample = img



    for box in boxes:

        # upper-left, lower-right

#         cv2.circle(sample, (int(box[0]), int(box[1])), 5, (0,255,0), 5)

        cv2.rectangle(sample,

                      (int(box[0]), int(box[1])),

                      (int(box[0]+box[2]), int(box[1]+box[3])),

                      (220, 0, 0), 3)

    return sample



def showgtbox(img, hm, regr, thresh=0.9):

    boxes, _ = pred2box(hm, regr, thresh=thresh)

    print("GT boxes:", boxes.shape)

    sample = img



    for box in boxes:

        cv2.rectangle(sample,

                      (int(box[0]), int(box[1])),

                      (int(box[0]+box[2]), int(box[1]+box[3])),

                      (0, 220, 0), 3)

    return sample





def pred2box(hm, regr, thresh=0.99):

    # make binding box from heatmaps

    # thresh: threshold for logits.

    c0, c1 = np.where(hm>thresh)

    # get regressions

    boxes = []

    scores =[]

    if len(c0)> 0:

        for cx, cy  in zip(c1, c0):

            x, y, r, b = regr[:, cy, cx]

            s = max(min(hm[cy, cx]*2, 1), 0) 

            

            minx = min(cx - x, cx + r)*4

            maxx = max(cx - x, cx + r)*4

            miny = min(cy - y, cy + r)*4

            maxy = max(cy - y, cy + r)*4

            w, h = maxx - minx, maxy-miny

            if w>0 and h >0:

                scores.append(s)

                boxes.append([minx, miny, w, h])

        boxes = np.asarray(boxes, dtype=np.float32)

        scores = np.asarray(scores, dtype=np.float32)

        keep = nms(boxes[:, :4], scores, thresh)

        boxes = boxes[keep, :]

    else:

        boxes = np.empty((0, 4))

        scores = np.empty(0)

    return boxes, scores
def _get_border(border, size):

    i = 1

    while size - border // i <= border // i:

            i *= 2

    return border // i



def nms(boxes, scores, nms_thresh):

    x1 = boxes[:, 0]

    y1 = boxes[:, 1]

    x2 = boxes[:, 0] + boxes[:, 2]

    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.argsort(scores)[::-1]

    num_detections = boxes.shape[0]

    suppressed = np.zeros((num_detections,), dtype=np.bool)

    keep = []

    for _i in range(num_detections):

        i = order[_i]

        if suppressed[i]:

            continue

        keep.append(i)

        ix1 = x1[i]

        iy1 = y1[i]

        ix2 = x2[i]

        iy2 = y2[i]

        iarea = areas[i]

        for _j in range(_i + 1, num_detections):

            j = order[_j]

            if suppressed[j]:

                continue

            xx1 = max(ix1, x1[j])

            yy1 = max(iy1, y1[j])

            xx2 = min(ix2, x2[j])

            yy2 = min(iy2, y2[j])

            w = max(0, xx2 - xx1 + 1)

            h = max(0, yy2 - yy1 + 1)

            inter = w * h

            ovr = inter / (iarea + areas[j] - inter)

            if ovr >= nms_thresh:

                suppressed[j] = True

    return keep



def distance_2p(p1, p2):

    return distance.cdist([p1], [p2])[0][0]
# Wrapped heatmap function

def make_hm_regr(img, targets):

    _h, _w, _ = img.shape

    # affine transform

    c = np.array([_w / 2., _h / 2.], dtype=np.float32)

    s = max(_h, _w) * 1.0

    s = s * np.random.choice(np.arange(0.8, 2., 0.1))

    w_border = _get_border(128, _w)

    h_border = _get_border(128, _h)

    c[0] = np.random.randint(low=w_border, high=_w - w_border)

    c[1] = np.random.randint(low=h_border, high=_h - h_border)

    r = np.random.randint(-180, 180)

    

    trans_input = get_affine_transform(

            c, s, r, [input_size, input_size])

    

    inp = cv2.warpAffine(img, trans_input, 

                         (input_size, input_size),

                         flags=cv2.INTER_LINEAR)

    

    trans_output = get_affine_transform(c, s, r, [output_size, output_size])

    

    # make output heatmap for single class

    hm = np.zeros([output_size, output_size])

    # make regr heatmap 

    regr = np.zeros([1, output_size, output_size])

    reg_tlrb            = np.zeros((1 * 4, output_size, output_size), np.float32)

    reg_mask            = np.zeros((1,     output_size, output_size), np.float32)

    error = False

    if len(targets) == 0:

        return hm, reg_tlrb

    

    width_ratio = input_size/_w

    height_ratio = input_size/_h

    

    

    # try gaussian points.

    for target in targets:

        p1, p2, p3, p4 = target

        

        p1_trans = affine_transform(p1, trans_output)

        p2_trans = affine_transform(p2, trans_output)

        p3_trans = affine_transform(p3, trans_output)

        p4_trans = affine_transform(p4, trans_output)

        

        h, w = distance_2p(p1_trans, p4_trans), distance_2p(p1_trans, p2_trans)



        if h > 0 and w > 0:

            xmin = np.array([p1_trans[0], p2_trans[0], p3_trans[0], p4_trans[0]]).min()

            ymin = np.array([p1_trans[1], p2_trans[1], p3_trans[1], p4_trans[1]]).min()

            xmax = np.array([p1_trans[0], p2_trans[0], p3_trans[0], p4_trans[0]]).max()

            ymax = np.array([p1_trans[1], p2_trans[1], p3_trans[1], p4_trans[1]]).max()

            

            ct_base = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2], dtype=np.float32)

            

            if ct_base[0] > 0 and ct_base[1] > 0 and ct_base[0] < output_size and ct_base[1] < output_size:

                ct_base = np.clip(ct_base, 1, output_size - 1)

                radius = gaussian_radius((math.ceil(h), math.ceil(w)))

                radius = max(0, int(radius))

                ct_int = ct_base.astype(np.int32)

                draw_umich_gaussian(hm, [ct_int[0], ct_int[1]], radius)

                reg_mask[0,  ct_int[1],ct_int[0]] = 1.

                reg_tlrb[0:4, ct_int[1],ct_int[0]] = np.array([ct_int[0] - xmin, ct_int[1] - ymin, xmax - ct_int[0], ymax - ct_int[1]])

                                                                                                                    

    return inp, hm, reg_tlrb, reg_mask
img = cv2.imread('../input/fisheye/COLLECT/COLLECT/0.jpg')

f = open('../input/fisheye/COLLECT/COLLECT/0.txt')

targets = []

for line in f.readlines():

    _, x, y, w, h, a = line.strip().replace('\n', '').split(' ')

    x, y, w, h, a = int(x), int(y), int(w), int(h), int(a)

    a = a*np.pi/180

    C, S = np.cos(a), np.sin(a)

    R = np.asarray([[-C, -S], [S, -C]])

    pts = np.asarray([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])

    box = [((x, y) + pt @ R).astype(int) for pt in pts]

    pt1, pt2, pt3, pt4 = tuple(box[0]), tuple(box[1]), tuple(box[2]), tuple(box[3])

    targets.append([pt1, pt2, pt3, pt4])



img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img, hm, regr, regr_mask = make_hm_regr(img, targets)

img = cv2.resize(img, (input_size, input_size))

sample = img



fig, ax = plt.subplots(1, 1, figsize=(16, 8))

sample = showbox(sample, hm, regr, 0.99)

plt.imshow(sample)

plt.show()

plt.imshow(hm)
# plt.imshow(regr_mask.squeeze())
from torchvision import transforms

import random

mean=[0.5, 0.5, 0.5]

std=[0.5, 0.5, 0.5]

# mean= (0.485, 0.456, 0.406)

# std=(0.229, 0.224, 0.225)

class Normalize(object):

    def __init__(self):

        self._data_rng = np.random.RandomState(123)

        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],

                                                         dtype=np.float32)

        self._eig_vec = np.array([

                [-0.58752847, -0.69563484, 0.41340352],

                [-0.5832747, 0.00994535, -0.81221408],

                [-0.56089297, 0.71832671, 0.41158938]

        ], dtype=np.float32)

        

    def __call__(self, image):

        image = image.astype(np.float32)/255

        color_aug(self._data_rng, image, self._eig_val, self._eig_vec)

        image -= mean

        image /= std

        return image

    

def saturation_(data_rng, image, gs, gs_mean, var):

    alpha = 1. + data_rng.uniform(low=-var, high=var)

    blend_(alpha, image, gs[:, :, None])



def brightness_(data_rng, image, gs, gs_mean, var):

    alpha = 1. + data_rng.uniform(low=-var, high=var)

    image *= alpha



def contrast_(data_rng, image, gs, gs_mean, var):

    alpha = 1. + data_rng.uniform(low=-var, high=var)

    blend_(alpha, image, gs_mean)

    

def grayscale(image):

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def lighting_(data_rng, image, alphastd, eigval, eigvec):

    alpha = data_rng.normal(scale=alphastd, size=(3, ))

    image += np.dot(eigvec, eigval * alpha)



def blend_(alpha, image1, image2):

    image1 *= alpha

    image2 *= (1 - alpha)

    image1 += image2

    

def color_aug(data_rng, image, eig_val, eig_vec):

    functions = [brightness_, contrast_, saturation_]

    random.shuffle(functions)



    gs = grayscale(image)

    gs_mean = gs.mean()

    for f in functions:

        if np.random.random() < 0.4:

            f(data_rng, image, gs, gs_mean, 0.1)

    lighting_(data_rng, image, 0.1, eig_val, eig_vec)



def augment_hsv(img, hgain=0.0138, sgain=0.638, vgain=0.36):

    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains

    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    dtype = img.dtype  # uint8



    x = np.arange(0, 256, dtype=np.int16)

    lut_hue = ((x * r[0]) % 180).astype(dtype)

    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)

    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)



    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)

    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    

    return img_hsv



def add_gaussian(img, max_var = 0.5):

    '''

    max_var: variance is uniformly ditributed between 0~max_var

    '''

    var = np.random.uniform(0, max_var)

    gauss_img = img * var

    

    return gauss_img



def motion_blur(img, ksize=15): 

    # generating the kernel

    kernel_motion_blur = np.zeros((ksize, ksize))

    kernel_motion_blur[int((ksize-1)/2), :] = np.ones(ksize)

    kernel_motion_blur = kernel_motion_blur / ksize



    # applying the kernel to the input image

    blur_img = cv2.filter2D(img, -1, kernel_motion_blur)

    

    return blur_img



# pool duplicates

def pool(data):

    stride = 3

    for y in np.arange(1,data.shape[1]-1, stride):

        for x in np.arange(1, data.shape[0]-1, stride):

            a_2d = data[x-1:x+2, y-1:y+2]

            max = np.asarray(np.unravel_index(np.argmax(a_2d), a_2d.shape))            

            for c1 in range(3):

                for c2 in range(3):

                    #print(c1,c2)

                    if not (c1== max[0] and c2 == max[1]):

                        data[x+c1-1, y+c2-1] = -1

    return data



class FisheyeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path = dataset_path, transform=None):

        self.imgs_path = []

        self.targets = []

        for root_data in os.listdir(data_path):

            if root_data == 'COLLECT':

                data_dir = osp.join(data_path, root_data)

                for dir_ in os.listdir(data_dir):

                    dir_path = osp.join(data_dir, dir_)

                    for img_path in glob.glob(osp.join(dir_path, '*.jpg')):

                        txt_path = img_path.replace('.jpg', '.txt')

                        self.imgs_path.append(img_path)

                        f = open(txt_path)

                        cur_target=[]

                        for line in f.readlines():

                            _, x, y, w, h, a = line.strip().replace('\n', '').split(' ')

                            x, y, w, h, a = int(x), int(y), int(w), int(h), int(a)

                            a = a*np.pi/180

                            C, S = np.cos(a), np.sin(a)

                            R = np.asarray([[-C, -S], [S, -C]])

                            pts = np.asarray([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])

                            box = [((x, y) + pt @ R).astype(int) for pt in pts]

                            pt1, pt2, pt3, pt4 = tuple(box[0]), tuple(box[1]), tuple(box[2]), tuple(box[3])

                            cur_target.append([pt1, pt2, pt3, pt4])

                        self.targets.append(cur_target)

                        f.close()

            else:

                data_dir = osp.join(data_path, root_data+'/Fisheye')

                for dir_ in os.listdir(data_dir):

                    dir_path = osp.join(data_dir, dir_)

                    for place_dir in os.listdir(dir_path):

                        place_dir_path = osp.join(dir_path, place_dir)

                        for img_path in glob.glob(osp.join(place_dir_path, '*.jpg')):

                                txt_path = img_path.replace('.jpg', '.txt')

                                self.imgs_path.append(img_path)

                                f = open(txt_path)

                                cur_target=[]

                                for line in f.readlines():

                                    _, x, y, w, h, a = line.strip().replace('\n', '').split(' ')

                                    x, y, w, h, a = int(x), int(y), int(w), int(h), int(a)

                                    a = a*np.pi/180

                                    C, S = np.cos(a), np.sin(a)

                                    R = np.asarray([[-C, -S], [S, -C]])

                                    pts = np.asarray([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])

                                    box = [((x, y) + pt @ R).astype(int) for pt in pts]

                                    pt1, pt2, pt3, pt4 = tuple(box[0]), tuple(box[1]), tuple(box[2]), tuple(box[3])

                                    cur_target.append([pt1, pt2, pt3, pt4])

                                self.targets.append(cur_target)

                                f.close()

                gc.collect()

                        

        if transform:

            self.transform = transform

        self.normalize = Normalize()



    def __len__(self):

        return len(self.imgs_path)



    def __getitem__(self, idx):

        if os.path.exists(self.imgs_path[idx]):

            img = cv2.imread(self.imgs_path[idx])

        else:

            print("%s not exists"%self.imgs_path)

        target = self.targets[idx]

        

        # color augment

        if np.random.rand() > 0.4:

            img = augment_hsv(img)

        if np.random.rand() > 0.6:

            img = motion_blur(img)

        # make image and heatmap

        img, hm, regr, regr_mask = make_hm_regr(img, target)

        # normalize image

        img = self.normalize(img)

        img = img.transpose([2,0,1])

        return img, hm, regr, regr_mask
traindataset = FisheyeDataset()

# Pack to dataloaders

train_loader = torch.utils.data.DataLoader(traindataset,batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

# val_loader = torch.utils.data.DataLoader(valdataset,batch_size=batch_size,shuffle=True, num_workers=0)

# test_loader = torch.utils.data.DataLoader(testdataset,batch_size=batch_size,shuffle=False, num_workers=0)
# data = next(iter(train_loader))
# batch = 6

# img, hm = data[0][batch], data[1][batch]

# img.shape, hm.shape

# img = img.permute(1,2,0)

# fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# plt.imshow(img)

# plt.show()

# plt.imshow(hm)
class double_conv(nn.Module):

    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):

        super(double_conv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True)

        )



    def forward(self, x):

        x = self.conv(x)

        return x



class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):

        super(up, self).__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:

            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

        

    def forward(self, x1, x2=None):

        x1 = self.up(x1)

        if x2 is not None:

            x = torch.cat([x2, x1], dim=1)

            # input is CHW

            diffY = x2.size()[2] - x1.size()[2]

            diffX = x2.size()[3] - x1.size()[3]



            x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,

                            diffY // 2, diffY - diffY//2))

        else:

            x = x1

        x = self.conv(x)

        return x



class centernet(nn.Module):

    def __init__(self, n_classes=1, model_name="resnet18", training_mode=False):

        super(centernet, self).__init__()

        # create backbone.

        basemodel = torchvision.models.resnet18(pretrained=training_mode) # turn this on for training

        basemodel = nn.Sequential(*list(basemodel.children())[:-2])

        # set basemodel

        self.base_model = basemodel

        

        if model_name == "resnet34" or model_name=="resnet18":

            num_ch = 512

        else:

            num_ch = 2048

        

        self.up1 = up(num_ch, 512)

        self.up2 = up(512, 256)

        self.up3 = up(256, 256)

        # output classification

        self.outc = nn.Conv2d(256, n_classes, 1)

        # output residue

        self.outr = nn.Conv2d(256, 4, 1)

        

    def forward(self, x):

        batch_size = x.shape[0]

        x = self.base_model(x)

        # Add positional info        

        x = self.up1(x)

        x = self.up2(x)

        x = self.up3(x)

        outc = self.outc(x)

        outr = self.outr(x)

        return outc, outr
model = centernet(training_mode=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
# From centernet repo

def neg_loss(pred, gt):

    ''' Modified focal loss. Exactly the same as CornerNet.

      Runs faster and costs a little bit more memory

    Arguments:

      pred (batch x c x h x w)

      gt_regr (batch x c x h x w)

    '''

    pred = pred.unsqueeze(1).float()

    gt = gt.unsqueeze(1).float()



    pos_inds = gt.eq(1).float()

    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)



    loss = 0



    pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 3) * pos_inds

    neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, 3) * neg_weights * neg_inds



    num_pos  = pos_inds.float().sum()

    pos_loss = pos_loss.sum()

    neg_loss = neg_loss.sum()



    if num_pos == 0:

        loss = loss - neg_loss

    else:

        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss



def _reg_loss(regr, gt_regr, mask):

    ''' L1 regression loss

    Arguments:

      regr (batch x max_objects x dim)

      gt_regr (batch x max_objects x dim)

      mask (batch x max_objects)

    '''

    num = mask.float().sum()

    #print(gt_regr.size())

    mask = mask.sum(1).unsqueeze(1).expand_as(gt_regr)

    #print(mask.size())



    regr = regr * mask

    gt_regr = gt_regr * mask



    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)

    regr_loss = regr_loss / (num + 1e-4)

    return regr_loss

  

def centerloss(prediction, mask, regr, regr_mask, weight=0.4, size_average=True):

    # Binary mask loss

    pred_mask = torch.sigmoid(prediction[:, 0])

    mask_loss = neg_loss(pred_mask, mask)

    

    # Regression L1 loss

    pred_regr = prediction[:, 1:]

    regr_mask = regr_mask.expand_as(pred_regr).float()

    regr_loss = F.l1_loss(pred_regr*regr_mask, regr*regr_mask, size_average=False)

    regr_loss = regr_loss / (regr_mask.sum() + 1e-4)

#     regr_loss = (torch.abs(pred_regr*regr_mask - regr*regr_mask).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)

#     regr_loss = regr_loss.mean(0)

  

    # Sum

    loss = mask_loss + regr_loss * 0.1

    if not size_average:

        loss *= prediction.shape[0]

    return loss ,mask_loss , regr_loss
def train(epoch):

    model.train()

    print('epochs {}/{} '.format(epoch+1,epochs))

    running_loss = 0.0

    running_mask = 0.0

    running_regr = 0.0

    t = tqdm(train_loader)

    rd = np.random.rand()

    

    for idx, (img, hm, regr, regr_mask) in enumerate(t):       

        # send to gpu

        img = img.to(device)

        hm_gt = hm.to(device)

        regr_gt = regr.to(device)

        regr_mask = regr_mask.to(device)

        # set opt

        optimizer.zero_grad()

        

        # run model

        hm, regr = model(img)

        preds = torch.cat((hm, regr), 1)

            

        loss, mask_loss, regr_loss = centerloss(preds, hm_gt, regr_gt, regr_mask)

        # misc

        running_loss += loss

        running_mask += mask_loss

        running_regr += regr_loss

        

        loss.backward()

        optimizer.step()

        

        t.set_description(f't (l={running_loss/(idx+1):.3f})(m={running_mask/(idx+1):.4f})(r={running_regr/(idx+1):.4f})')

        

    #scheduler.step()

    print('train loss : {:.4f}'.format(running_loss/len(train_loader)))

    print('maskloss : {:.4f}'.format(running_mask/(len(train_loader))))

    print('regrloss : {:.4f}'.format(running_regr/(len(train_loader))))

    

    # save logs

    log_epoch = {'epoch': epoch+1, 'lr': optimizer.state_dict()['param_groups'][0]['lr'],

                    'loss': running_loss/len(train_loader), "mask": running_mask/(len(train_loader)), 

                 "regr": running_regr/(len(train_loader))}

    logs.append(log_epoch)

    torch.save(model.state_dict(), "checkpoint.pt")
# Optimizer

import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler

optimizer = optim.Adam(model.parameters(), lr=1e-4)
resume = 0

epochs = 5

logs = []

if MODEL_PATH is not None:

    print('Load model')

    model.load_state_dict(torch.load(MODEL_PATH))

#     lf = lambda x: (((1 + math.cos(x * math.pi / 10)) / 2) ** 1.0) * 0.95 + 0.05  # cosine

#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

for epoch in range(epochs):

    if epoch < resume:

        continue

    train(epoch)

#             scheduler.step()
def preprocess(img, input_size, mean, std):

    img = cv2.resize(img, (input_size, input_size))

    img_tr = img/255

    img_tr = (img_tr - mean)/std

    img_tr = img_tr.transpose([2,0,1])

    

    return img_tr
thresh = 0.3

mean=[0.5, 0.5, 0.5]

std=[0.5, 0.5, 0.5]

img = cv2.imread('../input/test-data/exhibition.jpg')

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (input_size, input_size))

img_tensor = preprocess(img, input_size, mean, std)



model.load_state_dict(torch.load(MODEL_PATH))

model.eval()



t0 = timeit.default_timer()

with torch.no_grad():

    outs = model(torch.from_numpy(img_tensor).to(device).float().unsqueeze(0))

    hm, regr = outs[0], outs[1]

t1 = timeit.default_timer()

print(t1-t0)

hm = hm.cpu().numpy().squeeze(0).squeeze(0)

regr = regr.cpu().numpy().squeeze(0)

# get boxes

hm = torch.sigmoid(torch.from_numpy(hm)).numpy()

hm = pool(hm)

plt.imshow(hm>thresh)

plt.show()

sample = showbox(img, hm, regr, thresh)

# show gt

# sample = showgtbox(img, hm_gt, regr_gt, thresh)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

plt.imshow(sample)

plt.show()