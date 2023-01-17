import numpy as np
import cv2
import math
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
def do_resize(image, H, W):
    image = cv2.resize(image, dsize=(W, H))

    return image

def do_resize2(image, mask, H, W):
    image = cv2.resize(image, dsize=(W, H))
    mask = cv2.resize(mask, dsize=(W, H))
    mask = (mask > 0.5).astype(np.float32)

    return image, mask


#################################################################

def compute_center_pad(H, W, factor=32):
    if H % factor == 0:
        dy0, dy1 = 0, 0
    else:
        dy = factor - H % factor
        dy0 = dy // 2
        dy1 = dy - dy0

    if W % factor == 0:
        dx0, dx1 = 0, 0
    else:
        dx = factor - W % factor
        dx0 = dx // 2
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1


def do_center_pad_to_factor(image, factor=32):
    H, W = image.shape[:2]
    dy0, dy1, dx0, dx1 = compute_center_pad(H, W, factor)

    image = cv2.copyMakeBorder(image, dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
    # cv2.BORDER_CONSTANT, 0)
    return image


def do_center_pad_to_factor2(image, mask, factor=32):
    image = do_center_pad_to_factor(image, factor)
    mask = do_center_pad_to_factor(mask, factor)
    return image, mask


# ---

def do_horizontal_flip(image):
    # flip left-right
    image = cv2.flip(image, 1)
    return image


def do_horizontal_flip2(image, mask):
    image = do_horizontal_flip(image)
    mask = do_horizontal_flip(mask)
    return image, mask


# ---

def compute_random_pad(H, W, limit=(-4, 4), factor=32):
    if H % factor == 0:
        dy0, dy1 = 0, 0
    else:
        dy = factor - H % factor
        dy0 = dy // 2 + np.random.randint(limit[0], limit[1])  # np.random.choice(dy)
        dy1 = dy - dy0

    if W % factor == 0:
        dx0, dx1 = 0, 0
    else:
        dx = factor - W % factor
        dx0 = dx // 2 + np.random.randint(limit[0], limit[1])  # np.random.choice(dx)
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1


def do_random_pad_to_factor2(image, mask, limit=(-4, 4), factor=32):
    H, W = image.shape[:2]
    dy0, dy1, dx0, dx1 = compute_random_pad(H, W, limit, factor)

    image = cv2.copyMakeBorder(image, dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)

    return image, mask


# ----
def do_invert_intensity(image):
    # flip left-right
    image = np.clip(1 - image, 0, 1)
    return image


def do_brightness_shift(image, alpha=0.125):
    image = image + alpha
    image = np.clip(image, 0, 1)
    return image


def do_brightness_multiply(image, alpha=1):
    image = alpha * image
    image = np.clip(image, 0, 1)
    return image


# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):
    image = image ** (1.0 / gamma)
    image = np.clip(image, 0, 1)
    return image


def do_flip_transpose2(image, mask, type=0):
    # choose one of the 8 cases

    if type == 1:  # rotate90
        image = image.transpose(1, 0)
        image = cv2.flip(image, 1)

        mask = mask.transpose(1, 0)
        mask = cv2.flip(mask, 1)

    if type == 2:  # rotate180
        image = cv2.flip(image, -1)
        mask = cv2.flip(mask, -1)

    if type == 3:  # rotate270
        image = image.transpose(1, 0)
        image = cv2.flip(image, 0)

        mask = mask.transpose(1, 0)
        mask = cv2.flip(mask, 0)

    if type == 4:  # flip left-right
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if type == 5:  # flip up-down
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    if type == 6:
        image = cv2.flip(image, 1)
        image = image.transpose(1, 0)
        image = cv2.flip(image, 1)

        mask = cv2.flip(mask, 1)
        mask = mask.transpose(1, 0)
        mask = cv2.flip(mask, 1)

    if type == 7:
        image = cv2.flip(image, 0)
        image = image.transpose(1, 0)
        image = cv2.flip(image, 1)

        mask = cv2.flip(mask, 0)
        mask = mask.transpose(1, 0)
        mask = cv2.flip(mask, 1)

    return image, mask

##================================
def do_shift_scale_crop(image, mask, x0=0, y0=0, x1=1, y1=1):
    # cv2.BORDER_REFLECT_101
    # cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    image = image[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]

    image = cv2.resize(image, dsize=(width, height))
    mask = cv2.resize(mask, dsize=(width, height))
    mask = (mask > 0.5).astype(np.float32)
    return image, mask


def do_random_shift_scale_crop_pad2(image, mask, limit=0.10):
    H, W = image.shape[:2]

    dy = int(H * limit)
    y0 = np.random.randint(0, dy)
    y1 = H - np.random.randint(0, dy)

    dx = int(W * limit)
    x0 = np.random.randint(0, dx)
    x1 = W - np.random.randint(0, dx)

    # y0, y1, x0, x1
    image, mask = do_shift_scale_crop(image, mask, x0, y0, x1, y1)
    return image, mask


# ===========================================================================

def do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=0):
    borderMode = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    sx = scale
    sy = scale
    cc = math.cos(angle / 180 * math.pi) * (sx)
    ss = math.sin(angle / 180 * math.pi) * (sy)
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=borderMode, borderValue=(
        0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,  # cv2.INTER_LINEAR
                               borderMode=borderMode, borderValue=(
        0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = (mask > 0.5).astype(np.float32)
    return image, mask


# https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations
# https://github.com/letmaik/lensfunpy/blob/master/lensfunpy/util.py
def do_elastic_transform2(image, mask, grid=32, distort=0.2):
    borderMode = cv2.BORDER_REFLECT_101
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width, np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * (1 + np.random.uniform(-distort, distort))

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = int(grid)
    yy = np.zeros(height, np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * (1 + np.random.uniform(-distort, distort))

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    # grid
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # image = map_coordinates(image, coords, order=1, mode='reflect').reshape(shape)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=borderMode,
                      borderValue=(0, 0, 0,))

    mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=borderMode, borderValue=(0, 0, 0,))
    mask = (mask > 0.5).astype(np.float32)
    return image, mask


def do_horizontal_shear2(image, mask, dx=0):
    borderMode = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    dx = int(dx * width)

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = np.array([[+dx, 0], [width + dx, 0], [width - dx, height], [-dx, height], ], np.float32)

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=borderMode, borderValue=(
        0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,  # cv2.INTER_LINEAR
                               borderMode=borderMode, borderValue=(
        0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = (mask > 0.5).astype(np.float32)
    return image, mask

def normalize(im):
    max = np.max(im)
    min = np.min(im)
    if (max - min) > 0:
        im = (im - min) / (max - min)
    return im

def basic_augment(image, mask):
    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)
        pass

    if np.random.rand() < 0.5:
        c = np.random.choice(4)
        if c == 0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)  # 0.125

        if c == 1:
            image, mask = do_horizontal_shear2(image, mask, dx=np.random.uniform(-0.07, 0.07))
            pass

        if c == 2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 15))  # 10

        if c == 3:
            image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0, 0.15))  # 0.10
            pass

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c == 0:
            image = do_brightness_shift(image, np.random.uniform(-0.1, +0.1))
        if c == 1:
            image = do_brightness_multiply(image, np.random.uniform(1 - 0.08, 1 + 0.08))
        if c == 2:
            image = do_gamma(image, np.random.uniform(1 - 0.08, 1 + 0.08))
        # if c==1:
        #     image = do_invert_intensity(image)

    return image, mask

class Airbus_Dataset():

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.segmentation_df, self.train_imgs, self.test_imgs = self.create_dataset_df(self.folder_path)

    @staticmethod
    def create_dataset_df(folder_path):
        '''Create a dataset for a specific dataset folder path'''

        train_path = folder_path + '/train_v2/'
        test_path = folder_path + '/test_v2/'
        segmentation_csv = folder_path + '/train_ship_segmentations_v2.csv'

        # 5% of data in the validation set is sufficient for model evaluation
        segmentation_df = pd.read_csv(segmentation_csv).set_index('ImageId')

        train_imgs = [os.path.join(train_path, id) for id in os.listdir(train_path)]
        test_imgs = [os.path.join(test_path, id) for id in os.listdir(test_path)]

        return segmentation_df, train_imgs, test_imgs
    def cut_empty(self, names):
        return [name for name in names
            if (type(self.segmentation_df.loc[name.strip().split('/')[-1]]['EncodedPixels']) != float)]

    def yield_dataloader(self, data='train', shuffle=True, seed=1234, num_workers=8, batch_size=32, size=672):

        if data == 'train':

            tr_n, val_n = train_test_split(self.train_imgs, test_size=0.05, random_state=seed)

            tr_n = self.cut_empty(tr_n)
            val_n = self.cut_empty(val_n)
            
#             tr_n = tr_n[:20]
#             val_n = val_n[:20]
#             print(len(tr_n), len(val_n))
            
            train_dataset = TorchDataset(img_ids=tr_n, df=self.segmentation_df, transform=basic_augment, size=size)
            train_loader = DataLoader(train_dataset,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      batch_size=batch_size,
                                      pin_memory=True)

            val_dataset = TorchDataset(img_ids=val_n, df=self.segmentation_df, size=size)
            val_loader = DataLoader(val_dataset,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    batch_size=batch_size,
                                    pin_memory=True)
            return train_loader, val_loader

        elif data == 'test':
            test_dataset = TorchDataset(img_ids=self.test_imgs, is_test=True, size=size)
            test_loader = DataLoader(test_dataset,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     batch_size=batch_size,
                                     pin_memory=True)
            return test_loader

class TorchDataset(Dataset):

    def __init__(self, img_ids, size=672, df = None, is_test=False, transform=None):
        self.img_ids = img_ids
        self.df = df
        self.is_test = is_test
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.img_ids)

    def get_mask(self, img_id, df):
        shape = (768, 768)
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        masks = df.loc[img_id]['EncodedPixels']
        if (type(masks) == float): return img.reshape(shape)
        if (type(masks) == str): masks = [masks]
        for mask in masks:
            s = mask.split()
            for i in range(len(s) // 2):
                start = int(s[2 * i]) - 1
                length = int(s[2 * i + 1])
                img[start:start + length] = 1
        return img.reshape(shape).T

    def load_images(self, img_id):
        im = normalize(cv2.imread(img_id, cv2.IMREAD_COLOR).astype(np.float32))
        if not self.is_test:
            ImageId = img_id.strip().split('/')[-1]
            mask = self.get_mask(ImageId, self.df)
            return im, mask, ImageId
        else:
            ImageId = img_id.strip().split('/')[-1]
            return im, ImageId

    def __getitem__(self, index):

        img_id = self.img_ids[index]

        if not self.is_test:
            im, mask, z= self.load_images(img_id)
            if self.transform is not None:
                im, mask = self.transform(im, mask)

            im = cv2.resize(im, (self.size, self.size))
            mask = np.array(cv2.resize(mask, (self.size ,self.size))>0.5).astype('float32')

            mask = np.expand_dims(mask, 0)
            mask = torch.from_numpy(mask).float()
            im = im.transpose((2,0,1))
            im = torch.from_numpy(im).float()

            return im, mask, z
        else:
            im, z = self.load_images(img_id)

            im = cv2.resize(im, (self.size, self.size))

            im = im.transpose((2,0,1))
            im = torch.from_numpy(im).float()
            return im, z
def show_image_mask(im, mask, n=2, label='Image', show=True, cmap='jet', format='channels_last'):
    im = np.squeeze(im)
    mask = np.squeeze(mask)
    if format == 'channels_first':
        im = im.transpose((0, 2, 3, 1))

    n_batch = im.shape[0]
    # idx = np.random.choice(np.arange(n_batch), n, replace=False)
    idx = range(n_batch)
    fig, axs = plt.subplots(2, n)
    for i in range(n):
        axs[0, i].imshow(im[idx[i], :, :, :], cmap=cmap)
        axs[0, i].set_title('{}: {}'.format(label, i))
        axs[1, i].imshow(mask[idx[i], :, :], cmap=cmap)
        axs[1, i].set_title('mask: {}'.format(i))
    if show:
        plt.show()

def show_test_images(im, n=2, label='Image', show=True, cmap='jet', format='channels_first'):
    im = np.squeeze(im)
    if format == 'channels_first':
        im = im.transpose((0, 2, 3, 1))
    n_batch = im.shape[0]
    # idx = np.random.choice(np.arange(n_batch), n, replace=False)
    idx = range(n_batch)
    fig, axs = plt.subplots(1, n)
    for i in range(n):
        axs[i].imshow(im[idx[i], :, :, :], cmap=cmap)
        axs[i].set_title('{}: {}'.format(label, i))
    if show:
        plt.show()

FOLDER_PATH = '../input'

dataset = Airbus_Dataset(FOLDER_PATH)
train_loader, val_loader = dataset.yield_dataloader(data='train', shuffle=True, seed=1234, num_workers=1, batch_size=2)
test_loader = dataset.yield_dataloader(data='test', shuffle=False, seed=42, num_workers=1, batch_size=2)

for i, (img, mask, z) in enumerate(train_loader):
    img = img.numpy().transpose((0, 2, 3, 1))
    mask = mask.numpy().transpose((0, 2, 3, 1))
    reimg = np.array([cv2.resize(i, (224,224)) for i in img])
    remask = np.array([cv2.resize(i, (224,224)) > 0.5 for i in mask]).astype('float32')
    show_image_mask(img, mask)
    show_image_mask(reimg, remask)
    if i >4:
        break
for i, (img, mask, z) in enumerate(val_loader):
    img = img.numpy().transpose((0, 2, 3, 1))
    mask = mask.numpy().transpose((0, 2, 3, 1))
    reimg = np.array([cv2.resize(i, (224,224)) for i in img])
    remask = np.array([cv2.resize(i, (224,224)) > 0.5 for i in mask]).astype('float32')
    show_image_mask(img, mask)
    show_image_mask(reimg, remask)
    if i >4:
        break
for i, (img, z) in enumerate(test_loader):
    ti = img.numpy()
    show_test_images(ti)
    if i >4:
        break
import matplotlib.pyplot as plt
import numpy as np

def show_image_mask(im, mask, n=3, label='Image', show=True, cmap='jet', format='channels_first'):
    im = np.squeeze(im)
    mask = np.squeeze(mask)
    if format == 'channels_first':
        im = im.transpose((0, 2, 3, 1))

    n_batch = im.shape[0]
    idx = np.random.choice(np.arange(n_batch), n, replace=False)
    fig, axs = plt.subplots(2, n)
    for i in range(n):
        axs[0, i].imshow(im[idx[i], :, :, :], cmap=cmap)
        axs[0, i].set_title('{}: {}'.format(label, i))
        axs[1, i].imshow(mask[idx[i], :, :], cmap=cmap)
        axs[1, i].set_title('mask: {}'.format(i))
    if show:
        plt.show()

def show_image_mask_pred(im, mask, logit,  n=3, label='Image', show=True, cmap='gray', format='channels_first'):
    mask = np.squeeze(mask)
    logit = np.squeeze(logit)

    if format == 'channels_first':
        if im.shape[1] == 1:
            im = np.squeeze(im)
            im_cmap = 'gray'
        else:
            im = im.transpose((0, 2, 3, 1))
            im_cmap = None

    n_batch = im.shape[0]
    idx = np.random.choice(np.arange(n_batch), n, replace=False)
    fig, axs = plt.subplots(3, n)
    for i in range(n):
        axs[0, i].imshow(im[idx[i]], cmap=im_cmap)
        axs[0, i].set_title('{}: {}'.format(label, i))
        axs[1, i].imshow(mask[idx[i]], cmap=cmap)
        axs[1, i].set_title('mask: {}'.format(i))
        axs[2, i].imshow(logit[idx[i]], cmap=cmap)
        axs[2, i].set_title('logit: {}'.format(i))
    if show:
        plt.show

def show_image_tta_pred(im, tta_im, logit, tta_logit,  n=3, label='Image', show=True, cmap='gray', format='channels_first'):
    logit = np.squeeze(logit)
    tta_logit = np.squeeze(tta_logit)

    if format == 'channels_first':
        if im.shape[1] == 1:
            im = np.squeeze(im)
            tta_im = np.squeeze(tta_im)
            im_cmap = 'gray'
        else:
            im = im.transpose((0, 2, 3, 1))
            tta_im = tta_im.transpose((0, 2, 3, 1))
            im_cmap = None

    n_batch = im.shape[0]
    idx = np.random.choice(np.arange(n_batch), n, replace=False)
    fig, axs = plt.subplots(n, 4)
    for i in range(n):
        axs[i, 0].imshow(im[idx[i]], cmap=im_cmap)
        axs[i, 0].set_title('{}: {}'.format(label, i))
        axs[i, 1].imshow(logit[idx[i]], cmap=cmap)
        axs[i, 1].set_title('logit: {}'.format(i))
        axs[i, 2].imshow(tta_im[idx[i]], cmap=im_cmap)
        axs[i, 2].set_title('TTA {}: {}'.format(label, i))
        axs[i, 3].imshow(tta_logit[idx[i]], cmap=cmap)
        axs[i, 3].set_title('TTA logit: {}'.format(i))
    if show:
        plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), groups=1, dilation=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              groups=groups,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SpatialGate2d(nn.Module):

    def __init__(self, in_channels):
        super(SpatialGate2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cal = self.conv1(x)
        cal = self.sigmoid(cal)
        return cal * x

class ChannelGate2d(nn.Module):

    def __init__(self, channels, reduction=2):
        super(ChannelGate2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

class scSqueezeExcitationGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super(scSqueezeExcitationGate, self).__init__()
        self.spatial_gate = SpatialGate2d(channels)
        self.channel_gate = ChannelGate2d(channels, reduction=reduction)

    def  forward(self, x, z=None):
        XsSE = self.spatial_gate(x)
        XcSe = self.channel_gate(x)
        return XsSE + XcSe
    
class Decoder_v3(nn.Module):
    def __init__(self, in_channels, convT_channels, out_channels, convT_ratio=2, SE=False):
        super(Decoder_v3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.SE = SE
        self.convT = nn.ConvTranspose2d(convT_channels, convT_channels // convT_ratio, kernel_size=2, stride=2)
        self.conv1 = ConvBn2d(in_channels  + convT_channels // convT_ratio, out_channels)
        self.conv2 = ConvBn2d(out_channels, out_channels)
        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)

        self.conv_res = nn.Conv2d(convT_channels // convT_ratio, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip):
        x = self.convT(x)
        residual = x
        x = torch.cat([x, skip], 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.SE:
            x = self.scSE(x)
        x += self.conv_res(residual)
        x = self.relu(x)
        return x
class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, SE=False):
        super(CenterBlock, self).__init__()
        self.SE = SE
        self.pool = pool
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvBn2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if SE:
            self.se = scSqueezeExcitationGate(out_channels)

    def forward(self, x):
        if self.pool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        residual = self.conv_res(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.SE:
            x = self.se(x)

        x += residual
        x = self.relu(x)
        return x
class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Conv2dBn(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.conv(x)
class FPAModule1(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(FPAModule1, self).__init__()

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        # midddle branch
        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=7, stride=1, padding=3)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            Conv2dBnRelu(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = Conv2dBnRelu(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(out_ch, out_ch, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b1)

        mid = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)

        x1 = self.conv1(x1)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)

        x = torch.mul(x, mid)
        x = x + b1
        return x
import torch.optim as optim
import torch.nn as nn
import torch

from contextlib import contextmanager
import datetime
import  time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.3f}s".format(title, time.time() - t0))


class SegmentationNetwork(nn.Module):

    def __init__(self, lr=0.005, fold=None, debug=False, val_mode='max', comment=''):
        super(SegmentationNetwork, self).__init__()
        self.lr = lr
        self.fold = fold
        self.debug = debug
        self.scheduler = None
        self.best_model_path = None
        self.epoch = 0
        self.val_mode = val_mode
        self.comment = comment

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf

        self.train_log = dict(loss=[], iou=[], mAP=[])
        self.val_log = dict(loss=[], iou=[], mAP=[])
        self.create_save_folder()

    def create_optmizer(self, optimizer='SGD', use_scheduler=None, gamma=0.25, patience=4,
                        milestones=None, T_max=10, T_mul=2, lr_min=0):
        self.cuda()
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                  lr=self.lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                       self.parameters()), lr=self.lr)

        if use_scheduler == 'ReduceOnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  mode='max',
                                                                  factor=gamma,
                                                                  patience=patience,
                                                                  verbose=True,
                                                                  threshold=0.01,
                                                                  min_lr=1e-05,
                                                                  eps=1e-08)

        elif use_scheduler == 'Milestones':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=milestones,
                                                            gamma=gamma,
                                                            last_epoch=-1)

        elif use_scheduler == 'CosineAnneling':
            self.scheduler = CosineAnnealingLR(self.optimizer,
                                                         T_max=T_max,
                                                         T_mul=T_mul,
                                                         lr_min=lr_min,
                                                         val_mode=self.val_mode,
                                                         last_epoch=-1,
                                                         save_snapshots=True)


    def train_network(self, train_loader, val_loader, n_epoch=10):
        print('Model created, total of {} parameters'.format(
            sum(p.numel() for p in self.parameters())))
        while self.epoch < n_epoch:
            self.epoch += 1
            lr = np.mean([param_group['lr'] for param_group in self.optimizer.param_groups])
            with timer('Train Epoch {:}/{:} - LR: {:.3E}'.format(self.epoch, n_epoch, lr)):
                # Training step
                train_loss, train_iou, train_mAP = self.training_step(train_loader)
                #  Validation
                val_loss, val_iou, val_mAP = self.perform_validation(val_loader)
                # Learning Rate Scheduler
                if self.scheduler is not None:
                    if type(self.scheduler).__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(np.mean(val_mAP))
                    elif type(self.scheduler).__name__ == 'CosineAnnealingLR':
                        self.scheduler.step(self.epoch,
                                            save_dict=dict(metric=np.mean(val_loss),
                                                           save_dir=self.save_dir,
                                                           fold=self.fold,
                                                           state_dict=self.state_dict()))
                    else:
                        self.scheduler.step(self.epoch)
                # Save best model
                if type(self.scheduler).__name__ != 'CosineAnnealingLR':
                    self.save_best_model(np.mean(val_mAP))

            # Print statistics
            print(('train loss: {:.3f}  val_loss: {:.3f}  '
                   'train iou:  {:.3f}  val_iou:  {:.3f}  '
                   'train mAP:  {:.3f}  val_mAP:  {:.3f}').format(
                np.mean(train_loss),
                np.mean(val_loss),
                np.mean(train_iou),
                np.mean(val_iou),
                np.mean(train_mAP),
                np.mean(val_mAP)))

        self.save_training_log()

    def training_step(self, train_loader):
        self.set_mode('train')
        train_loss = []
        train_iou = []
        train_mAP = []
        for i, (im, mask, z) in enumerate(train_loader):
            self.optimizer.zero_grad()
            im = im.cuda()
            mask = mask.cuda()
            logit = self.forward(im)
            pred = torch.sigmoid(logit)

            loss = self.criterion(logit, mask)
            iou  = dice_accuracy(pred, mask, is_average=False)
            mAP = do_mAP(pred.data.cpu().numpy(), mask.cpu().numpy())

            train_loss.append(loss.item())
            train_iou.extend(iou)
            train_mAP.extend(mAP)

            loss.backward()
            self.optimizer.step()

            if self.debug and not self.epoch % 5 and not i % 30:
                show_image_mask_pred(
                    im.cpu().data.numpy(), mask.cpu().data.numpy(), logit.cpu().data.numpy())
        # Append epoch data to metrics dict
        for metric in ['loss', 'iou', 'mAP']:
            self.train_log[metric].append(np.mean(eval('train_{}'.format(metric))))
        return train_loss, train_iou, train_mAP


    def perform_validation(self, val_loader):
        self.set_mode('valid')
        val_loss = []
        val_iou = []
        val_mAP = []
        for im, mask, z in val_loader:
            im = im.cuda()
            mask = mask.cuda()

            with torch.no_grad():
                logit = self.forward(im)
                pred = torch.sigmoid(logit)
                loss = self.criterion(logit, mask)
                iou  = dice_accuracy(pred, mask, is_average=False)
                mAP = do_mAP(pred.cpu().numpy(), mask.cpu().numpy())

            val_loss.append(loss.item())
            val_iou.extend(iou)
            val_mAP.extend(mAP)
        # Append epoch data to metrics dict
        for metric in ['loss', 'iou', 'mAP']:
            self.val_log[metric].append(np.mean(eval('val_{}'.format(metric))))

        return val_loss, val_iou, val_mAP


    def predict(self, test_loader, return_rle=False, tta_transform=None, threshold=0.45):
        self.set_mode('test')
        self.cuda()
        for i, (im, z) in enumerate(test_loader):
            with torch.no_grad():
                # Apply TTA and predict
                batch_pred = []
                # TTA
                if tta_transform is not None:
                    tta_list = torch.FloatTensor(tta_transform(im.cpu().numpy(), mode='in'))
                    tta_pred = []
                    for t_im in tta_list:
                        t_im = t_im.cuda()
                        t_logit = self.forward(t_im)
                        pred = torch.sigmoid(t_logit)
                        pred = unpad_im(pred.cpu().numpy())
                        tta_pred.append(pred)
                    batch_pred.extend(tta_transform(tta_pred, mode='out'))

                # Predict original batch
                im = im.cuda()
                logit = self.forward(im)
                pred = torch.sigmoid(logit)
                pred = unpad_im(pred.cpu().numpy())
                batch_pred.append(pred)

                # Average TTA results
                batch_pred = np.mean(batch_pred, 0)
                # Threshold result
                if threshold > 0:
                    batch_pred = batch_pred > threshold

                if return_rle:
                    batch_pred = batch_encode(batch_pred)

                if not i:
                    out = batch_pred
                    ids = z
                else:
                    out = np.concatenate([out, batch_pred], axis=0)
                    ids = np.concatenate([ids, z], axis=0)

                if self.debug:
                    show_image_tta_pred(
                        im.cpu().data.numpy(), t_im.cpu().data.numpy(),
                        logit.cpu().data.numpy(), t_logit.cpu().data.numpy())

        if return_rle:
            out = dict(id=ids, rle_mask=out)
            out = pd.DataFrame(out)
        else:
            out = dict(id=ids, pred=out)
        return out


    def define_criterion(self, name):
        if name.lower() == 'bce+dice':
            self.criterion = BCE_Dice()
        elif name.lower() == 'dice':
            self.criterion = DiceLoss()
        elif name.lower() == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif name.lower() == 'robustfocal':
            self.criterion = RobustFocalLoss2d()
        elif name.lower() == 'lovasz-hinge' or name.lower() == 'lovasz':
            self.criterion = Lovasz_Hinge(per_image=True)
        elif name.lower() == 'bce+lovasz':
            self.criterion = BCE_Lovasz(per_image=True)
        else:
            raise NotImplementedError('Loss {} is not implemented'.format(name))


    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


    def save_best_model(self, metric):
        if (self.val_mode == 'max' and metric > self.best_metric) or (self.val_mode == 'min' and metric < self.best_metric):
            # Update best metric
            self.best_metric = metric
            # Remove old file
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            # Save new best model weights
            date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
#             if self.fold is not None:
#                 self.best_model_path = os.path.join(
#                     self.save_dir,
#                     '{:}_Fold{:}_Epoach{}_val{:.3f}'.format(date, self.fold, self.epoch, metric))
#             else:
#                 self.best_model_path = os.path.join(
#                     self.save_dir,
#                     '{:}_Epoach{}_val{:.3f}'.format(date, self.epoch, metric))
            if self.fold is not None:
                self.best_model_path = os.path.join(
                    self.save_dir,
                    'Fold{:}_Epoach{}_val{:.3f}'.format(self.fold, self.epoch, metric))
            else:
                self.best_model_path = os.path.join(
                    self.save_dir,
                    'Epoach{}_val{:.3f}'.format(self.epoch, metric))

            torch.save(self.state_dict(), self.best_model_path)


    def save_training_log(self):
        d = dict()
        for tk, vk in zip(self.train_log.keys(), self.val_log.keys()):
            d['train_{}'.format(tk)] = self.train_log[tk]
            d['val_{}'.format(vk)] = self.val_log[vk]

        df = pd.DataFrame(d)
        df.index += 1
        df.index.name = 'Epoach'

        date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
#         if self.fold is not None:
#             p = os.path.join(
#                 self.save_dir,
#                 '{:}_Fold{:}_TrainLog.csv'.format(date, self.fold))
#         else:
#             p = os.path.join(
#                 self.save_dir,
#                 '{:}_TrainLog.csv'.format(date))
        if self.fold is not None:
            p = os.path.join(
                self.save_dir,
                'Fold{:}_TrainLog.csv'.format(self.fold))
        else:
            p = os.path.join(
                self.save_dir,
                'TrainLog.csv')

        df.to_csv(p, sep=";")

        with open(p, 'a') as fd:
            fd.write(self.comment)


    def load_model(self, path=None, best_model=False):
        if best_model:
            self.load_state_dict(torch.load(self.best_model_path))
        else:
            self.load_state_dict(torch.load(path))

    def create_save_folder(self):
        name = type(self).__name__
        #self.save_dir = os.path.join('./Saves', name)
        self.save_dir = './'
        #if not os.path.exists(self.save_dir):
        #    os.makedirs(self.save_dir)

    def plot_training_curve(self, show=True):
        fig, axs = plt.subplots(3, 1)
        for i, metric in enumerate(['loss', 'iou', 'mAP']):
            axs[i].plot(self.train_log[metric], 'y', label='Train')
            axs[i].plot(self.val_log[metric], 'b', label='Validation')
            if metric == 'loss':
                min = np.argmin(self.val_log[metric])
                axs[i].plot(min, self.val_log[metric][min], "xr", label='best_loss')
            else:
                max = np.argmax(self.val_log[metric])
                axs[i].plot(max, self.val_log[metric][max], "xr", label='best_{}'.format(metric))
            axs[i].legend()
            axs[i].set_title(metric)
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel(metric)
        if show:
            plt.show()
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelGate2d(nn.Module):

    def __init__(self, channels, reduction=2):
        super(ChannelGate2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation=None, SE=False):
        super(BasicBlock, self).__init__()
        self.SE = SE
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        if SE:
            self.cSE = ChannelGate2d(planes, reduction=16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.SE:
            out = self.cSE(out)

        out += residual
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation=None, SE=False):
        super(Bottleneck, self).__init__()
        self.SE = SE
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.stride = stride
        if SE:
            self.cSE = ChannelGate2d(planes, reduction=16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.SE:
            out = self.cSE(out)

        out += residual
        out = self.activation(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, activation=None, num_classes=1000, SE=False):
        super(ResNet, self).__init__()

        self.SE = SE
        self.inplanes = 64
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, activation=self.activation, SE=self.SE))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=self.activation, SE=self.SE))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def load_pretrain_file(net, pretrain_file, skip=['cSE']):
    pretrain_state_dict = torch.load(pretrain_file)
    state_dict = net.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        if any(s in key for s in skip):
            continue
        else:
            state_dict[key] = pretrain_state_dict[key]
class UNetResNet34_SE_Hyper_FPA(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34_SE_Hyper_FPA, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = resnet34(pretrained=pretrained, activation=self.activation, SE=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = CenterBlock(512, 64, pool=False, SE=True)

        self.decoder4 = Decoder_v3(256, 64, 64, convT_ratio=1, SE=True)
        self.decoder3 = Decoder_v3(128, 64, 64, convT_ratio=1, SE=True)
        self.decoder2 = Decoder_v3(64, 64, 64, convT_ratio=1, SE=True)
        self.decoder1 = Decoder_v3(64, 64, 64, convT_ratio=1, SE=True)

        self.reducer = ConvBn2d(256, 64, kernel_size=1, padding=0)

        self.fpa = FPAModule1(in_ch=512, out_ch=256)

        self.logit = nn.Sequential(
            ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):

        x = self.conv1(x)  # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2)  # 64

        e1 = self.encoder1(p)  # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        f1 = self.fpa(e4)
        f2 = self.reducer(f1)

        d4 = self.decoder4(f2, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)  # 128

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(f2, scale_factor=16, mode='bilinear', align_corners=False)
        ], 1)
        logit = self.logit(f)
        return logit
import datetime
import math
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. This implementation
    contains restarts and T_mul.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mul, lr_min=0, last_epoch=-1, val_mode='max', save_snapshots=False):
        self.T_max = T_max
        self.T_mul = T_mul
        self.T_curr = 0
        self.lr_min = lr_min
        self.save_snapshots = save_snapshots
        self.val_mode = val_mode
        self.best_model_path = None
        self.reset = 0

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf

        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.T_curr % self.T_max

        if not r and self.last_epoch > 0:
            self.T_max *= self.T_mul
            self.T_curr = 1
            self.update_saving_vars()
        else:
            self.T_curr += 1

        return [self.lr_min + (base_lr - self.lr_min) *
                (1 + math.cos(math.pi * r / self.T_max)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None, save_dict=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.save_snapshots and save_dict is not None:
            self.save_best_model(save_dict)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def update_saving_vars(self):
        self.reset += 1
        self.best_model_path = None

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf


    def save_best_model(self, save_dict):
        metric = save_dict['metric']
        fold = save_dict['fold']
        save_dir = save_dict['save_dir']
        state_dict = save_dict['state_dict']

        if (self.val_mode == 'max' and metric > self.best_metric) or (
                self.val_mode == 'min' and metric < self.best_metric):
            # Update best metric
            self.best_metric = metric
            # Remove old file
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            # Save new best model weights
            date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
#             if fold is not None:
#                 self.best_model_path = os.path.join(
#                     save_dir,
#                     '{:}_Fold{:}_Epoach{}_reset{:}_val{:.3f}'.format(date, fold, self.last_epoch, self.reset, metric))
#             else:
#                 self.best_model_path = os.path.join(
#                     save_dir,
#                     '{:}_Epoach{}_reset{:}_val{:.3f}'.format(date, self.last_epoch, self.reset, metric))
            if fold is not None:
                self.best_model_path = os.path.join(
                    save_dir,
                    'Fold{:}_Epoach{}_reset{:}_val{:.3f}'.format(fold, self.last_epoch, self.reset, metric))
            else:
                self.best_model_path = os.path.join(
                    save_dir,
                    'Epoach{}_reset{:}_val{:.3f}'.format(self.last_epoch, self.reset, metric))

            torch.save(state_dict, self.best_model_path)
from skimage.transform import resize
from scipy import ndimage

def batch_encode(batch):
    rle = []
    for i in range(len(batch)):
        rle.append(do_length_encode(batch[i]))
    return rle

def unpad_im(im):
    return np.array([np.expand_dims(resize(np.squeeze(i), (768, 768), mode='constant', preserve_range=True), axis=0) for i in im])

def dice_accuracy(prob, truth, threshold=0.5, is_average=True, smooth=1e-12):
    # prob = unpad_im(prob)
    # truth = unpad_im(truth)

    batch_size = prob.size(0)
    p = prob.detach().contiguous().view(batch_size, -1)
    t = truth.detach().contiguous().view(batch_size, -1)

    p = p > threshold
    t = t > 0.5
    intersection = p & t
    union = p | t
    dice = (intersection.float().sum(1) + smooth) / (union.float().sum(1) + smooth)

    if is_average:
        dice = dice.sum() / batch_size

    return dice

def do_mAP(pred, truth, is_average=False, threshold=0.5):
    pred = pred > threshold
    batch_size = truth.shape[0]
    metric = []
    for batch in range(batch_size):
        p, t = pred[batch] > 0, truth[batch] > 0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    if is_average:
        return np.mean(metric)
    else:
        return metric
################### DICE ########################
def IoU(logit, truth, smooth=1):
    prob = torch.sigmoid(logit)
    intersection = torch.sum(prob * truth)
    union = torch.sum(prob + truth)
    iou = (2 * intersection + smooth) / (union + smooth)
    return iou

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logit, truth):
        iou = IoU(logit, truth, self.smooth)
        loss = 1 - iou
        return loss

################ FOCAL LOSS ####################
class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers

    def __init__(self, gamma=2, size_average=True):
        super(RobustFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = torch.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss

################# BCE + DICE ########################
class BCE_Dice(nn.Module):
    def __init__(self, smooth=1):
        super(BCE_Dice, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logit, truth):
        dice = self.dice(logit, truth)
        bce = self.bce(logit, truth)
        return dice + bce

############### LOVÁSZ-HINGE ########################
class Lovasz_Hinge(nn.Module):
    def __init__(self, per_image=True):
        super(Lovasz_Hinge, self).__init__()
        self.per_image = per_image

    def forward(self, logit, truth):
        return lovasz_hinge(logit, truth,
                            per_image=self.per_image)


############## BCE + LOVÁSZ #########################
class BCE_Lovasz(nn.Module):
    def __init__(self, per_image=True):
        super(BCE_Lovasz, self).__init__()
        self.per_image = per_image

    def forward(self, logit, truth):
        bce = binary_xloss(logit, truth)
        lovasz = lovasz_hinge(logit, truth, per_image=self.per_image)
        return bce + lovasz
"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()

    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
from contextlib import contextmanager
import time
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

##############################
TRAIN_PATH = '../input'
LOAD_PATHS = None

DEBUG = False
##############################
LOSS = 'lovasz'
OPTIMIZER = 'SGD'
VAL_MODE = 'max'
PRETRAINED = True
N_EPOCH = 1
BATCH_SIZE = 8
SIZE = 224
NET = UNetResNet34_SE_Hyper_FPA
ACTIVATION = 'relu'
###########OPTIMIZER###########
LR = 1e-2
USE_SCHEDULER = 'CosineAnneling'
MILESTONES = [20, 40, 75]
GAMMA = 0.5
PATIENCE = 10
T_MAX = 50
T_MUL = 1
LR_MIN = 0
##############################
COMMENT = 'SGDR (Tmax40, Tmul1), Lovasz, relu, pretrained'

train_dataset = Airbus_Dataset(TRAIN_PATH)
train_loader, val_loader = train_dataset.yield_dataloader(num_workers=1, batch_size=BATCH_SIZE, size=SIZE
                                              # auxiliary_df=TGS_Dataset.create_dataset_df(AUX_PATH)
                                              )


net = NET(lr=LR, debug=DEBUG, pretrained=PRETRAINED, fold=None, activation=ACTIVATION, val_mode=VAL_MODE, comment=COMMENT)
net.define_criterion(LOSS)
net.create_optmizer(optimizer=OPTIMIZER, use_scheduler=USE_SCHEDULER, milestones=MILESTONES,
                    gamma=GAMMA, patience=PATIENCE, T_max=T_MAX, T_mul=T_MUL, lr_min=LR_MIN)


net.train_network(train_loader, val_loader, n_epoch=N_EPOCH)
net.plot_training_curve(show=True)



