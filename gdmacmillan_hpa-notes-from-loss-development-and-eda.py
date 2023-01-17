%matplotlib inline

import os
import zipfile
import logging
import torch
import kaggle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from skimage import io, transform, img_as_float
from torch.utils.data import Dataset, DataLoader

os.chdir('..') # change working directory to 1 level up

LOCAL=False # SET THIS
GPU=True # SET THIS
sns.set(style="white")
train_csv_path = "data/train.csv" if LOCAL else "input/human-protein-atlas-image-classification/train.csv"
train_images_path = "data/train_images" if LOCAL else "input/human-protein-atlas-image-classification/train"
test_images_path = "data/test_images" if LOCAL else "input/human-protein-atlas-image-classification/test"
weights_path = "work/vgg16/hpa/vgg16-3ch-cyclic_lr-25epoch/25.pth" if LOCAL else "input/vgg16pretrainedweights/weights/25.pth"
sample_submission_path = "sample_submission.csv" if LOCAL else "input/human-protein-atlas-image-classification/sample_submission.csv"
train_labels = pd.read_csv(train_csv_path)
train_labels.head()
train_labels.shape
one_hot = train_labels.Target.str.get_dummies(sep=' ')
one_hot.columns = map(int, one_hot.columns); one_hot.head()
label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}
counts = one_hot.agg('sum')[:].rename(lambda x: label_names[x]).sort_values(ascending=False)
plt.figure(figsize=(12,10))
counts.plot('bar')
counts.head()
counts.tail()
train_labels = train_labels.join(one_hot.sort_index(axis=1))
tmp1 = train_labels.iloc[:,2:]
co_occur = tmp1.T.dot(tmp1); co_occur.head()
mask = np.zeros_like(co_occur, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(12,10))
sns.heatmap(co_occur, mask=mask, cmap=cmap, vmax=10000, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
co_occur1 = co_occur.apply(np.log, args=10)
mask = np.zeros_like(co_occur1, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,10))
sns.heatmap(co_occur1, mask=mask, cmap=cmap, vmax=10, vmin=0, center=5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
print(len([name for name in os.listdir(train_images_path) if os.path.isfile(os.path.join(train_images_path, name))]))
len(train_labels) * 4
id_list = train_labels.sample(4).Id.tolist(); id_list
def plot_images_row(img_id, ax_row):
    filters = ['red', 'green', 'blue', 'yellow']
    colormaps = ['Reds', 'Greens', 'Blues', 'Oranges']
    
    for c, ax, cmap in zip(filters, ax_row, colormaps):
        filename = img_id + '_' + c + '.png'
        img=mpimg.imread(os.path.join(train_images_path, filename))
        imgplot = ax.imshow(img, cmap=cmap)

fig = plt.figure(figsize=(12,10))
axes = fig.subplots(nrows=4, ncols=4)

for img_id, ax_row in zip(id_list, axes):
    plot_images_row(img_id, ax_row)

plt.tight_layout()
color = 'blue'
filename = id_list[0] + '_' + color + '.png'
img=mpimg.imread(os.path.join(train_images_path, filename))
plt.figure(figsize=(12,10))
plt.imshow(img, cmap="Blues")
def to_one_hot(df):
    tmp = df.Target.str.get_dummies(sep=' ')
    tmp.columns = map(int, tmp.columns)
    return df.join(tmp.sort_index(axis=1))

def get_image_ids_from_dir_contents(image_dir):
    all_images = [name for name in os.listdir(image_dir) \
                  if os.path.isfile(os.path.join(image_dir, name))]
    return list(set([name.split('_')[0] for name in all_images]))
class TrainImageDataset(Dataset):
    """Fluorescence microscopy images of protein structures training dataset"""

    def __init__(self,
        image_dir,
        label_file,
        transform=None,
        idxs=None,
        using_pil=False
    ):
        """
        Args:
            label_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.idxs = idxs
        self.labels = to_one_hot(pd.read_csv(label_file))
        self.using_pil = using_pil
        if self.idxs is not None:
            self.labels = self.labels.iloc[self.idxs, :].\
                                                reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        img_red = img_name + '_red.png'
        img_blue = img_name + '_blue.png'
        img_green = img_name + '_green.png'
        img_yellow = img_name + '_yellow.png'

        if self.using_pil:
            pth2img = lambda x: io.imread(x)
        else:
            pth2img = lambda x: img_as_float(io.imread(x))

        img_red = pth2img(os.path.join(self.image_dir, img_red))
        img_blue = pth2img(os.path.join(self.image_dir, img_blue))
        img_green = pth2img(os.path.join(self.image_dir, img_green))
        img_yellow = pth2img(os.path.join(self.image_dir, img_yellow))
        labels = self.labels.iloc[idx, 2:].values
        labels = labels.astype('int')
        sample = {'image_id': img_name,
                  'image_red': img_red,
                  'image_blue': img_blue,
                  'image_green': img_green,
                  'image_yellow': img_yellow,
                  'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class TestImageDataset(Dataset):
    """Fluorescence microscopy images of protein structures test dataset"""

    def __init__(self,
        image_dir,
        transform=None,
        using_pil=False
    ):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_ids = get_image_ids_from_dir_contents(image_dir)
        self.image_dir = image_dir
        self.transform = transform
        self.using_pil = using_pil

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_red = img_name + '_red.png'
        img_blue = img_name + '_blue.png'
        img_green = img_name + '_green.png'
        img_yellow = img_name + '_yellow.png'

        if self.using_pil:
            pth2img = lambda x: io.imread(x)
        else:
            pth2img = lambda x: img_as_float(io.imread(x))

        img_red = pth2img(os.path.join(self.image_dir, img_red))
        img_blue = pth2img(os.path.join(self.image_dir, img_blue))
        img_green = pth2img(os.path.join(self.image_dir, img_green))
        img_yellow = pth2img(os.path.join(self.image_dir, img_yellow))
        sample = {'image_id': img_name,
                  'image_red': img_red,
                  'image_blue': img_blue,
                  'image_green': img_green,
                  'image_yellow': img_yellow,
                  'labels' : np.zeros(28)}

        if self.transform:
            sample = self.transform(sample)

        return sample

class CombineColors(object):
    """Combines the the image in a sample to a given size."""

    def __call__(self, sample):
        img_name = sample['image_id']
        img_red = sample['image_red']
        img_blue = sample['image_blue']
        img_green = sample['image_green']
        img_yellow = sample['image_yellow']
        labels = sample['labels']
        image = np.dstack((img_red, img_green, img_blue, img_yellow))

        return {'image': image, 'labels': labels, 'image_id': img_name}


class ToPILImage(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.ToPILImage(self.mode)(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}


class RandomResizedCrop(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, size=224):
        self.size = size

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.RandomResizedCrop(self.size)(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}


class RandomHorizontalFlip(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.RandomHorizontalFlip()(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}


class Resize(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, size=256):
        self.size = size

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.Resize(self.size)(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}


class CenterCrop(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, size=224):
        self.size = size

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.CenterCrop(self.size)(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.ToTensor()(image)

        return {'image': image.type(torch.FloatTensor),
                'labels': torch. \
                    from_numpy(labels).type(torch.FloatTensor),
                'image_id': img_name}


class NumpyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch. \
                    from_numpy(image).type(torch.FloatTensor),
                'labels': torch. \
                    from_numpy(labels).type(torch.FloatTensor),
                'image_id': img_name}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input ``torch.*Tensor``
    i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        img_name = sample['image_id']
        image = sample['image']
        labels = sample['labels']
        image = transforms.Normalize(self.mean, self.std)(image)

        return {'image': image,
                'labels': labels,
                'image_id': img_name}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.\
                                            format(self.mean, self.std)


def get_transforms(pretrained=False):
    if pretrained:
        transform = {
            'TRAIN': transforms.Compose(
                            [CombineColors(),
                             ToPILImage(),
                             RandomResizedCrop(224),
                             RandomHorizontalFlip(),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]
            ),
            'DEV': transforms.Compose(
                            [CombineColors(),
                             ToPILImage(),
                             Resize(256),
                             CenterCrop(224),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ]
            )
        }
    else:
        transform = {
            'TRAIN': transforms.Compose(
                            [CombineColors(),
                             NumpyToTensor()
                             ]
            ),
            'DEV': transforms.Compose(
                            [CombineColors(),
                             NumpyToTensor()
                             ]
            )
        }

    return transform
train_dataset = TrainImageDataset(image_dir=train_images_path,
                                     label_file=train_csv_path)
sample = train_dataset[120]
sample
sample['image_red'].shape == sample['image_blue'].shape == \
sample['image_green'].shape ==  sample['image_yellow'].shape
len(sample['labels']) == 28
plt.figure(figsize=(12,10))
plt.imshow(sample['image_yellow'], cmap="Oranges")
transform = get_transforms(pretrained=False)
train_dataset = TrainImageDataset(image_dir=train_images_path,
                                     label_file=train_csv_path,
                                     transform=transform['TRAIN'])
kwargs = {'batch_size': 32}
trainLoader = DataLoader(train_dataset, shuffle=True, **kwargs)
data = next(iter(trainLoader))
inputs, labels = data['image'], data['labels']
nbs = kwargs['batch_size'] # num examples in batch
ncl = 28 # num classes
hjk = labels.sum(0); hjk
cls_labels = torch.arange(ncl); cls_labels # not required 
sorted_cls_labels = np.argsort(hjk); sorted_cls_labels
sorted_hjk = hjk[sorted_cls_labels]; sorted_hjk
th = .5 * nbs

def get_min_class_boundary(arr):
    for idx in torch.arange(len(arr)):
        if arr[:idx].sum() > th:
            return idx - 1
    return arr.size
bound = get_min_class_boundary(sorted_hjk); bound
sorted_hjk[:bound].sum() # should be less than or equal  16
sorted_hjk = sorted_hjk[:bound]
min_cls_labels = sorted_cls_labels[:bound]
idxs = np.argsort(min_cls_labels) # unsort
min_cls_labels = min_cls_labels[idxs] 
hjk = sorted_hjk[idxs]
print(min_cls_labels)
print(hjk)
msk = hjk > 1
hjk = hjk[msk]
min_cls_labels = min_cls_labels[msk]
print(min_cls_labels)
print(hjk)
def get_minority_classes(y, batchSz):
    sorted_hjk, ix = y.sum(0).sort()
    mask = torch.cumsum(sorted_hjk, 0) <= .5 * batchSz
    sorted_hjk = sorted_hjk[mask]
    sorted_, sorted_ix = ix = ix[mask].sort()
    
    return sorted_[sorted_hjk[sorted_ix] > 1]

min_cls_labels = get_minority_classes(labels, nbs); min_cls_labels
bs = (nbs, ncl) # batch array size
preds = np.random.rand(*bs) # random predictions for batch 
preds = torch.Tensor(preds)
y_min = labels[:, min_cls_labels]
# y_min = labels.numpy()[:, min_cls_labels.numpy()]
msk = y_min == 1
P = torch.nonzero(msk); P # anchor instances
# P = np.argwhere(msk)
N = torch.nonzero(~msk)
# N = np.argwhere(~msk)
preds_min = preds[:, min_cls_labels]
preds_P = preds_min[msk]
preds_N = preds_min[~msk]
k = 3
preds_P[np.argsort(preds_P)][:k]
preds_N[np.argsort(preds_N)][-k:]
preds_min[:5] # head
k = 3
for idx, row in enumerate(P):
    anchor_idx, anchor_class = row
    mask = (P[:, 1] == anchor_class)
    mask[idx] = 0
    pos_idxs = P[mask]
    pos_preds, sorted_= preds_min[pos_idxs[:, 0], pos_idxs[:, 1]].sort()
    pos_idxs = pos_idxs[sorted_][:k]
    pos_preds = pos_preds[:k]

    mask = (N[:, 1] == anchor_class)
    neg_idxs = N[mask]
    neg_preds, sorted_= preds_min[neg_idxs[:, 0], neg_idxs[:, 1]].sort()
    neg_idxs = neg_idxs[sorted_][-k:]
    neg_preds = neg_preds[:k]
    
    a = [idx]
    n_p = pos_idxs.shape[0]
    n_n = neg_idxs.shape[0]
    grid = torch.stack(torch.meshgrid([torch.Tensor(a).long(), torch.arange(n_p), torch.arange(n_n)])).reshape(3, -1).t()
    print(torch.cat([P[grid[:, 0]], pos_idxs[grid[:, 1]], neg_idxs[grid[:, 2]]], 1))
    print("")
    print(torch.stack([preds_P[grid[:, 0]], pos_preds[grid[:, 1]], neg_preds[grid[:, 2]]], 1))
    print("")

# def mine_positives(anchor, labels, predictions):
#     cls = np.argwhere(labels[anchor] == 1)
#     P = np.argwhere(labels == 1)
#     preds_P = predictions[labels == 1]
#     out = P[np.isin(P[:, 1], cls)]
#     out_preds = preds_P[np.isin(P[:, 1], cls)]
#     input_mask = out[:, 0] != anchor
#     out = out[input_mask]
#     out_preds = out_preds[input_mask]
#     sorted_ = out_preds.argsort()
#     return out[sorted_], out_preds[sorted_]

# def mine_negatives(anchor, labels, predictions):
#     cls = np.argwhere(labels[anchor] == 0)
#     N = np.argwhere(labels == 0)
#     preds_N = predictions[labels == 0]
#     out = N[np.isin(N[:, 1], cls)]
#     out_preds = preds_N[np.isin(N[:, 1], cls)]
#     sorted_ = out_preds.argsort()
#     return out[sorted_], out_preds[sorted_]


# anchor_idxs = P[:, 0]
# k = 3
# for idx in anchor_idxs:
#     anc_examples, anc_preds = P[P[:, 0] == idx], preds_P[P[:, 0] == idx]
#     pos_examples, pos_preds = mine_positives(idx, y_min, preds_min)
#     neg_examples, neg_preds = mine_negatives(idx, y_min, preds_min)
#     pos_examples = pos_examples[:k]
#     neg_examples = neg_examples[-k:]
#     n_a = anc_examples.shape[0]
#     n_p = pos_examples.shape[0]
#     n_n = neg_examples.shape[0]
#     grid = np.array(np.meshgrid(np.arange(n_a), np.arange(n_p), np.arange(n_n))).T.reshape(-1,3)
#     print(anc_examples[grid[:, 0]], pos_examples[grid[:, 1]], neg_examples[grid[:, 2]])
#     print("")
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample

    source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.l1_loss(anchor, positive, reduction='sum')
        distance_negative = F.l1_loss(anchor, negative, reduction='sum')
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class IncrementalClassRectificationLoss(nn.Module):

    def __init__(self,
        margin,
        alpha,
        batchSz,
        k,
        class_level_hard_mining=True,
        sigmoid=True
    ):
        super(IncrementalClassRectificationLoss, self).__init__()

        self.margin = margin
        self.alpha = alpha
        self.batchSz = batchSz
        self.k = k
        self.class_level_hard_mining = class_level_hard_mining
        self.sigmoid = sigmoid
        self.trip_loss = TripletLoss(margin)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target, X):
        bce = self.bce(input, target)
        idxs = get_minority_classes(target, batchSz=self.batchSz)
        if self.sigmoid:
            input = torch.sigmoid(input)
            y_min = target[:, idxs]
            preds_min = input[:, idxs]
        else:
            y_min = target[:, idxs]
            preds_min = input[:, idxs]

        y_mask = y_min == 1
        P = torch.nonzero(y_mask)
        N = torch.nonzero(~y_mask)
        preds_P = preds_min[y_mask]

        k = self.k
        idx_tensors = []
        pred_tensors = []
        # would like to vectorize this
        for idx, row in enumerate(P):
            anchor_idx, anchor_class = row
            mask = (P[:, 1] == anchor_class)
            mask[idx] = 0
            pos_idxs = P[mask]
            pos_preds, sorted_= preds_min[pos_idxs[:, 0], pos_idxs[:, 1]].sort()
            pos_idxs = pos_idxs[sorted_][:k]
            pos_preds = pos_preds[:k]

            mask = (N[:, 1] == anchor_class)
            neg_idxs = N[mask]
            neg_preds, sorted_= preds_min[neg_idxs[:, 0], neg_idxs[:, 1]].sort()
            neg_idxs = neg_idxs[sorted_][-k:]
            neg_preds = neg_preds[:k]

            a = [idx] # anchor index in P
            n_p = pos_idxs.shape[0]
            n_n = neg_idxs.shape[0]
            # create 2d array with indices for anchor, pos and neg examples
            grid = torch.stack(torch.meshgrid([torch.Tensor(a).long(), torch.arange(n_p), torch.arange(n_n)])).reshape(3, -1).t()
            idx_tensors.append(torch.cat([P[grid[:, 0]], pos_idxs[grid[:, 1]], neg_idxs[grid[:, 2]]], 1))
            pred_tensors.append(torch.stack([preds_P[grid[:, 0]], pos_preds[grid[:, 1]], neg_preds[grid[:, 2]]], 1))

        try:
            if self.class_level_hard_mining:
                idx_tensors = torch.cat(idx_tensors, 0)
                pred_tensors = torch.cat(pred_tensors, 0)
            else:
                # TODO: implement instance level hard mining
                pass
            crl = self.trip_loss(pred_tensors[:, 0], pred_tensors[:, 1], pred_tensors[:, 2])
            loss = self.alpha * crl + (1 - self.alpha) * bce

            return loss

        except RuntimeError:
            # TODO: figure out why we are sometimes getting RuntimeError in test
            logging.warning('RuntimeError in loss statement')

            return bce
omega = counts.min() / counts.max() # class imbalance measure as 
eta = 0.01
alpha = omega * eta
print(alpha)
# don't use sigmoid layer since preds are already in 0-1 range
criterion = IncrementalClassRectificationLoss(0.5, alpha, 28, 3, sigmoid=False)
criterion(preds, labels, inputs)
class ArgContainer():
    def __init__(self, 
                 network_name,
                 crit,
                 batchSz, 
                 train_images_path,
                 test_images_path,
                 train_csv_path, 
                 nSubsample, 
                 pretrained,
                 cuda,
                 sigmoid,
                 thresholds,
    ):
        self.network_name = network_name
        self.crit = crit
        self.batchSz = batchSz
        self.train_images_path = train_images_path
        self.test_images_path = test_images_path
        self.train_csv_path = train_csv_path
        self.nSubsample = nSubsample
        self.pretrained = pretrained
        self.cuda = cuda
        self.sigmoid = sigmoid
        self.thresholds = thresholds
def get_dataset(args, idxs=None, train=True):
    if args.pretrained:
        using_pil = True
    else:
        using_pil = False

    transform = get_transforms(args.pretrained)
    if train:
        image_dir = args.train_images_path
        label_file = args.train_csv_path
        if label_file is None:
            raise ValueError('no label_file provided for training')
        if idxs is None:
            raise ValueError('must specify idxs for training')
        dataset = TrainImageDataset(
                         image_dir=image_dir,
                         label_file=label_file,
                         transform=transform['TRAIN'],
                         idxs=idxs,
                         using_pil=using_pil)
    else:
        image_dir = args.test_images_path
        dataset = TestImageDataset(
                         image_dir=image_dir,
                         transform=transform['DEV'],
                         using_pil=using_pil)

    return dataset

def get_train_test_split(args, val_split=0.10, distributed=False, **kwargs):
    n_subsample = args.nSubsample

    with open(args.train_csv_path, 'r') as f:
        n_images = sum(1 for row in f.readlines()) - 1 # -1 for header row
    if n_subsample != 0:
        arr = np.random.choice(n_images, n_subsample, replace=False)
        train_idxs = arr[:int(n_subsample * (1 - val_split))]
        dev_idxs = arr[int(n_subsample * (1 - val_split)):]
    else:
        arr = np.random.choice(n_images, n_images, replace=False)
        train_idxs = arr[:int(n_images * (1 - val_split))]
        dev_idxs = arr[int(n_images * (1 - val_split)):]

    trainset = get_dataset(args, idxs=train_idxs)
    devset = get_dataset(args, idxs=dev_idxs)

    if distributed:
        trainLoader, devLoader, args.batchSz = partition_dataset(trainset, devset, args.batchSz)
    else:
        trainLoader = DataLoader(trainset, shuffle=True, **kwargs)
        devLoader = DataLoader(devset, shuffle=False, **kwargs)

    return trainLoader, devLoader

def get_loss_function(lf='bce', args=None):
    if lf == 'bce':
        return BCEWithLogitsLoss()

    elif lf == 'f1':
        return f1_loss

    elif lf == 'crl':
        if args:
            return IncrementalClassRectificationLoss(*args)
        raise ValueError('args for CRL not found')
    else:
        raise ModuleNotFoundError('loss function not found')
import torchvision.models as models
import torch.nn as nn

from torch import cat


RESNET_ENCODERS = {
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152,
}

VGG_CLASSIFIERS = {
    11: models.vgg11,
    13: models.vgg13,
    16: models.vgg16,
    19: models.vgg19,
}

VGG_BN_CLASSIFIERS = {
    11: models.vgg11_bn,
    13: models.vgg13_bn,
    16: models.vgg16_bn,
    19: models.vgg19_bn,
}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5) # 4 channel in, 6 channels out, filter size 5
        self.pool = nn.MaxPool2d(2, 2) # 6 channel in, 6 channels out, filter size 2, stride 2
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 channel in, 16 channels out, filter size 5
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 28)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(torch.relu(x))
        x = self.conv2(x)
        x = self.pool(torch.relu(x))
        x = x.view(-1, 16 * 125 * 125)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return x


class Resnet4Channel(nn.Module):
    def __init__(self, encoder_depth=34, pretrained=True, num_classes=28):
        super().__init__()

        encoder = RESNET_ENCODERS[encoder_depth](pretrained=pretrained)

        if pretrained:
            for param in encoder.parameters():
                param.requires_grad=False

        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        w = encoder.conv1.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1.weight = nn.Parameter(cat((w,w[:,:1,:,:]),dim=1))

        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.avgpool = encoder.avgpool
        num_features = encoder.fc.in_features
        self.fc = nn.Linear(num_features, 28)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class VGG4Channel(nn.Module):
    def __init__(self, n_layers=11, batch_norm=False, pretrained=True, num_classes=28):
        super().__init__()

        if batch_norm:
            vgg_net = VGG_BN_CLASSIFIERS[n_layers](pretrained=pretrained)
        else:
            vgg_net = VGG_CLASSIFIERS[n_layers](pretrained=pretrained)

        if pretrained:
            for param in vgg_net.features.parameters():
                param.requires_grad=False
            for param in vgg_net.classifier.parameters():
                param.requires_grad=False

        # initialize conv2d to take in 4 channels instead of 3
        feature_layers = []
        w = vgg_net.features[0].weight
        conv2d = nn.Conv2d(4, 64, kernel_size=3, padding=1) # Create 2d conv layer
        conv2d.weight = nn.Parameter(cat((w,w[:,:1,:,:]),dim=1))
        feature_layers.append(conv2d)

        remaining_features = list(vgg_net.features.children())[1:] # Remove first layer
        feature_layers.extend(remaining_features)

        # swap last layer for fc layer with 28 outputs
        num_features = vgg_net.classifier[-1].in_features
        classifier_layers = list(vgg_net.classifier.children())[:-1] # Remove last layer
        classifier_layers.extend([nn.Linear(num_features, 28)]) # Add layer with 28 outputs. activation in loss function

        self.features = nn.Sequential(*feature_layers)
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def resnet34(pretrained):
    net = Resnet4Channel(encoder_depth=34, pretrained=pretrained)
    return net

def resnet50(pretrained):
    net = Resnet4Channel(encoder_depth=50, pretrained=pretrained)
    return net

def resnet101(pretrained):
    net =  Resnet4Channel(encoder_depth=101, pretrained=pretrained)
    return net

def resnet152(pretrained):
    net = Resnet4Channel(encoder_depth=152, pretrained=pretrained)
    return net

def vgg11(pretrained):
    net = VGG4Channel(n_layers=11, batch_norm=False, pretrained=pretrained)
    return net

def vgg13(pretrained):
    net = VGG4Channel(n_layers=13, batch_norm=False, pretrained=pretrained)
    return net

def vgg16(pretrained):
    net = VGG4Channel(n_layers=16, batch_norm=False, pretrained=pretrained)
    return net

def vgg19(pretrained):
    net = VGG4Channel(n_layers=19, batch_norm=False, pretrained=pretrained)
    return net

def vgg11_bn(pretrained):
    net = VGG4Channel(n_layers=11, batch_norm=True, pretrained=pretrained)
    return net

def vgg13_bn(pretrained):
    net = VGG4Channel(n_layers=13, batch_norm=True, pretrained=pretrained)
    return net

def vgg16_bn(pretrained):
    net = VGG4Channel(n_layers=16, batch_norm=True, pretrained=pretrained)
    return net

def vgg19_bn(pretrained):
    net = VGG4Channel(n_layers=19, batch_norm=True, pretrained=pretrained)
    return net

def baseline(pretrained):
    if pretrained:
        print('Baseline net not pretrained. Training from scratch')
    return Net()

NETWORKS_DICT = {
    'resnet34' : resnet34,
    'resnet50' : resnet50,
    'resnet101' : resnet101,
    'resnet152' : resnet152,
    'vgg11' : vgg11,
    'vgg13' : vgg13,
    'vgg16' : vgg16,
    'vgg19' : vgg19,
    'vgg11_bn' : vgg11_bn,
    'vgg13_bn' : vgg13_bn,
    'vgg16_bn' : vgg16_bn,
    'vgg19_bn' : vgg19_bn,
}
args = ArgContainer("resnet152", "crl", 64, train_images_path, test_images_path, train_csv_path, 0, True, True if GPU else False, True, None)

kwargs = {'batch_size': args.batchSz}
net = NETWORKS_DICT[args.network_name](args.pretrained)
net.eval()
trainLoader, devLoader = get_train_test_split(args, **kwargs)
net = torch.nn.DataParallel(net) if LOCAL else torch.nn.DataParallel(net).cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
lf_args = [0.5, 8.537058595265812e-06, args.batchSz, 5, True, True]
criterion = get_loss_function('crl', lf_args)
def plot_lr(optimizer, net, trainLoader, criterion, start_lr=-7, end_lr=-1, num_iter=100):
    xs = np.logspace(start_lr, end_lr, num_iter)
    ys = []
    for i, data in enumerate(trainLoader, 0):
        if i == 100:
            break
        for param_group in optimizer.param_groups:
            param_group['lr'] = xs[i]
        net.train()
        # get the inputs
        inputs, labels = data['image'].cuda(), data['labels'].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss_inputs = (outputs, labels, inputs)
        loss = criterion(*loss_inputs)
        loss.backward()
        optimizer.step()
        ys.append(loss.item())
    plt.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
    plt.plot(xs, ys, 'b-')
    plt.xscale('log')
    plt.title('Learning rate curve')
    plt.ylabel('loss')
    plt.xlabel('learning rate')
# if not LOCAL and GPU:
#     plot_lr(optimizer, net, trainLoader, criterion)
def positive_predictions(predictions):
    positives = []

    for prediction in predictions:
        output = []
        i = 0
        for label in prediction:
            if(label == 1):
                output.append(str(i))
            i += 1
        positives.append(' '.join(output))

    return positives

def predict(args, net, dataLoader, predF):
    net.eval()

    with torch.no_grad():
        predF.write('Id,Predicted\n')
        print('writing predictions...')
        for batch_idx, data in enumerate(dataLoader):
            inputs, image_ids = data['image'], data['image_id']
            if args.cuda:
                inputs = inputs.cuda()

            outputs = net(inputs)
            if args.sigmoid:
                outputs = torch.sigmoid(outputs)
            if args.thresholds is not None:
                thresholds = [float(val) for val in
                                            args.thresholds.split(",")]

                thresholds = torch.tensor(thresholds)
                if args.cuda:
                    thresholds = thresholds.cuda()
                pred = outputs.data.gt(thresholds)
            else:
                pred = outputs.data.gt(0.5)
            preds = positive_predictions(pred)
            for _ in zip(image_ids, preds):
                predF.write(",".join(_) + '\n')
                predF.flush()

def get_testloader(args, **kwargs):
    testset = get_dataset(args, train=False)
    testloader = DataLoader(testset, shuffle=False, **kwargs)

    return testloader
testLoader = get_testloader(args, **kwargs)
if GPU:
    net.module.load_state_dict(torch.load(weights_path))
else:
    net.module.load_state_dict(torch.load(weights_path), map_location='cpu')
args.thresholds = ".3"
submission_path = 'working/submission.csv'
predF = open(submission_path, 'a')

predict(args, net, testLoader, predF)

predF.close
sample_df = pd.read_csv(sample_submission_path, encoding='utf-8')
output_df = pd.read_csv(submission_path, encoding='utf-8')
output_df = output_df.replace(np.nan, '', regex=True)
new_df = sample_df.merge(output_df, left_on='Id', right_on='Id', how='outer')
new_df = new_df.loc[:, ['Id', 'Predicted_y']]
new_df.columns = ['Id','Predicted']
new_df.to_csv(submission_path, index=False, encoding='utf-8')
