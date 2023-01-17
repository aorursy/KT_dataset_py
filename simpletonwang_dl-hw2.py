import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from PIL import Image

from pathlib import Path

from matplotlib.patches import Rectangle

from scipy.io import loadmat
devkit_path = Path('../input/car_devkit/devkit')

train_path = Path('../input/cars_train/cars_train')

test_path = Path('../input/cars_test/cars_test')
os.listdir(devkit_path)
cars_meta = loadmat(devkit_path/'cars_meta.mat')

cars_train_annos = loadmat(devkit_path/'cars_train_annos.mat')

cars_test_annos = loadmat(devkit_path/'cars_test_annos.mat')
labels = [c for c in cars_meta['class_names'][0]]

labels = pd.DataFrame(labels, columns=['labels'])

labels.head()
frame = [[i.flat[0] for i in line] for line in cars_train_annos['annotations'][0]]

columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']

df_train = pd.DataFrame(frame, columns=columns)

df_train['class'] = df_train['class']-1 # Python indexing starts on zero.

df_train['fname'] = [train_path/f for f in df_train['fname']] #  Appending Path

df_train.head()
df_train = df_train.merge(labels, left_on='class', right_index=True)

df_train = df_train.sort_index()

df_train.head()
frame = [[i.flat[0] for i in line] for line in cars_test_annos['annotations'][0]]

columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']

df_test = pd.DataFrame(frame, columns=columns)

df_test['fname'] = [test_path/f for f in df_test['fname']] #  Appending Path

df_test.head()
# Returns (Image, title, rectangle patch) for drawing

def get_assets(df, i):

    is_train = df is df_train

    folder = train_path if is_train else test_path

    image = Image.open(df['fname'][i])

    title = df['labels'][i] if is_train else 'Unclassified'



    xy = df['bbox_x1'][i], df['bbox_y1'][i]

    width = df['bbox_x2'][i] - df['bbox_x1'][i]

    height = df['bbox_y2'][i] - df['bbox_y1'][i]

    rect = Rectangle(xy, width, height, fill=False, color='r', linewidth=2)

    

    return (image, title, rect)
def display_image(df, i):

    image, title, rect = get_assets(df, i)

    print(title)



    plt.imshow(image)

    plt.axis('off')

    plt.title(title)

    plt.gca().add_patch(rect)
display_image(df_train, 0)
def display_range(end, start = 0):



    n = end - start

    fig, ax = plt.subplots(n, 2, figsize=(15, 5*end))



    for i in range(start, end):

        line = i - start

        

        im, title, rect = get_assets(df_train, i)

        sub = ax[line, 0]

        sub.imshow(im)

        sub.axis('off')

        sub.set_title(title)

        sub.add_patch(rect)

        

        im, title, rect = get_assets(df_test, i)

        sub = ax[line, 1]

        sub.imshow(im)

        sub.axis('off')

        sub.set_title(title)

        sub.add_patch(rect)

        

    plt.show()
display_range(5)
import torch

import torch.nn as nn

from pathlib import Path

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

from torchvision import models

import torch.optim as optim

import cv2
# def resize_images(im, sz=300):

#     sz2 = int(1.778*sz)

#     return cv2.resize(im, (sz2, sz))



def normalize(im):

    """Normalizes images with Imagenet stats."""

    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])

    return (im - imagenet_stats[0])/imagenet_stats[1]



import math

def crop(im, r, c, target_r, target_c): return im[r:r+target_r, c:c+target_c]



# random crop to the original size

def random_crop(x, r_pix=8):

    """ Returns a random crop"""

    r, c,*_ = x.shape

    r, c,*_ = x.shape

    c_pix = round(r_pix*c/r)

    rand_r = random.uniform(0, 1)

    rand_c = random.uniform(0, 1)

    start_r = np.floor(2*rand_r*r_pix).astype(int)

    start_c = np.floor(2*rand_c*c_pix).astype(int)

    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)



def center_crop(x, r_pix=8):

    r, c,*_ = x.shape

    c_pix = round(r_pix*c/r)

    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)





def rotate_cv(im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):

    """ Rotates an image by deg degrees"""

    r,c,*_ = im.shape

    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)

    return cv2.warpAffine(im,M,(c,r), borderMode=mode, 

                          flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def make_bb_px(y, x_shape):

    """ Makes an image of size x retangular bounding box"""

    r,c,*_ = x_shape 

    Y = np.zeros((r, c))

    y = y.astype(np.int)

    Y[y[0]:y[2], y[1]:y[3]] = 1.

    return Y



def to_bb(Y):

    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""

    cols, rows = np.nonzero(Y)

    if len(cols)==0: return np.zeros(4, dtype=np.float32)

    top_row = np.min(rows)

    left_col = np.min(cols)

    bottom_row = np.max(rows)

    right_col = np.max(cols)

    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)



def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):

    """ Rotates an image by deg degrees"""

    r,c,*_ = im.shape

    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)

    if y:

        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)

    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)



def transformsXY(path, bb, transforms):

    x = cv2.imread(str(path)).astype(np.float32)

    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255

    Y = make_bb_px(bb, x.shape)

    if transforms:

        rdeg = (np.random.random()-.50)*20

        x = rotate_cv(x, rdeg)

        Y = rotate_cv(Y, rdeg, y=True)

        if np.random.random() > 0.5: 

            x = np.fliplr(x).copy()

            Y = np.fliplr(Y).copy()

        x, Y = random_cropXY(x, Y)

    else:

        x, Y = center_crop(x), center_crop(Y)

    return x, to_bb(Y)



def create_corner_rect(bb, color='red'):

    bb = np.array(bb, dtype=np.float32)

    return plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color=color,

                         fill=False, lw=3)

def show_corner_bb(im, bb):

    plt.imshow(im)

    plt.gca().add_patch(create_corner_rect(bb))
path = df_train.iloc[1,5]

bb = df_train.iloc[1,:4]

img = cv2.imread(str(path))


plt.imshow(img)

plt.gca().add_patch(create_corner_rect(bb))

plt.show()
class CarDataset(Dataset):

    def __init__(self, df, transforms=False, labels=True,sz=300):

        self.paths = df['fname'].values

        self.transforms = transforms

        self.labels = labels

        self.sz=sz

        self.sz2=int(self.sz*1.778)

        self.bb = df.iloc[:,:4].values

        if labels:            

            self.label = df['class'].values

        

        

    def __len__(self):

        return len(self.paths)

    

    def __getitem__(self, idx):

        path = self.paths[idx]

        x = cv2.imread(str(path))#.astype(np.float32)

        h = x.shape[1]

        w = x.shape[0]

        x_scale = self.sz2 / h

        y_scale = self.sz / w

        

#         print(x_scale, y_scale)

        x = cv2.resize(x, (self.sz2, self.sz));

#         print(x.shape)

        

        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255

#         print(x.shape)



        x = normalize(x)

        x = np.rollaxis(x, 2)

        (origLeft, origTop, origRight, origBottom) = self.bb[idx]

        scaled_left = int(np.round(origLeft * x_scale))

        scaled_top = int(np.round(origTop * y_scale))

        scaled_right = int(np.round(origRight * x_scale))

        scaled_bottom = int(np.round(origBottom * y_scale))

        y_scaled_bb = np.array([scaled_left,scaled_top,scaled_right,scaled_bottom])                

        if self.labels:

            y_class = self.label[idx]

            # original frame as named values

#             print(self.bb[idx])



#             print(y_scaled_bb)

            return x, y_class, y_scaled_bb

        else:

            return x, y_scaled_bb
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(df_train, test_size=0.2)
train_ds = CarDataset(train_df, labels=True)

valid_ds = CarDataset(valid_df, labels=True)

test_ds = CarDataset(df_test, labels=False)
BATCH = 40

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

valid_dl = DataLoader(valid_ds, batch_size=BATCH)

test_dl = DataLoader(test_ds,batch_size=BATCH)
x, y_class, y_bb  = next(iter(train_dl))
plt.imshow(x[1].cpu().numpy().transpose(1,2,0))



plt.gca().add_patch(create_corner_rect(y_bb[1]))
class Net2(nn.Module):

    def __init__(self):

        super(Net2, self).__init__()

        self.resnet = models.resnet34(pretrained=True)

        self.freeze()

        layers = list(self.resnet.children())[:8]

        self.groups = nn.ModuleList([nn.Sequential(*h) for h in [layers[:6], layers[6:]]])

        self.linears = nn.ModuleList([nn.Linear(512, 196), nn.Linear(512, 4)])

        self.groups.append(self.linears)



    def forward(self, x):

        for group in self.groups[:2]:

            x = group(x)

        x = F.relu(x)

        x = nn.AdaptiveAvgPool2d((1,1))(x)

        x = x.view(x.shape[0], -1)

        x1 = self.linears[0](x)

        x2 = self.linears[1](x)

        return x1, x2

    

    def freeze(self):

        for param in self.resnet.parameters():

            param.requires_grad = False

    

    def unfreeze(self,  group_idx:int):

        group = self.groups[group_idx]

        parameters = filter(lambda x: hasattr(x,'requires_grad'), group.parameters())

        for p in parameters: 

            p.requires_grad = True
net = Net2().cuda()

net
def cosine_segment(start_lr, end_lr, iterations):

    i = np.arange(iterations)

    c_i = 1 + np.cos(i*np.pi/iterations)

    return end_lr + (start_lr - end_lr)/2 *c_i



def get_cosine_triangular_lr(max_lr, iterations):

    min_start, min_end = max_lr/25, max_lr/(25*1e4)

    iter1 = int(0.3*iterations)

    iter2 = iterations - iter1

    segs = [cosine_segment(min_start, max_lr, iter1), cosine_segment(max_lr, min_end, iter2)]

    return np.concatenate(segs)

def create_optimizer(model, lr0):

    param_groups = [list(model.groups[i].parameters()) for i in range(3)]

    params = [{'params':p, 'lr': lr} for p,lr in zip(param_groups, [lr0/9, lr0/3, lr0] )]

    return optim.Adam(params)



def update_optimizer(optimizer, group_lrs):

    for i, param_group in enumerate(optimizer.param_groups):

        param_group["lr"] = group_lrs[i]
def val_metrics2(model, valid_dl, C=1000):

    model.eval()

    total = 0

    sum_loss = 0

    correct = 0 

    for x, y_class, y_bb in valid_dl:

        batch = y_class.shape[0]

        x = x.cuda().float()

        y_class = y_class.cuda()

        y_bb = y_bb.cuda().float()

#         z = z.cuda().float()

        out_class, out_bb = model(x)

        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")

        loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)

        loss_bb = loss_bb.sum()

        loss = loss_class + loss_bb/C

        _, pred = torch.max(out_class, 1)

        correct += pred.eq(y_class).sum().item()

        sum_loss += loss.item()

        total += batch

    return sum_loss/total, correct/total
x.shape
display(net(x.cuda().float())[0].shape)

display(net(x.cuda().float())[1].shape)
from tqdm import tqdm_notebook
def train_triangular_policy2(model, train_dl, valid_dl, max_lr=0.01, epochs=9, C=1000):

    idx = 0

    iterations = epochs*len(train_dl)

    lrs = get_cosine_triangular_lr(max_lr, iterations)

    optimizer = create_optimizer(model, lrs[0])

    prev_val_acc = 0.0

    for i in range(epochs):

        model.train()

        total = 0

        sum_loss = 0

        for x, y_class, y_bb in tqdm_notebook(train_dl):

            lr = lrs[idx]

            update_optimizer(optimizer, [lr/9, lr/3, lr])

            batch = y_class.shape[0]

#             print(x)

            x = x.cuda().float()

            y_class = y_class.cuda()

            y_bb = y_bb.cuda().float()

#             z = z.cuda().float()

            out_class, out_bb = model(x)

#             print(out_class, out_bb)

            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")

#             print(out_bb,y_bb)

            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)

            

            loss_bb = loss_bb.sum()

            loss = loss_class + loss_bb/C

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            idx += 1

            if idx == int(0.1*iterations):

                model.unfreeze(1)

                print(idx, "unfreezing 1")

            if idx == int(0.2*iterations):

                model.unfreeze(0)

                print(idx, "unfreezing 0")

            total += batch

            sum_loss += loss.item()

        train_loss = sum_loss/total

        val_loss, val_acc = val_metrics2(model, valid_dl, C)

        print("train_loss %.3f val_loss %.3f val_acc %.3f" % (train_loss, val_loss, val_acc))

        if val_acc > prev_val_acc: 

            prev_val_acc = val_acc

            if val_acc > 0.9:

                path = "{0}/models/model_resnet34_loss_{1:.0f}.pth".format(PATH, 100*val_acc)

                save_model(model, path)

                print(path)

    return sum_loss/total
train_triangular_policy2(net, train_dl, valid_dl, max_lr=0.001, epochs=10)


test_dl = DataLoader(test_ds,batch_size=1)
test_x, test_bb_y = next(iter(test_dl))
test_x = test_x.cuda().float()

test_class, test_bb = net(test_x)
plt.imshow(test_x[0].cpu().numpy().transpose(1,2,0))

plt.gca().add_patch(create_corner_rect(test_bb_y[0].detach().cpu(), color='blue'))

plt.gca().add_patch(create_corner_rect(test_bb[0].detach().cpu()))
test_class.argmax(1)