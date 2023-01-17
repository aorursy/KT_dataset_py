!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!export XLA_USE_BF16=1

!pip install -q colored

!pip install -q efficientnet_pytorch
import os

import gc

import cv2

import time

import numpy as np

import pandas as pd



from colored import fg, attr

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from sklearn.utils import shuffle



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import torch

import torch.nn as nn

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader

from torch import FloatTensor, LongTensor, DoubleTensor

from torch.utils.data.sampler import WeightedRandomSampler



import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp



from efficientnet_pytorch import EfficientNet

from albumentations import Normalize, VerticalFlip, HorizontalFlip, Compose
!unzip ../input/siim-isic-melanoma-resized-images/test.zip

!unzip ../input/siim-isic-melanoma-resized-images/train_1.zip

!unzip ../input/siim-isic-melanoma-resized-images/train_2.zip
W = 512

H = 512

B = 0.5

SPLIT = 0.8

SAMPLE = True

MU = [0.485, 0.456, 0.406]

SIGMA = [0.229, 0.224, 0.225]



EPOCHS = 5

LR = 1e-3, 1e-3

BATCH_SIZE = 32

VAL_BATCH_SIZE = 32

MODEL = 'efficientnet-b3'

IMG_PATHS = ['../working/test',

             '../working/train_1',

             '../working/train_2']
PATH_DICT = {}

for folder_path in tqdm(IMG_PATHS):

    for img_path in os.listdir(folder_path):

        PATH_DICT[img_path] = folder_path + '/'
np.random.seed(42)

torch.manual_seed(42)
print(os.listdir('../input/siim-isic-melanoma-classification'))
test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test_df.head(10)
train_df.head(10)
sfn = lambda x: "🔎 " + str(x)

fs = (fg('#7efc79'), attr('reset'))

gs = (fg('#fac3c3'), attr('reset'))



for column in train_df.columns[[7, 2, 6, 4, 5]]:

    column_string = ("%s" + column + "%s 🎯\n") % gs

    column_values = set(train_df[column].apply(sfn))

    column_values = "\n".join(sorted(column_values, key=len))

    hyphens = ("%s" + ''.join(['-']*len(column)) + "\n%s") % gs

    print(hyphens + column_string + hyphens + "\n" + ("%s" + column_values + "\n%s") % fs)
x = ['healthy', 'melanoma']

x_1 = len(train_df.query('sex == "male" and target == 0'))

x_2 = len(train_df.query('sex == "male" and target == 1'))

x_3 = len(train_df.query('sex == "female" and target == 0'))

x_4 = len(train_df.query('sex == "female" and target == 1'))



x_1, x_3 = x_1/sum([x_1, x_3]), x_3/sum([x_1, x_3])

x_2, x_4 = x_2/sum([x_2, x_4]), x_4/sum([x_2, x_4])

fig = go.Figure(data=[go.Bar(name='male', y=x, x=[x_1, x_2], orientation="h",

                             marker=dict(color='#02cc45', line=dict(width=0.5, color='black'))),

                      go.Bar(name='female', y=x, x=[x_3, x_4], orientation="h",

                             marker=dict(color='#f7766f', line=dict(width=0.5, color='black')))])



fig.update_layout(xaxis_title='Target', yaxis_title='Proportion', template="plotly_white",

                  title='Target vs. Proportion (per sex)', barmode='stack', paper_bgcolor="#edebeb"); fig.show()
x = ['healthy', 'melanoma']

x_1 = len(train_df.query('benign_malignant == "benign" and target == 0'))

x_2 = len(train_df.query('benign_malignant == "benign" and target == 1'))

x_3 = len(train_df.query('benign_malignant == "malignant" and target == 0'))

x_4 = len(train_df.query('benign_malignant == "malignant" and target == 1'))



x_1, x_3 = x_1/sum([x_1, x_3]), x_3/sum([x_1, x_3])

x_2, x_4 = x_2/sum([x_2, x_4]), x_4/sum([x_2, x_4])

fig = go.Figure(data=[go.Bar(name='benign', y=x, x=[x_1, x_2], orientation="h",

                             marker=dict(color='#02cc45', line=dict(width=0.5, color='black'))),

                      go.Bar(name='malignant', y=x, x=[x_3, x_4], orientation="h",

                             marker=dict(color='#f7766f', line=dict(width=0.5, color='black')))])



fig.update_layout(xaxis_title='Target', yaxis_title='Proportion', template="plotly_white",

                  title='Target vs. Proportion (per benign_malignant)', barmode='stack', paper_bgcolor="#edebeb"); fig.show()
x = ['healthy', 'melanoma']



x_2 = len(train_df.query('anatom_site_general_challenge == "torso" and target == 0'))

x_3 = len(train_df.query('anatom_site_general_challenge == "head/neck" and target == 0'))

x_4 = len(train_df.query('anatom_site_general_challenge == "palms/soles" and target == 0'))

x_5 = len(train_df.query('anatom_site_general_challenge == "oral/genital" and target == 0'))

x_6 = len(train_df.query('anatom_site_general_challenge == "upper extremity" and target == 0'))

x_7 = len(train_df.query('anatom_site_general_challenge == "lower extremity" and target == 0'))

x_1 = len(train_df.query('anatom_site_general_challenge != anatom_site_general_challenge and target == 0'))



x_9 = len(train_df.query('anatom_site_general_challenge == "torso" and target == 1'))

x_10 = len(train_df.query('anatom_site_general_challenge == "head/neck" and target == 1'))

x_11 = len(train_df.query('anatom_site_general_challenge == "palms/soles" and target == 1'))

x_12 = len(train_df.query('anatom_site_general_challenge == "oral/genital" and target == 1'))

x_13 = len(train_df.query('anatom_site_general_challenge == "upper extremity" and target == 1'))

x_14 = len(train_df.query('anatom_site_general_challenge == "lower extremity" and target == 1'))

x_8 = len(train_df.query('anatom_site_general_challenge != anatom_site_general_challenge and target == 1'))



total = sum([x_1, x_2, x_3, x_4, x_5, x_6, x_7])

x_1, x_2, x_3, x_4, x_5, x_6, x_7 = [y/total for y in [x_1, x_2, x_3, x_4, x_5, x_6, x_7]]



total = sum([x_8, x_9, x_10, x_11, x_12, x_13, x_14])

x_8, x_9, x_10, x_11, x_12, x_13, x_14 = [y/total for y in [x_8, x_9, x_10, x_11, x_12, x_13, x_14]]





fig = go.Figure(data=[go.Bar(name='nan', y=x, x=[x_1, x_8], orientation="h",

                             marker=dict(color='red', line=dict(width=0.5, color='black'))),

                      go.Bar(name='torso', y=x, x=[x_2, x_9], orientation="h",

                             marker=dict(color='orange', line=dict(width=0.5, color='black'))),

                      go.Bar(name='head/neck', y=x, x=[x_3, x_10], orientation="h",

                             marker=dict(color='gold', line=dict(width=0.5, color='black'))),

                      go.Bar(name='palms/soles', y=x, x=[x_4, x_11], orientation="h",

                             marker=dict(color='green', line=dict(width=0.5, color='black'))),

                      go.Bar(name='oral/genital', y=x, x=[x_5, x_12], orientation="h",

                             marker=dict(color='blue', line=dict(width=0.5, color='black'))),

                      go.Bar(name='upper extremity', y=x, x=[x_6, x_13], orientation="h",

                             marker=dict(color='violet', line=dict(width=0.5, color='black'))),

                      go.Bar(name='lower extremity', y=x, x=[x_7, x_14], orientation="h",

                             marker=dict(color='indigo', line=dict(width=0.5, color='black')))])



fig.update_layout(xaxis_title='Target', yaxis_title='Proportion', template="plotly_white",

                  title='Target vs. Proportion (per anatom_site_general_challenge)', barmode='stack', paper_bgcolor="#edebeb"); fig.show()
x = ['healthy', 'melanoma']



x_1 = len(train_df.query('diagnosis == "nevus" and target == 0'))

x_2 = len(train_df.query('diagnosis == "unknown" and target == 0'))

x_3 = len(train_df.query('diagnosis == "melanoma" and target == 0'))

x_4 = len(train_df.query('diagnosis == "lentigo NOS" and target == 0'))

x_5 = len(train_df.query('diagnosis == "solar lentigo" and target == 0'))

x_6 = len(train_df.query('diagnosis == "lichenoid keratosis" and target == 0'))

x_7 = len(train_df.query('diagnosis == "cafe-au-lait macule" and target == 0'))

x_8 = len(train_df.query('diagnosis == "seborrheic keratosis" and target == 0'))

x_9 = len(train_df.query('diagnosis == "atypical melanocytic proliferation" and target == 0'))



x_10 = len(train_df.query('diagnosis == "nevus" and target == 1'))

x_11 = len(train_df.query('diagnosis == "unknown" and target == 1'))

x_12 = len(train_df.query('diagnosis == "melanoma" and target == 1'))

x_13 = len(train_df.query('diagnosis == "lentigo NOS" and target == 1'))

x_14 = len(train_df.query('diagnosis == "solar lentigo" and target == 1'))

x_15 = len(train_df.query('diagnosis == "lichenoid keratosis" and target == 1'))

x_16 = len(train_df.query('diagnosis == "cafe-au-lait macule" and target == 1'))

x_17 = len(train_df.query('diagnosis == "seborrheic keratosis" and target == 1'))

x_18 = len(train_df.query('diagnosis == "atypical melanocytic proliferation" and target == 1'))



total = sum([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9])

x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9 = [y/total for y in [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]]



total = sum([x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18])

x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18 = [y/total for y in [x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18]]





fig = go.Figure(data=[go.Bar(name='nevus', y=x, x=[x_1, x_10], orientation="h",

                             marker=dict(color='red', line=dict(width=0.5, color='black'))),

                      go.Bar(name='unknown', y=x, x=[x_2, x_11], orientation="h",

                             marker=dict(color='orange', line=dict(width=0.5, color='black'))),

                      go.Bar(name='melanoma', y=x, x=[x_3, x_12], orientation="h",

                             marker=dict(color='gold', line=dict(width=0.5, color='black'))),

                      go.Bar(name='lentigo NOS', y=x, x=[x_4, x_13], orientation="h",

                             marker=dict(color='green', line=dict(width=0.5, color='black'))),

                      go.Bar(name='solar lentigo', y=x, x=[x_5, x_14], orientation="h",

                             marker=dict(color='blue', line=dict(width=0.5, color='black'))),

                      go.Bar(name='lichenoid keratosis', y=x, x=[x_6, x_15], orientation="h",

                             marker=dict(color='violet', line=dict(width=0.5, color='black'))),

                      go.Bar(name='cafe-au-lait macule', y=x, x=[x_7, x_16], orientation="h",

                             marker=dict(color='indigo', line=dict(width=0.5, color='black'))),

                      go.Bar(name='seborrheic keratosis', y=x, x=[x_8, x_17], orientation="h",

                             marker=dict(color='goldenrod', line=dict(width=0.5, color='black'))),

                      go.Bar(name='atypical melanocytic proliferation', y=x, x=[x_9, x_18], orientation="h",

                             marker=dict(color='silver', line=dict(width=0.5, color='black')))])



fig.update_layout(xaxis_title='Target', yaxis_title='Proportion', template="plotly_white",

                  title='Target vs. Proportion (per anatom_site_general_challenge)', barmode='stack', paper_bgcolor="#edebeb"); fig.show()
nums_1 = train_df.query("target == 1")["age_approx"]

nums_2 = train_df.query("target == 0")["age_approx"]



nums_1 = nums_1.fillna(nums_1.mean())

nums_2 = nums_2.fillna(nums_2.mean())



fig = ff.create_distplot(hist_data=[nums_1, nums_2],

                         group_labels=["0", "1"],

                         colors=["darkorange", "dodgerblue"], show_hist=False)



fig.update_layout(title_text="Approximate age vs. Target", xaxis_title="Approximate age",

                  template="plotly_white", paper_bgcolor="#edebeb")

fig.show()
def display_images(num):

    sq_num = np.sqrt(num)

    assert sq_num == int(sq_num)



    sq_num = int(sq_num)

    image_ids = os.listdir(IMG_PATHS[0])

    fig, ax = plt.subplots(nrows=sq_num, ncols=sq_num, figsize=(20, 20))



    for i in range(sq_num):

        for j in range(sq_num):

            idx = i*sq_num + j

            ax[i, j].axis('off')

            img = cv2.imread(IMG_PATHS[0] + '/' + image_ids[idx])

            ax[i, j].imshow(img); ax[i, j].set_title('Test Image {}'.format(idx), fontsize=12)



    plt.show()
display_images(36)
def to_tensor(data):

    return [FloatTensor(point) for point in data]



def set_image_transformations(dataset, aug):

    norm = Normalize(mean=MU, std=SIGMA, p=1)

    vflip, hflip = VerticalFlip(p=0.5), HorizontalFlip(p=0.5)

    dataset.transformation = Compose([norm, vflip, hflip]) if aug else norm



class SIIMDataset(Dataset):

    def __init__(self, df, aug, targ, ids):

        set_image_transformations(self, aug)

        self.df, self.targ, self.aug, self.image_ids = df, targ, aug, ids



    def __len__(self):

        return len(self.image_ids)



    def __getitem__(self, i):

        image_id = self.image_ids[i]

        target = [self.df.target[i]] if self.targ else 0

        image = cv2.imread(PATH_DICT[image_id] + image_id)

        return to_tensor([self.transformation(image=image)['image'], target])
def GlobalAveragePooling(x):

    return x.mean(axis=-1).mean(axis=-1)



class CancerNet(nn.Module):

    def __init__(self, features):

        super(CancerNet, self).__init__()

        self.avgpool = GlobalAveragePooling

        self.dense_output = nn.Linear(features, 1)

        self.efn = EfficientNet.from_pretrained(MODEL)

        

    def forward(self, x):

        x = x.view(-1, 3, H, W)

        x = self.efn.extract_features(x)

        return self.dense_output(self.avgpool(x))
def bce(y_true, y_pred):

    return nn.BCEWithLogitsLoss()(y_pred, y_true)



def acc(y_true, y_pred):

    y_true = y_true.squeeze()

    y_pred = nn.Sigmoid()(y_pred).squeeze()

    return (y_true == torch.round(y_pred)).float().sum()/len(y_true)
def print_metric(data, batch, epoch, start, end, metric, typ):

    t = typ, metric, "%s", data, "%s"

    if typ == "Train": pre = "BATCH %s" + str(batch-1) + "%s  "

    if typ == "Val": pre = "\nEPOCH %s" + str(epoch+1) + "%s  "

    time = np.round(end - start, 1); time = "Time: %s{}%s s".format(time)

    fonts = [(fg(211), attr('reset')), (fg(212), attr('reset')), (fg(213), attr('reset'))]

    print(pre % fonts[0] + "{} {}: {}{}{}".format(*t) % fonts[1] + "  " + time % fonts[2])
split = int(SPLIT*len(train_df))

train_df, val_df = train_df.loc[:split], train_df.loc[split:]

train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
C = np.array([B, (1 - B)])*2

ones = len(train_df.query('target == 1'))

zeros = len(train_df.query('target == 0'))



weightage_fn = {0: C[1]/zeros, 1: C[0]/ones}

weights = [weightage_fn[target] for target in train_df.target]
length = len(train_df)

val_ids = val_df.image_name.apply(lambda x: x + '.jpg')

train_ids = train_df.image_name.apply(lambda x: x + '.jpg')



val_set = SIIMDataset(val_df, False, True, ids=val_ids)

train_set = SIIMDataset(train_df, True, True, ids=train_ids)
train_sampler = WeightedRandomSampler(weights, length)

if_sample, if_shuffle = (train_sampler, False), (None, True)

sample_fn = lambda is_sample, sampler: if_sample if is_sample else if_shuffle



sampler, shuffler = sample_fn(SAMPLE, train_sampler)

val_loader = DataLoader(val_set, VAL_BATCH_SIZE, shuffle=False)

train_loader = DataLoader(train_set, BATCH_SIZE, sampler=sampler, shuffle=shuffler)
device = xm.xla_device()

network = CancerNet(features=1536).to(device)

optimizer = Adam([{'params': network.efn.parameters(), 'lr': LR[0]},

                  {'params': network.dense_output.parameters(), 'lr': LR[1]}])
print("STARTING TRAINING ...\n")



start = time.time()

train_batches = len(train_loader) - 1



for epoch in range(EPOCHS):

    fonts = (fg(48), attr('reset'))

    print(("EPOCH %s" + str(epoch+1) + "%s") % fonts)

    

    batch = 1

    network.train()

    for train_batch in train_loader:

        train_img, train_targ = train_batch

        train_targ = train_targ.view(-1, 1)

        train_img, train_targ = train_img.to(device), train_targ.to(device)

        

        if batch >= train_batches: break

        train_preds = network.forward(train_img)

        train_acc = acc(train_targ, train_preds)

        train_loss = bce(train_targ, train_preds)

            

        optimizer.zero_grad()

        train_loss.backward()

        xm.optimizer_step(optimizer, barrier=True)

            

        end = time.time()

        batch = batch + 1

        log = batch % 10 == 1

        accuracy = np.round(train_acc.item(), 3)

        if log: print_metric(accuracy, batch, 0, start, end, "acc", "Train")

            

    network.eval()

    val_loss, val_acc, val_points = 0, 0, 0

        

    with torch.no_grad():

        for val_batch in tqdm(val_loader):

            val_img, val_targ = val_batch

            val_targ = val_targ.view(-1, 1)

            val_img, val_targ = val_img.to(device), val_targ.to(device)



            val_points += len(val_targ)

            val_preds = network.forward(val_img)

            val_acc += acc(val_targ, val_preds).item()*len(val_preds)

            val_loss += bce(val_targ, val_preds).item()*len(val_preds)

        

    end = time.time()

    val_acc /= val_points

    val_loss /= val_points

    accuracy = np.round(val_acc, 3)

    print_metric(accuracy, 0, epoch, start, end, "acc", "Val")

    

    print("")



print("ENDING TRAINING ...")
test_ids = test_df.image_name.apply(lambda x: x + '.jpg')
def sigmoid(x):

    return 1/(1 + np.exp(-x))



test_set = SIIMDataset(test_df, False, False, test_ids)

test_loader = tqdm(DataLoader(test_set, VAL_BATCH_SIZE, shuffle=False))



network.eval()

test_preds = []

with torch.no_grad():

    for test_batch in test_loader:

        test_img, _ = test_batch

        test_img = test_img.to(device)

        test_preds.extend(network.forward(test_img).squeeze().detach().cpu().numpy())
def display_preds(num):

    sq_num = np.sqrt(num)

    assert sq_num == int(sq_num)



    sq_num = int(sq_num)

    image_ids = os.listdir(IMG_PATHS[0])

    few_preds = sigmoid(np.array(test_preds[:num]))

    pred_dict = {0: '"No Melanoma"', 1: '"Melanoma"'}

    fig, ax = plt.subplots(nrows=sq_num, ncols=sq_num, figsize=(20, 20))

    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1)



    for i in range(sq_num):

        for j in range(sq_num):

            idx = i*sq_num + j

            ax[i, j].axis('off')

            pred = few_preds[idx]

            img = cv2.imread(IMG_PATHS[0] + '/' + image_ids[idx])

            ax[i, j].imshow(img); ax[i, j].set_title('Prediction: {}'.format(pred_dict[round(pred.item())]), fontsize=12)



    plt.show()
display_preds(16)
path = '../input/siim-isic-melanoma-classification/'

sample_submission = pd.read_csv(path + 'sample_submission.csv')
sample_submission.target = sigmoid(np.array(test_preds))
sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)