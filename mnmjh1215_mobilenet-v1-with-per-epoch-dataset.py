# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import glob



import ast



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, ConcatDataset



import torchvision

from torchvision import transforms, utils



import matplotlib.pyplot as plt

%matplotlib inline



import cv2  # to generate image from vectors (strokes)



import tqdm



device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
NROWS = 10000

VAL_NROWS = 500



path = '../input/train_simplified/'

filenames = glob.glob(os.path.join(path, '*.csv'))

NUM_CLASSES = len(filenames)
# this drawing function was adopted from https://github.com/ebouteillon/kaggle-quickdraw-doodle-recognition-challenge/blob/master/2-training-resnet18-from-scratch-with-128px-images.ipynb



shift_colors = (

    (255, 0, 0),

    (255, 128, 0),

    (255, 255, 0),

    (128, 255, 0),

    (0, 255, 0),

    (0, 255, 128),

    (0, 255, 255),

    (0, 128, 255),

    (0, 0, 255),

    (128, 0, 255),

    (255, 0, 255),

    (255, 0, 128)

)





def draw_cv2(raw_strokes, size=128, lw=1):

    # draw function inspired from https://towardsdatascience.com/10-lessons-learned-from-participating-to-google-ai-challenge-268b4aa87efa

    BASE_SIZE = 256

    border = 2  # keep some margin with image border



    img = np.zeros((size, size, 3), np.uint8)

    coef = (size - 2 * lw - 2 * border) / (BASE_SIZE - 1)

    num_stokes = len(raw_strokes)

    for t, stroke in enumerate(raw_strokes[::-1]):  # iterate in reverse order, so that earlier strokes, which are more important, are drawn later so that they are not overlapped

        rgb = shift_colors[(num_stokes-t-1)%12]



        for i in range(len(stroke[0]) - 1):

            p1 = (int(coef * stroke[0][i] + lw + border), int(coef * stroke[1][i] + lw+ border))

            p2 = (int(coef * stroke[0][i + 1] + lw + border), int(coef * stroke[1][i + 1] + lw + border))

            _ = cv2.line(img, p1, p2, rgb, lw, cv2.LINE_AA)

    return img
encode_dict = {}

path = '../input/train_simplified/'



filenames = glob.glob(os.path.join(path, '*.csv'))

filenames = sorted(filenames)

print(len(filenames))



for ix, filename in enumerate(filenames):

    class_name = filename.split('/')[-1].split('.')[0].replace(' ', '_')

    encode_dict[class_name] = ix

    

decode_dict = {value:key for key, value in encode_dict.items()}
class ClassDataset(Dataset):

    def __init__(self, csv_file_path, mode='train', nrows=15000, skiprows=0, size=128):

        # try nrows=20000

        super().__init__()

        

        self.df = pd.read_csv(csv_file_path, usecols=['drawing'], nrows=nrows, skiprows=0)

        self.mode = mode

        self.size = size

        if self.mode == 'train':

            self.class_name = csv_file_path.split('/')[-1].split('.')[0].replace(' ', '_')

            self.label = encode_dict[self.class_name]

            

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        raw_strokes = ast.literal_eval(self.df.drawing[index])

        image = draw_cv2(raw_strokes, size=self.size)  # (size, size, 3)

        image = image.transpose(2, 0, 1)

        

        if self.mode == 'train':

            return (image/255).astype('float32'), self.label

        else:

            return (image/255).astype('float32')

    

    
%%time



val_dset = ConcatDataset([ClassDataset(filename, nrows=VAL_NROWS, skiprows=0) for filename in filenames])

dset = ConcatDataset([ClassDataset(filename, nrows=NROWS, skiprows=VAL_NROWS) for filename in filenames])
print(len(dset))  # 340 * NROWS

print(len(val_dset))   # 340 * VAL_NROWS
%%time



dloader = DataLoader(dset, batch_size=128, shuffle=True, num_workers=2)

val_dloader = DataLoader(val_dset, batch_size=128, num_workers=2, shuffle=False)
batch = iter(dloader).next()

print(batch)  # it works well, shuffled.
plt.imshow(np.transpose(batch[0][0], (1, 2, 0)))
# our dataloader is ready.

# let's make model



# building blocks for mobilenet.

# name Conv and Conv_dw are following the terms used in mobilenet paper (https://arxiv.org/pdf/1704.04861.pdf)



class Conv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):

        super().__init__()

        self.layers = nn.Sequential(

                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),

                nn.BatchNorm2d(out_channel),

                nn.ReLU(inplace=True)

            )

        

    def forward(self, input):

        return self.layers(input)

    

class Conv_dw_Conv(nn.Module):

    # Conv dw layer followed by Conv layer.

    # implemented this way since every conv dw layer is followed by conv layer with kernel size 1, stride 1 with some out_channel

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):

        super().__init__()

        self.layers = nn.Sequential(

                nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, bias=False, groups=in_channel),

                nn.BatchNorm2d(in_channel),

                nn.ReLU(inplace=True),

                Conv(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

            )

        

    def forward(self, input):

        return self.layers(input)

    

    

class MobileNet(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        

        self.num_classes = num_classes

        

        self.model = nn.Sequential(

                Conv(3, 32, stride=2),

                Conv_dw_Conv(32, 64, kernel_size=3, stride=1),

                Conv_dw_Conv(64, 128, kernel_size=3, stride=2),

                Conv_dw_Conv(128, 128, kernel_size=3, stride=1),

                Conv_dw_Conv(128, 256, kernel_size=3, stride=2),

                Conv_dw_Conv(256, 256, kernel_size=3, stride=1),

                Conv_dw_Conv(256, 512, kernel_size=3, stride=2),

                Conv_dw_Conv(512, 512, kernel_size=3, stride=1),

                Conv_dw_Conv(512, 512, kernel_size=3, stride=1),

                Conv_dw_Conv(512, 512, kernel_size=3, stride=1),

                Conv_dw_Conv(512, 512, kernel_size=3, stride=1),

                Conv_dw_Conv(512, 512, kernel_size=3, stride=1),

                Conv_dw_Conv(512, 1024, kernel_size=3, stride=2),

                Conv_dw_Conv(1024, 1024, kernel_size=3, stride=1)

        )

        

        

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(1024, num_classes)

        

    def forward(self, input):

        x = self.model(input)

        x = self.avg_pool(x)

        x = x.view(-1, 1024)

        out = self.fc(x)

        return out
# before training, let's set up our metric MAP@3

# I slightly modified implementation of https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py



def apk(actual, predicted, k=3):

    """

    Computes the average precision at k.

    This function computes the average prescision at k between two lists of

    items.

    Parameters

    ----------

    actual : int

             element that are to be predicted

    predicted : list

                A list of predicted elements (order does matter)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The average precision at k over the input lists

    """

    if len(predicted)>k:

        predicted = predicted[:k]



    score = 0.0



    for i,p in enumerate(predicted):

        if p == actual:

            score = 1 / (i+1.0)

    

    return score



def mapk(actual, predicted, k=3):

    """

    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists

    of lists of items.

    Parameters

    ----------

    actual : list

             A list of elements that are to be predicted 

    predicted : list

                A list of lists of predicted elements

                (order matters in the lists)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The mean average precision at k over the input lists

    """

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
# one last thing before start training, we need a function that returns validation map@3.



def validation_score(model, val_data_loader):

    model.eval()

    sum_score = 0

    count = 0

    for images, labels in val_data_loader:

        images = images.to(device)

        labels = labels.to(device)

        batch_size = images.size(0)

        output = model(images)

        topk = output.detach().topk(3, dim=1)[1]

        sum_score += mapk(labels.cpu().numpy(), topk.cpu().numpy()) * batch_size

        count += batch_size

        

    model.train()

    return sum_score / count



    
# train time!



model = MobileNet(NUM_CLASSES).to(device)



criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)



scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60000, 130000, 160000, 190000], gamma=0.5)
%%time



print("Start training...")



epochs = 8



print_every = 1000 # print every N iterations

validate_every = 10000  # do validation every N iterations

model.train()



best_val_score = 0



curr_iter = 0

avg_loss = 0

avg_score = 0



for epoch in range(epochs):

    

    # delete existing dataset and dataloader, and create new dataset and dataloader

    del dset

    del dloader

    

    dset = ConcatDataset([ClassDataset(filename, nrows=NROWS, skiprows=VAL_NROWS + NROWS * epoch) for filename in filenames])  # skip rows that the model has already trained on.

    dloader = DataLoader(dset, batch_size=128, shuffle=True, num_workers=2)



    for ix, (images, labels) in enumerate(dloader):

        

        images = images.to(device)

        labels = labels.to(device)

        

        model.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        

        avg_loss += loss.item()

        

        topk = output.detach().topk(3, dim=1)[1]

        avg_score += mapk(labels.cpu().numpy(), topk.cpu().numpy())

        # scheduler.step()

            

        curr_iter += 1

        if (curr_iter) % print_every == 0:

            print('Epoch {}, Iteration {} - Train Loss: {:.4f}, MAP@3: {:.3f}'.format(epoch + 1, curr_iter, avg_loss/print_every, avg_score/print_every))

            avg_loss = 0

            avg_score = 0

            

        if curr_iter % validate_every == 0:

            val_score = validation_score(model, val_dloader)

            print('Epoch {}, Iteration {}: validation map@3: {}'.format(epoch + 1, curr_iter, val_score))

            if val_score > best_val_score:

                print('New best validation score: {}, saving model...'.format(val_score))

                best_val_score = val_score

                torch.save(model.state_dict(), 'model_checkpoint_best_val.ckpt')

# make subimssion using final model

test_dset = ClassDataset('../input/test_simplified.csv', mode='test', nrows=None)

test_dloader = DataLoader(test_dset, batch_size=128, shuffle=False, num_workers=0)



import tqdm

model.eval()

labels = []

for images in tqdm.tqdm(test_dloader):

    images = images.to(device)

    output = model(images)

    _, pred = output.topk(3, 1)

    for i in range(len(images)):

        labels.append(' '.join([decode_dict[pred[i][j].item()]for j in range(3)]))

len(labels)
%%time

submission = pd.read_csv('../input/test_simplified.csv', index_col='key_id' ,usecols=['key_id'])

print(len(submission))

submission['word'] = labels

submission.to_csv('submission_final.csv')
# make submission using best val checkpoint



model.load_state_dict(torch.load('model_checkpoint_best_val.ckpt'))

model.eval()



labels = []

for images in tqdm.tqdm(test_dloader):

    images = images.to(device)

    output = model(images)

    _, pred = output.topk(3, 1)

    for i in range(len(images)):

        labels.append(' '.join([decode_dict[pred[i][j].item()]for j in range(3)]))

        

print(len(labels))

        

submission = pd.read_csv('../input/test_simplified.csv', index_col='key_id' ,usecols=['key_id'])

print(len(submission))

submission['word'] = labels



submission.to_csv('submission_best_val.csv')