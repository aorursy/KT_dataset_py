# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null # no output
import numpy as np

import pandas as pd 

import os

import torch

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models

from tqdm import tqdm_notebook as tqdm

import math

import torch.nn.functional as F

from torch.nn import init

import gc

import cv2

import pretrainedmodels

import torchvision
HEIGHT = 137

WIDTH = 236

TRAIN = False
def prepare_image(dataType = 'train', indices = [0,1,2,3]):

    assert dataType in ['train', 'test']

    HEIGHT = 137

    WIDTH = 236

    images = []

    for i in indices:

        image_df = pd.read_parquet(f'../input/bengaliai-cv19/{dataType}_image_data_{i}.parquet') 

        images.append(image_df.iloc[:,1:].values.reshape(-1,HEIGHT,WIDTH))

        del image_df

        gc.collect()

    

    

    images = np.concatenate(images, axis = 0)

    return images
if TRAIN:

    train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

    train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    train_images = prepare_image()
def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax
def crop_resize(img0, size=128, pad=16):

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img,(size,size))

    #return img
class BengaliAIDataset(Dataset):

    def __init__(self, images, labels=None, indices=None):

        super(BengaliAIDataset, self).__init__()

        self.images = images

        self.labels = labels

        if indices is None:

            indices = np.arange(len(images))

        self.indices = indices

        self.train = labels is not None



    def __len__(self):

        """return length of this dataset"""

        return len(self.indices)



    def __getitem__(self, i):

        """Return i-th data"""

        i = self.indices[i]

        x = self.images[i]

        # Opposite white and black: background will be white and

        # for future Affine transformation

        x = (255 - x).astype(np.float32) 

        x = (x*(255.0/x.max())).astype(np.float32)

        x = crop_resize(x,224)

        x = np.stack((x,)*3,axis=-1)

        x = x * [0.229, 0.224, 0.225]

        x = np.rollaxis(x, 2, 0)

        x = x.astype(np.float32)/255.0

        if self.train:

            y = self.labels[i]

            return x, y

        else:

            return x
def MyModel():

    model = pretrainedmodels.__dict__['densenet161'](pretrained=None)

    model.last_linear = nn.Linear(model.last_linear.in_features, 186)

    return model
device = torch.device('cuda:0')
model = MyModel().to(device)
if TRAIN:

    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.3)

    criterion = nn.CrossEntropyLoss()

    batch_size=32

    num_images = len(train_images)

    train_data_size = int(num_images*0.9)

    test_data_size = num_images - train_data_size

    perm = np.random.RandomState(111).permutation(num_images)

    train_dataset = BengaliAIDataset(   train_images, train_labels, indices=perm[:train_data_size]   )

    valid_dataset = BengaliAIDataset(   train_images, train_labels, indices=perm[train_data_size:train_data_size+test_data_size]   )

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=True)

    epochs = 10

    losses=[]

    accs=[]

    train_root_accs = []

    val_root_accs = []

    train_vowel_accs = []

    val_vowel_accs = []

    train_con_accs = []

    val_con_accs = []

    train_accs = []

    val_accs = []
if not TRAIN:

    load_model_path = '/kaggle/input/dense161/seresnet50 (3).pth'

    checkpoint = torch.load(load_model_path)

    model.load_state_dict(checkpoint['model_state_dict'])
import numpy
if TRAIN:

    for epoch in range(epochs):



        gc.collect()

        print('epochs {}/{} '.format(epoch+1,epochs))

        running_loss = 0.0

        root_acc = 0.0

        vowel_acc = 0.0

        con_acc = 0.0

        model.train()

        for idx, (inputs,labels) in tqdm(enumerate(train_loader),total=len(train_loader)):



            inputs = inputs.to(device)

            labels = labels.to(device)

            output = model(inputs.float())

            outputs = torch.split(output, [168, 11, 7], dim=1)

            loss1 = criterion(outputs[0],labels[:,0])

            loss2 = criterion(outputs[1],labels[:,1])

            loss3 = criterion(outputs[2],labels[:,2])    

            running_loss += (2*loss1 + loss2 + loss3)

            root_acc += (outputs[0].argmax(1)==labels[:,0]).float().mean()

            vowel_acc += (outputs[1].argmax(1)==labels[:,1]).float().mean()

            con_acc += (outputs[2].argmax(1)==labels[:,2]).float().mean()

            (2*loss1 + loss2 + loss3).backward()

            optimizer.step()

            del inputs

            del labels



        losses.append(running_loss/len(train_loader))

        train_root_acc = root_acc/(len(train_loader))

        train_vowel_acc = vowel_acc/(len(train_loader))

        train_con_acc = con_acc/(len(train_loader))

        train_root_accs.append(train_root_acc)

        train_vowel_accs.append(train_vowel_acc)

        train_con_accs.append(train_con_acc)

        act_train = (2*train_root_acc+train_vowel_acc+train_con_acc)/4.0

        accs.append(act_train)



        print('acc : {:.4f}'.format(act_train))

        print('root_acc : {:.4f}'.format(train_root_acc))

        print('vowel_acc : {:.4f}'.format(train_vowel_acc))

        print('con acc : {:.4f}'.format(train_con_acc))

        print('loss : {:.4f}'.format(running_loss/len(train_loader)))









        val_loss = 0.0

        root_acc = 0.0

        vowel_acc = 0.0

        con_acc = 0.0

        model.eval()

        with torch.no_grad():

            for idx, (inputs,labels) in tqdm(enumerate(valid_loader),total=len(valid_loader)):

                inputs = inputs.to(device)

                labels = labels.to(device)

                #print(inputs.shape)

                #print(labels.shape)

                output = model(inputs.float())

                outputs = torch.split(output, [168, 11, 7], dim=1)

                loss11 = criterion(outputs[0],labels[:,0])

                loss21 = criterion(outputs[1],labels[:,1])

                loss31 = criterion(outputs[2],labels[:,2])

                val_loss += (2*loss11 + loss21 + loss31)

                root_acc += (outputs[0].argmax(1)==labels[:,0]).float().mean()

                vowel_acc += (outputs[1].argmax(1)==labels[:,1]).float().mean()

                con_acc += (outputs[2].argmax(1)==labels[:,2]).float().mean()

                del inputs

                del labels

                gc.collect()



        val_root_acc = root_acc/(len(valid_loader))

        val_vowel_acc = vowel_acc/(len(valid_loader))

        val_con_acc = con_acc/(len(valid_loader))

        val_root_accs.append(val_root_acc)

        val_vowel_accs.append(val_vowel_acc)

        val_con_accs.append(val_con_acc)

        act_val = (2*val_root_acc+val_vowel_acc+val_con_acc)/4.0

        val_loss = val_loss/3.0

        scheduler.step(val_loss)



        print('val_acc : {:.4f}'.format(act_val))

        print('root_acc : {:.4f}'.format(val_root_acc))

        print('vowel_acc : {:.4f}'.format(val_vowel_acc))

        print('con acc : {:.4f}'.format(val_con_acc))

        print('loss : {:.4f}'.format(val_loss/len(valid_loader)))



        file_name = str(act_train)+'_'+str(act_val)+'_.tar'

        torch.save({

            'model_state_dict':model.state_dict(),

            'optimizer_state_dict':optimizer.state_dict(),



        },'weight.pth')

        print()

        print()
data_type = 'test'

test_preds_list = []

for i in range(4):

    gc.collect()

    indices = [i]

    test_images = prepare_image(data_type,indices=indices)

    n_dataset = len(test_images)

    print(f'i={i}, n_dataset={n_dataset}')

    # test_data_size = 200 if debug else int(n_dataset * 0.9)

    test_dataset = BengaliAIDataset(test_images )

    print('test_dataset', len(test_dataset))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    for  idx, inputs in enumerate(test_loader):

        inputs = inputs.to(device)

        output = model(inputs.float())

        outputs = torch.split(output, [168, 11, 7], dim=1)

        outputs = outputs[0].argmax(1), outputs[1].argmax(1), outputs[2].argmax(1)

        test_preds_list.append(outputs)

    del test_images

    gc.collect()
p0 = np.concatenate([test_preds[0].cpu().detach().numpy() for test_preds in test_preds_list], axis=0)

p1 = np.concatenate([test_preds[1].cpu().detach().numpy() for test_preds in test_preds_list], axis=0)

p2 = np.concatenate([test_preds[2].cpu().detach().numpy() for test_preds in test_preds_list], axis=0)

print('p0', p0.shape, 'p1', p1.shape, 'p2', p2.shape)
row_id = []

target = []

for i in range(len(p0)):

    row_id += [f'Test_{i}_consonant_diacritic', f'Test_{i}_grapheme_root',

               f'Test_{i}_vowel_diacritic']

    target += [p2[i], p0[i], p1[i]]
submission_df = pd.DataFrame({'row_id': row_id, 'target': target})

submission_df.to_csv('submission.csv', index=False)
submission_df