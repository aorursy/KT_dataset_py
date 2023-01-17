# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

import cv2

import glob2

from tqdm import tqdm

from PIL import Image

import matplotlib.pyplot as plt

import lightgbm as lgb

print(os.listdir("../input"))



from sklearn.model_selection import train_test_split,StratifiedKFold

from sklearn.metrics import roc_auc_score

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision.models import resnet50, resnet34, densenet201, densenet121

from torch.utils.data import Dataset, DataLoader

train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
train_df.head()
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
test_df.head()
sns.countplot(x='diagnosis',data=train_df)
len(os.listdir('../input/aptos2019-blindness-detection/train_images'))
len(os.listdir('../input/aptos2019-blindness-detection/test_images'))
train = glob2.glob('../input/aptos2019-blindness-detection/train_images/*.png')

test = glob2.glob('../input/aptos2019-blindness-detection/test_images/*.png')
def read_image(filename):

    img = cv2.imread(str(filename))

    

    x_tot = img.mean() #image statistics

    x_rot2 = img.std()

    return x_tot, x_rot2



def get_stats(stats): # get dataset statistics 

    x_tot, x2_tot = 0.0, 0.0

    for x, x2 in stats:

        x_tot += x

        x2_tot += x2

    

    img_avr =  x_tot/len(stats)

    img_std = x2_tot/len(stats)

    print('mean:',img_avr, ', std:', img_std)
trn_stats = []

for fname in tqdm(train, total=len(train)):

    trn_stats.append(read_image(fname))
test_stats = []        

for fname in tqdm(test, total=len(test)):

    test_stats.append(read_image(fname))
get_stats(trn_stats)

get_stats(test_stats)
IMG_SIZE = 512

BATCH_SIZE = 16


def img_to_torch(image):

    return torch.from_numpy(np.transpose(image, (2, 0, 1)))



def pad_to_square(image):

    h, w = image.shape[0:2]

    new_size = max(h, w)

    delta_top = (new_size-h)//2

    delta_bottom = new_size-h-delta_top

    delta_left = (new_size-w)//2

    delta_right = new_size-delta_left-w

    new_im = cv2.copyMakeBorder(image, delta_top, delta_bottom, delta_left, delta_right, 

                                cv2.BORDER_CONSTANT,  value=[0,0,0])

    return new_im



class AptosDataset(Dataset):

    def __init__(self, df,datatype='train'):

        self.df = df

        self.datatype = datatype

        self.image_files_list = [f'../input/aptos2019-blindness-detection/{self.datatype}_images/{i}.png' for i in df['id_code'].values]

        self.cache = {}

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        if index not in range(0, len(self.df)):

            return self.__getitem__(np.random.randint(0, self.__len__()))

        

        # only take on channel

#         if index not in self.cache:

        image = cv2.imread(self.image_files_list[index])

        image = pad_to_square(image)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

#             self.cache[index] = img_to_torch(image)



        return img_to_torch(image)
train_image = AptosDataset(train_df,datatype='train')

train_image_loader = DataLoader(train_image, batch_size=BATCH_SIZE, shuffle=False, 

                       num_workers=1, pin_memory=True)
class ResnetModel(nn.Module):

    def __init__(self, resnet_fun=resnet50, freeze_basenet = True):

        super(ResnetModel, self).__init__()

        self.resnet = resnet_fun(pretrained=False)

        if freeze_basenet:

            for p in self.resnet.parameters():

                p.requires_grad = False

       

    def init_resnet(self, path):

        state = torch.load(path)

        self.resnet.load_state_dict(state)

        

    def forward(self, x):

        batch_size = x.shape[0]

        x = x/255.0

        mean = [0.485, 0.456, 0.406]

        std = [0.229, 0.224, 0.225]

        x = torch.cat([

            (x[:, [0]] - mean[0]) / std[0],

            (x[:, [1]] - mean[1]) / std[1],

            (x[:, [2]] - mean[2]) / std[2],

        ], 1)

        x = self.resnet.conv1(x)

        x = self.resnet.bn1(x)

        x = self.resnet.relu(x)

        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)

        x = self.resnet.layer2(x)

        x = self.resnet.layer3(x)

        x = self.resnet.layer4(x)

        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)

        return x
resnet50_feature = []

model = ResnetModel()

model.init_resnet('../input/pytorch-pretrained-image-models/resnet50.pth')

model.cuda()

model.eval()

with torch.no_grad():

    for img_batch in tqdm(train_image_loader):

        img_batch = img_batch.float().cuda()

        y_pred = model(img_batch)

        resnet50_feature.append(y_pred.cpu().numpy()) 

resnet50_feature = np.vstack(resnet50_feature)
RES50_IMG_FEATURE_DIM = resnet50_feature.shape[1]
train_df.head()
resnet50_feature_df = pd.DataFrame(resnet50_feature, dtype=np.float32,

                                   columns=['resnet50_%d'%i for i in range(RES50_IMG_FEATURE_DIM)])

resnet50_feature_df['id_code'] = train_df['id_code'].values
resnet50_feature_df_avg = resnet50_feature_df.groupby('id_code').agg('mean').reset_index()

resnet50_feature_df_avg.columns = ['id_code']+['resnet50_mean_%d'%i for i in range(RES50_IMG_FEATURE_DIM)]
resnet50_feature_df_avg.head()
resnet50_feature_train = train_df[['id_code','diagnosis']].merge(resnet50_feature_df_avg, on='id_code', how='left')
resnet50_feature_train.head()
test_image = AptosDataset(test_df,datatype='test')

test_image_loader = DataLoader(test_image, batch_size=BATCH_SIZE, shuffle=False, 

                       num_workers=1, pin_memory=True)
resnet50_feature = []

model = ResnetModel()

model.init_resnet('../input/pytorch-pretrained-image-models/resnet50.pth')

model.cuda()

model.eval()

with torch.no_grad():

    for img_batch in tqdm(test_image_loader):

        img_batch = img_batch.float().cuda()

        y_pred = model(img_batch)

        resnet50_feature.append(y_pred.cpu().numpy()) 

resnet50_feature = np.vstack(resnet50_feature)
RES50_IMG_FEATURE_DIM = resnet50_feature.shape[1]
resnet50_feature_df = pd.DataFrame(resnet50_feature, dtype=np.float32,

                                   columns=['resnet50_%d'%i for i in range(RES50_IMG_FEATURE_DIM)])

resnet50_feature_df['id_code'] = test_df['id_code'].values

#resnet50_feature_df['PicID'] = image_df['PicID'].values

resnet50_feature_df_avg = resnet50_feature_df.groupby('id_code').agg('mean').reset_index()

resnet50_feature_df_avg.columns = ['id_code']+['resnet50_mean_%d'%i for i in range(RES50_IMG_FEATURE_DIM)]
resnet50_feature_test = test_df[['id_code']].merge(resnet50_feature_df_avg, on='id_code', how='left')
resnet50_feature_test.head()
lgb_params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'verbose': 1,

    'learning_rate': 0.05,

    'num_leaves': 31,

    'feature_fraction': 0.7,

    'min_data_in_leaf': 200,

    'bagging_fraction': 0.8,

    'bagging_freq': 20,

    'min_hessian': 0.01,

    'feature_fraction_seed': 2,

    'bagging_seed': 3,

    "seed": 1234

}
features = [c for c in resnet50_feature_train.columns if c not in ['id_code', 'diagnosis']]



len_train = len(resnet50_feature_train)

resnet50_feature_train['target'] = 1

resnet50_feature_train = resnet50_feature_train.append(resnet50_feature_test).reset_index(drop = True)

resnet50_feature_train['target'] = resnet50_feature_train['target'].fillna(0)
resnet50_feature_train.head()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

oof = resnet50_feature_train[['id_code', 'target']]

oof['predict'] = 0

val_aucs = []
for fold, (trn_idx, val_idx) in enumerate(skf.split(resnet50_feature_train, resnet50_feature_train['target'])):

    X_train, y_train = resnet50_feature_train.iloc[trn_idx][features], resnet50_feature_train.iloc[trn_idx]['target']

    X_valid, y_valid = resnet50_feature_train.iloc[val_idx][features], resnet50_feature_train.iloc[val_idx]['target']

    trn_data = lgb.Dataset(X_train, label=y_train)

    val_data = lgb.Dataset(X_valid, label=y_valid)

    evals_result = {}

    lgb_clf = lgb.train(lgb_params,

                        trn_data,

                        7500,

                        valid_sets=[val_data],

                        early_stopping_rounds=100,

                        verbose_eval=50,

                        evals_result=evals_result)



    p_valid = lgb_clf.predict(X_valid[features], num_iteration=lgb_clf.best_iteration)



    oof['predict'][val_idx] = p_valid

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)
mean_auc = np.mean(val_aucs)

std_auc = np.std(val_aucs)

all_auc = roc_auc_score(oof['target'], oof['predict'])

print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))