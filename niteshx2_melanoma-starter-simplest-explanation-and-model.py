

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import json

import math

import cv2

import PIL

from PIL import Image



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

from sklearn.decomposition import PCA

import os

import imagesize



%matplotlib inline
import os

print(os.listdir("../input/siimisic-melanoma-resized-images/"))
#Loading Train and Test Data

train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")

print("{} images in train set.".format(train.shape[0]))

print("{} images in test set.".format(test.shape[0]))
train.head()
grp = train.groupby(['anatom_site_general_challenge']).mean().sort_values(by = 'target')

grp
%matplotlib inline
import seaborn as sns

sns.set(style="darkgrid")

# titanic = sns.load_dataset("titanic")

ax = sns.barplot(x="target", y=grp.index, data=grp)

# plt.bar( grp.index , grp['target'])
test.head()
import seaborn as sns

sns.set(style="darkgrid")

# titanic = sns.load_dataset("titanic")

ax = sns.countplot(x="target", data=train)
np.mean(train.target)
plt.figure(figsize=(12, 5))

plt.hist(train['age_approx'].values, bins=200)

plt.title('Histogram age_approx counts in train')

plt.xlabel('Value')

plt.ylabel('Count')

plt.show()
images = []

for i, image_id in enumerate(tqdm(train['image_name'].head(10))):

    im = Image.open(f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg')

    im = im.resize((128, )*2, resample=Image.LANCZOS)

    images.append(im)

    
images[0]
images[1]
images[3]
plt.figure(figsize=(12, 5))

plt.hist(test['age_approx'].values, bins=200)

plt.title('Histogram age_approx counts in test')

plt.xlabel('Value')

plt.ylabel('Count')

plt.show()
x_train_32 = np.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')

x_test_32 = np.load('../input/siimisic-melanoma-resized-images/x_test_32.npy')
x_train_32.shape
x_train_32 = x_train_32.reshape((x_train_32.shape[0], 32*32*3))

x_train_32.shape
x_test_32 = x_test_32.reshape((x_test_32.shape[0], 32*32*3))

x_test_32.shape
y = train.target.values
train_oof = np.zeros((x_train_32.shape[0], ))

test_preds = 0

train_oof.shape
x_train_32
n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):

    print("Fitting fold", jj+1)

    train_features = x_train_32[train_index]

    train_target = y[train_index]

    

    val_features = x_train_32[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial', max_iter=60)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof[val_index] = val_pred

    print(len(train_oof))

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds += model.predict_proba(x_test_32)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()
print(roc_auc_score(y, train_oof))
train['age_approx'].unique()
train['sex'] = (train['sex'].values == 'male')*1

test['sex'] = (test['sex'].values == 'male')*1

train.head()
test.head()
train['sex'].mean()
test['sex'].mean()
train['age_approx'].mean()
test['age_approx'].mean()
train['age_approx'] = train['age_approx'].fillna(train['age_approx'].mean())

test['age_approx'] = test['age_approx'].fillna(test['age_approx'].mean())
x_train_32 = np.hstack([x_train_32, train['sex'].values.reshape(-1,1), train['age_approx'].values.reshape(-1,1)])

x_test_32 = np.hstack([x_test_32, test['sex'].values.reshape(-1,1), test['age_approx'].values.reshape(-1,1)])
x_train_32[0].shape
train['anatom_site_general_challenge'].unique()



test['anatom_site_general_challenge'].unique()



train['anatom_site_general_challenge'].mode()



test['anatom_site_general_challenge'].mode()



train['anatom_site_general_challenge'].fillna(train['anatom_site_general_challenge'].mode(), inplace=True)

test['anatom_site_general_challenge'].fillna(test['anatom_site_general_challenge'].mode(), inplace=True)



train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype(str)

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype(str)



# test['anatom_site_general_challenge'].isnull().sum()
train.isna().sum()
x_train_32.shape
x_train_32 = np.hstack([x_train_32, pd.get_dummies(train['anatom_site_general_challenge']).values])

x_test_32 = np.hstack([x_test_32, pd.get_dummies(test['anatom_site_general_challenge']).values])
train.head()
x_train_32 = np.hstack([x_train_32, train[['sex','age_approx']].values])

x_test_32 = np.hstack([x_test_32, test[['sex','age_approx']].values])
x_train_32.shape
# x_train_32[: , -6:]
train_oof_3 = np.zeros((x_train_32.shape[0], ))

test_preds_3 = 0





n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):

    print("Fitting fold", jj+1)

    train_features = x_train_32[train_index]

    train_target = y[train_index]

    

    val_features = x_train_32[val_index]

    val_target = y[val_index]

    

    model = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial', max_iter=60)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)[:,1]

    train_oof_3[val_index] = val_pred

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds_3 += model.predict_proba(x_test_32)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()
ans = pd.DataFrame(train_oof_3)
print(roc_auc_score(y, train_oof_3))
sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sample_submission.head()
sample_submission['target'] = ans

sample_submission.to_csv('submission_3.csv', index=False)