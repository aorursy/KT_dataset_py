# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import os

import random

import math

import gc

import glob



from sklearn.model_selection import KFold, GroupKFold, train_test_split

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error





import operator

import typing as tp

from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

from functools import partial



import matplotlib.pyplot as plt

import seaborn as sns



from tqdm.notebook import tqdm



import pydicom



print(os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/'))



from time import time, strftime, gmtime



start = time()

#print(start)



import datetime

print(str(datetime.datetime.now()))
seed = 2019

seed_everything(seed)
input_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/'
train = pd.read_csv(input_path + 'train.csv')

train
test = pd.read_csv(input_path + 'test.csv')

test
sample = pd.read_csv(input_path + 'sample_submission.csv')

sample
print('Number of patients in the train set: ', train['Patient'].nunique())
plt.figure(figsize = (8, 8))



plt.title('Age')

sns.distplot(train['Age'])
plt.figure(figsize = (8, 8))



plt.title('Weeks')

sns.distplot(train['Weeks'])
plt.figure(figsize = (8, 8))



plt.title('FVC')

sns.distplot(train['FVC'], bins = int(1 + math.log2(train['Patient'].nunique())))

#use Sturgess's formula to find the appropriate number of classes in the histogram
print('Max. FVC: ', train['FVC'].max())

print('Min. FVC: ', train['FVC'].min())

print('Mean FVC: ', train['FVC'].mean())
train['SmokingStatus'].value_counts()
fig, ax = plt.subplots(1, 2, figsize = (15, 15))

#plt.figure(figsize = (10, 10))

plt.suptitle('Smoking Status')

sns.countplot(train['SmokingStatus'], ax = ax[0])



lbls, freqs = np.unique(train['SmokingStatus'].values, return_counts = True)



ax[1].pie(freqs, labels = lbls, autopct = '%1.1f%%', shadow = False, startangle = 90)

plt.show()
fig, ax = plt.subplots(1, 2, figsize = (15, 15))



plt.suptitle('Sex')

sns.countplot(train['Sex'], ax = ax[0])



lbls, freqs = np.unique(train['Sex'].values, return_counts = True)



ax[1].pie(freqs, labels = lbls, autopct = '%1.1f%%', shadow = False, startangle = 90)

plt.show()
plt.figure(figsize = (8, 8))

plt.title('Smoking Status by Sex')

sns.countplot(train['SmokingStatus'], hue = train['Sex'])
plt.figure(figsize = (8, 8))

plt.title('Smoking Status by Sex')

sns.countplot(train['Age'], hue = train['SmokingStatus'])
train.isnull().sum()
plt.figure(figsize = (8, 8))

sns.heatmap(train.corr(), annot = True)
print('Correlation coeff between Age and FVC is: ', train.corr()['Age']['FVC'])
plt.figure(figsize = (8, 8))

sns.scatterplot(data = train, x = 'Age', y = 'FVC')
#Corr for smokers

train_cs = train.loc[train['SmokingStatus'] == 'Currently smokes']



plt.figure(figsize = (8, 8))

sns.scatterplot(data = train_cs, x = 'Age', y = 'FVC')



print('Correlation coeff between Age and FVC (Current Smokers) is: ', train_cs.corr()['Age']['FVC'])
train_dcm = glob.glob(input_path + 'train/*/*')

test_dcm = glob.glob(input_path + 'test/*/*')



print('Num of train dicom: ', len(train_dcm))

print('Num of test dicom: ', len(test_dcm))
num_imgs_pid = [len(os.listdir(input_path + 'train/' + path)) for path in os.listdir(input_path + 'train/')]

plt.figure(figsize = (8, 8))

plt.hist(num_imgs_pid)

plt.ylabel('Number of patients')

plt.xlabel('DICOM files')

plt.title('DICOM Images per patient')

plt.show()

print('Max. number of dicom images per patient: ', max(num_imgs_pid))

print('Min. number of dicom images per patient: ', min(num_imgs_pid))

print('Mean. number of dicom images per patient: ', np.mean(num_imgs_pid))
pydicom.dcmread(train_dcm[10])
random_dcm_train = np.random.choice(train_dcm, 6)

fig, ax = plt.subplots(2, 3, figsize = (15, 10))



ax = ax.ravel()



for i, file in enumerate(random_dcm_train):

    img = pydicom.dcmread(file)

    img = img.pixel_array

    # Since the scanning equipment is cylindrical in nature and image output is square,

    # we set the out-of-scan pixels to 0

    img[img == -2000] = 0

    ax[i].imshow(img, cmap = plt.cm.bone)

plt.show()
fig, ax = plt.subplots(2, 3, figsize = (15, 10))



ax = ax.ravel()



for i, file in enumerate(random_dcm_train):

    img = pydicom.dcmread(file)

    img = img.pixel_array

    img[img == -2000] = 0

    ax[i].imshow(img, cmap = plt.cm.Reds)

plt.show()
def plot_dicom_images(pid = None, df = None, feature = None, dcm_path = None):

    fig, ax = plt.subplots(2, 3, figsize = (15, 10))

    ax = ax.ravel()

    for i in range(len(ax)):

        dcm = pydicom.dcmread(input_path + '/train/' + pid + '/' + str(i + 1) + '.dcm')

        img = dcm.pixel_array

        img[img == -2000] = 0

        ax[i].imshow(img, cmap = plt.cm.bone)

    return None
plot_dicom_images(pid = np.random.choice(train['Patient'].values, 1)[0], df = train)
train['Patient_Week'] = train['Patient'].astype(str) + '_' + train['Weeks'].astype(str)

train
train_out = pd.DataFrame()



train_grp = train.groupby('Patient')



for _, df_out in tqdm(train_grp):

    df_pid = pd.DataFrame()

    for wk, temp in df_out.groupby('Weeks'):

        rename_cols = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'}

        temp = temp.drop(columns = 'Patient_Week').rename(columns = rename_cols)

        drop_cols = ['Age', 'Sex', 'SmokingStatus', 'Percent']

        _df_pid = df_out.drop(columns = drop_cols).rename(columns = {'Weeks': 'predict_Week'}).merge(temp, on = 'Patient')

        _df_pid['Week_passed'] = _df_pid['predict_Week'] - _df_pid['base_Week']

        df_pid = pd.concat([df_pid, _df_pid])

    train_out = pd.concat([train_out, df_pid])



train = train_out[train_out['Week_passed'] != 0].reset_index(drop = True)

train
test = test.rename(columns = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'})



sample['Patient'] = sample['Patient_Week'].apply(lambda x: x.split('_')[0])

sample['predict_Week'] = sample['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)

test = sample.drop(columns = ['FVC', 'Confidence']).merge(test, on = 'Patient')

test['Week_passed'] = test['predict_Week'] - test['base_Week']

test
folds = train[['Patient_Week', 'Patient', 'FVC']].copy()

gkf = GroupKFold(n_splits = 5)

groups = folds['Patient'].values

for i, (trn_idx, val_idx) in enumerate(gkf.split(folds, folds['FVC'], groups)):

    folds.loc[val_idx, 'fold'] = int(i)

folds['fold'] = folds['fold'].astype(int)

folds
import lightgbm as lgbm



params = {

    'num_class': 2,

    'metric': 'None',

    'boosting_type': 'gbdt',

    'learning_rate': 5e-02,

    'seed': seed,

    "subsample": 0.4,

    "subsample_freq": 1,

    'max_depth': 1,

    'verbosity': -1,

}
class OSICLossForLGBM:

    """

    Custom Loss for LightGBM.

    

    * Objective: return grad & hess of NLL of gaussian

    * Evaluation: return competition metric

    """

    

    def __init__(self, epsilon: float=1) -> None:

        """Initialize."""

        self.name = "osic_loss"

        self.n_class = 2  # FVC & Confidence

        self.epsilon = epsilon

    

    def __call__(self, preds: np.ndarray, labels: np.ndarray, weight: tp.Optional[np.ndarray]=None) -> float:

        """Calc loss."""

        sigma_clip = np.maximum(preds[:, 1], 70)

        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000)

        loss_by_sample = - np.sqrt(2) * Delta / sigma_clip - np.log(np.sqrt(2) * sigma_clip)

        loss = np.average(loss_by_sample, weight)

        

        return loss

    

    def _calc_grad_and_hess(

        self, preds: np.ndarray, labels: np.ndarray, weight: tp.Optional[np.ndarray]=None

    ) -> tp.Tuple[np.ndarray]:

        """Calc Grad and Hess"""

        mu = preds[:, 0]

        sigma = preds[:, 1]

        

        sigma_t = np.log(1 + np.exp(sigma))

        grad_sigma_t = 1 / (1 + np.exp(- sigma))

        hess_sigma_t = grad_sigma_t * (1 - grad_sigma_t)

        

        grad = np.zeros_like(preds)

        hess = np.zeros_like(preds)

        grad[:, 0] = - (labels - mu) / sigma_t ** 2

        hess[:, 0] = 1 / sigma_t ** 2

        

        tmp = ((labels - mu) / sigma_t) ** 2

        grad[:, 1] = 1 / sigma_t * (1 - tmp) * grad_sigma_t

        hess[:, 1] = (

            - 1 / sigma_t ** 2 * (1 - 3 * tmp) * grad_sigma_t ** 2

            + 1 / sigma_t * (1 - tmp) * hess_sigma_t

        )

        if weight is not None:

            grad = grad * weight[:, None]

            hess = hess * weight[:, None]

        return grad, hess

    

    def return_loss(self, preds: np.ndarray, data: lgbm.Dataset) -> tp.Tuple[str, float, bool]:

        """Return Loss for lightgbm"""

        labels = data.get_label()

        weight = data.get_weight()

        n_example = len(labels)

        

        # # reshape preds: (n_class * n_example,) => (n_class, n_example) =>  (n_example, n_class)

        preds = preds.reshape(self.n_class, n_example).T

        # # calc loss

        loss = self(preds, labels, weight)

        

        return self.name, loss, True

    

    def return_grad_and_hess(self, preds: np.ndarray, data: lgbm.Dataset) -> tp.Tuple[np.ndarray]:

        """Return Grad and Hess for lightgbm"""

        labels = data.get_label()

        weight = data.get_weight()

        n_example = len(labels)

        

        # # reshape preds: (n_class * n_example,) => (n_class, n_example) =>  (n_example, n_class)

        preds = preds.reshape(self.n_class, n_example).T

        # # calc grad and hess.

        grad, hess =  self._calc_grad_and_hess(preds, labels, weight)



        # # reshape grad, hess: (n_example, n_class) => (n_class, n_example) => (n_class * n_example,) 

        grad = grad.T.reshape(n_example * self.n_class)

        hess = hess.T.reshape(n_example * self.n_class)

        

        return grad, hess
train.dtypes
cat_features = ['Sex', 'SmokingStatus']

num_features = [col for col in train.columns if (train[col].dtype != 'object') & (col not in cat_features)]

print(cat_features, num_features)

features = cat_features + num_features

features = [col for col in features if col not in ['Patient_Week', 'FVC', 'predict_Week', 'base_Week']]

features
import category_encoders as catenc



test['FVC'] = np.nan



ordenc = catenc.OrdinalEncoder(cols = cat_features, handle_unknown = 'impute')

ordenc.fit(train)

train = ordenc.transform(train)

test = ordenc.transform(test)

print('Categorical features encoded..')
nb_splits = 5

oof = np.zeros((len(train), 2))

predictions = np.zeros((len(test), 2))

feature_importance_df = pd.DataFrame()

osic_loss = OSICLossForLGBM()



for n_folds in range(nb_splits):

    print()

    print('Fold No: ', n_folds + 1)

    trn_idx = folds[folds['fold'] != n_folds].index

    val_idx = folds[folds['fold'] == n_folds].index

    #print(trn_idx, val_idx)

    ltrain = lgbm.Dataset(train.iloc[trn_idx][features], label = train.iloc[trn_idx]['FVC'])

    lvalid = lgbm.Dataset(train.iloc[val_idx][features], label = train.iloc[val_idx]['FVC'])

    

    clf = lgbm.train(params, ltrain, 

                    num_boost_round = 10000, 

                    verbose_eval = 100, 

                    early_stopping_rounds = 400, 

                    valid_sets = [ltrain, lvalid], 

                    fobj = osic_loss.return_grad_and_hess,

                    feval = osic_loss.return_loss

                    )

    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration = clf.best_iteration)

    

    # RMSE

    print("CV RMSE score: {:<8.5f}".format(np.sqrt(mean_squared_error(train['FVC'], oof[:, 0]))))

    # Metric

    print("CV Metric: {:<8.5f}".format(osic_loss(oof, train['FVC'])))

    

    fold_imp_df = pd.DataFrame()

    fold_imp_df['feature'] = train[features].columns

    fold_imp_df['importance'] = clf.feature_importance()

    fold_imp_df['fold'] = n_folds + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_imp_df], axis = 0)

    

    predictions += clf.predict(test[features], num_iteration = clf.best_iteration) / nb_splits
cols = (feature_importance_df[['feature', 'importance']]

        .groupby('feature')

        .mean()

        .sort_values(by = 'importance', ascending = False).index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]



plt.figure(figsize = (10,14))

sns.barplot(x = 'importance', y = 'feature', data = best_features.sort_values(by = 'importance', ascending = False))

plt.title('LightGBM Features')

plt.tight_layout()
predictions, oof
train["FVC_pred"] = oof[:, 0]

train["Confidence"] = oof[:, 1]

test["FVC_pred"] = predictions[:, 0]

test["Confidence"] = predictions[:, 1]
sub = pd.read_csv(input_path + 'sample_submission.csv')

sub
submission = sub.drop(columns = ['FVC', 'Confidence']).merge(test[['Patient_Week', 'FVC_pred', 'Confidence']], 

                                                           on = 'Patient_Week')

submission.columns = sub.columns

submission.to_csv('./submission.csv', index = False)

submission.head()
sns.distplot(submission['FVC'])
sns.distplot(submission['Confidence'])
finish = time()

print(strftime("%H:%M:%S", gmtime(finish - start)))