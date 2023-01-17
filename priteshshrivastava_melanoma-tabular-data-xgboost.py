import numpy as np 

import pandas as pd 

import os

from glob import glob

from tqdm import tqdm

import seaborn as sns

sns.set(style = 'dark')

import matplotlib.pyplot as plt
train_files_dir = glob('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/*')

test_files_dir = glob('/kaggle/input/siim-isic-melanoma-classification/jpeg/test/*')
train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
train_df.head()
test_df.head()
train_df['sex'].fillna('unkown',inplace = True) # missing value
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
train_df['sex_enc'] = enc.fit_transform(train_df.sex.astype('str'))

test_df['sex_enc'] = enc.transform(test_df.sex.astype('str'))
plt.figure(figsize = (12,6))

sns.countplot(x = 'sex', hue = 'target', data = train_df)
train_df.head()
test_df.anatom_site_general_challenge = test_df.anatom_site_general_challenge.fillna('unknown')

train_df.anatom_site_general_challenge = train_df.anatom_site_general_challenge.fillna('unknown')
train_df['anatom_enc']= enc.fit_transform(train_df.anatom_site_general_challenge.astype('str'))

test_df['anatom_enc']= enc.transform(test_df.anatom_site_general_challenge.astype('str'))
train_df['age_approx'] = train_df['age_approx'].fillna(train_df['age_approx'].mode().values[0])

test_df['age_approx']  = test_df['age_approx'].fillna(test_df['age_approx'].mode().values[0]) # Test data doesn't have any NaN in age_approx
plt.figure(figsize = (20,6))

sns.countplot(x = 'age_approx', hue = 'target', data = train_df)
train_df['n_images'] = train_df.patient_id.map(train_df.groupby(['patient_id']).image_name.count())

test_df['n_images'] = test_df.patient_id.map(test_df.groupby(['patient_id']).image_name.count())
train_images = train_df['image_name'].values

train_sizes = np.zeros(train_images.shape[0])

for i, img_path in enumerate(tqdm(train_images)):

    train_sizes[i] = os.path.getsize(os.path.join('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/', f'{img_path}.jpg'))

    

train_df['image_size'] = train_sizes





test_images = test_df['image_name'].values

test_sizes = np.zeros(test_images.shape[0])

for i, img_path in enumerate(tqdm(test_images)):

    test_sizes[i] = os.path.getsize(os.path.join('/kaggle/input/siim-isic-melanoma-classification/jpeg/test/', f'{img_path}.jpg'))

    

test_df['image_size'] = test_sizes
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scale = MinMaxScaler()

train_df['image_size_scaled'] = scale.fit_transform(train_df['image_size'].values.reshape(-1, 1))

test_df['image_size_scaled'] = scale.transform(test_df['image_size'].values.reshape(-1, 1))
train_df['age_id_min']  = train_df['patient_id'].map(train_df.groupby(['patient_id']).age_approx.min())

train_df['age_id_max']  = train_df['patient_id'].map(train_df.groupby(['patient_id']).age_approx.max())



test_df['age_id_min']  = test_df['patient_id'].map(test_df.groupby(['patient_id']).age_approx.min())

test_df['age_id_max']  = test_df['patient_id'].map(test_df.groupby(['patient_id']).age_approx.max())
features = [

            'age_approx',

            'age_id_min',

            'age_id_max',

            'sex_enc',

            'anatom_enc',

            'n_images',

            'image_size_scaled',

           ]
X = train_df[features]

y = train_df['target']



X_test = test_df[features]
# Load libraries for training

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
model = XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.8, gamma=1, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.002, max_delta_step=0, max_depth=10,

             min_child_weight=1, missing=None, monotone_constraints=None,

             n_estimators=700, n_jobs=-1, nthread=-1, num_parallel_tree=1,

             objective='binary:logistic', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, silent=True, subsample=0.8,

             tree_method=None, validate_parameters=False, verbosity=None)



kfold = StratifiedKFold(n_splits=5, random_state=1001, shuffle=True)

cv_results = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc', verbose = 3)

cv_results.mean()
model.fit(X,y)

pred_xgb = model.predict(X_test)
feature_important = model.get_booster().get_score(importance_type='weight')

keys = list(feature_important.keys())

values = list(feature_important.values())



data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

plt.figure(figsize= (12,10))

sns.barplot(x = data.score , y = data.index, orient = 'h', palette = 'Blues_r')
sub = pd.DataFrame({'image_name':test_df.image_name.values,

                    'target':pred_xgb})

sub.to_csv('submission.csv',index = False)