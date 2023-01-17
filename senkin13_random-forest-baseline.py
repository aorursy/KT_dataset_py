%%time

import pandas as pd

import numpy as np

import gc

import os

import random

import glob

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl

from matplotlib_venn import venn2

%matplotlib inline



inputPath = '/kaggle/input/used-car-price-forecasting/'

train = pd.read_csv(inputPath + 'train.csv')

test = pd.read_csv(inputPath + 'test.csv')

train['flag'] = 'train'

test['flag'] = 'test'



df = pd.concat([train,test],axis=0)

del train,test

gc.collect()
df.describe()
%%time

# fillna with most frequent value

df['year'].fillna(df['year'].mode()[0], inplace=True)



# fillna with new category

df['model'] = df['model'].fillna('nan')



# fillna with new category

df['condition'] = df['condition'].fillna('nan')



# fillna with new value

df['cylinders'] = df['cylinders'].fillna('-2 cylinders')

df['cylinders'] = df['cylinders'].map(lambda x:x.replace('other','-1 cylinders'))



# fillna with new category

df['fuel'] = df['fuel'].fillna('nan')



# fillna with new value

df['odometer'] = df['odometer'].fillna('-1')

df['odometer'] = df['odometer'].astype(float)



# fillna with new category

df['title_status'] = df['title_status'].fillna('nan')



# fillna with new category

df['transmission'] = df['transmission'].fillna('nan')



# fillna with new category

df['vin'] = df['vin'].fillna('nan')



# fillna with new category

df['drive'] = df['drive'].fillna('nan')



# fillna with new category

df['size'] = df['size'].fillna('nan')



# fillna with new category

df['type'] = df['type'].fillna('nan')



# fillna with new category

df['paint_color'] = df['paint_color'].fillna('nan')
%%time

df['cylinders'] = df['cylinders'].map(lambda x:x.split(' ')[0])

df['cylinders'] = df['cylinders'].astype(int)
%%time

df = pd.get_dummies(df, columns=['paint_color'])
%%time

for c in ['region','manufacturer','model','condition','fuel','title_status','transmission', 'vin', 'drive', 'size', 'type', 'state']:

    lbl = LabelEncoder()

    df[c] = lbl.fit_transform(df[c].astype(str))
%%time

for c in ['region','manufacturer','model','condition','fuel','title_status','transmission', 'vin', 'drive', 'size', 'type', 'state']:

    df['count_' + c] = df.groupby([c])['flag'].transform('count')
%%time

df['mean_manufacturer_odometer'] = df.groupby(['manufacturer'])['odometer'].transform('mean')

df['std_manufacturer_odometer'] = df.groupby(['manufacturer'])['odometer'].transform('std')

df['max_manufacturer_odometer'] = df.groupby(['manufacturer'])['odometer'].transform('max')

df['min_manufacturer_odometer'] = df.groupby(['manufacturer'])['odometer'].transform('min')

df['maxmin_manufacturer_odometer'] = df['max_manufacturer_odometer'] - df['min_manufacturer_odometer']
%%time

df['num_chars'] = df['description'].apply(len) 

df['num_words'] = df['description'].apply(lambda x: len(x.split()))

df['num_unique_words'] = df['description'].apply(lambda x: len(set(w for w in x.split())))
%%time

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



def rmse(y_true, y_pred):

    return (mean_squared_error(y_true, y_pred))** .5



train_df = df[df['flag']=='train']

train_df['price'] = np.log1p(train_df['price'])

test_df = df[df['flag']=='test']

del df

gc.collect()



drop_features = ['id', 'price', 'description', 'flag']

features = [f for f in train_df.columns if f not in drop_features]



train_x, valid_x, train_y, valid_y = train_test_split(train_df[features], train_df['price'], test_size=0.2, random_state=1,stratify=train_df['manufacturer'])

model = RandomForestRegressor(n_estimators=50,max_depth=10,random_state=1,verbose=1,n_jobs=-1)

model.fit(train_x, train_y)

valid_preds = model.predict(valid_x)

print('Valid RMSE Score:', rmse(valid_y, valid_preds))
importances = model.feature_importances_

indices = np.argsort(importances)[-20:]

plt.figure(figsize=(20, 10))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
%%time

test_preds = model.predict(test_df[features])

test_df['price'] = np.expm1(test_preds)

test_df[['id','price']].to_csv('submission.csv',index=False)