# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = '/kaggle/input/055241hk192p1'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
def get_features(df, exclude_cols=['# Id', 'Category']):

    exclude_cols = set(exclude_cols).intersection(set(df.columns))

    return df.drop(exclude_cols, axis=1)
train_feats_describe = get_features(df_train).describe()

train_feats_describe
missing_values = df_train.isna().sum(axis=0)

# missing cols

missing_values[missing_values > 0]
label_count = df_train['Category'].value_counts()
fig, ax = plt.subplots(figsize=(10, 10))



_ = ax.pie(label_count.values, labels=['Category {:02d}'.format(lbl) for lbl in list(label_count.index)], autopct='%1.1f%%', shadow=True, startangle=90, counterclock=False) 
train_feats_describe.loc['min', :].describe()
train_feats_describe.loc['max', :].describe()
sns.distplot(train_feats_describe.loc['max', :])
train_feats_describe.loc['mean', :].describe()
sns.distplot(train_feats_describe.loc['mean', :])
from sklearn.decomposition import PCA
pca_train = PCA(n_components=2).fit_transform(get_features(df_train))

pca_train = pd.DataFrame({

    'x': pca_train[:, 0],

    'y': pca_train[:, 1],

    'label': df_train['Category']

})
fig, ax = plt.subplots(figsize=(10, 10))

_ = sns.scatterplot(x='x', y='y', data=pca_train, hue='label', ax=ax, legend='full')