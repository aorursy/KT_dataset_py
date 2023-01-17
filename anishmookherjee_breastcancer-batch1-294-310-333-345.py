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
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

data.head(1).T
data.shape
data.diagnosis.value_counts()
data.dtypes.value_counts()
[col for col in data.select_dtypes(include = 'object')]
data.columns[data.isnull().any()]
data['Unnamed: 32']
data.drop('Unnamed: 32', axis = 1, inplace = True)

data.drop('id', axis=1,inplace = True)

data.columns
data.duplicated().any()
num_cols = [col for col in data.select_dtypes(exclude = 'object').columns]

print(num_cols)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



data[num_cols] = scaler.fit_transform(data[num_cols])



data.head()
dummies = pd.get_dummies(data['diagnosis'], drop_first = True)

print(dummies.shape)

dummies.head(2)
data = pd.concat([data,dummies], axis=1)



data.drop('diagnosis', axis = 1, inplace = True)

data.head()
data.M.value_counts()
data.rename(columns={"M": "target"}, inplace = True)

data.head(2)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
corr = data.corr(method = 'pearson')

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(10, 275, as_cmap=True)



sns.heatmap(corr, cmap=cmap, square=True,

            linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax)

plt.show()
#drop those related features

drop_features = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se',

              'perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst',

              'compactness_se','concave points_se','texture_worst','area_worst']

data.drop(drop_features, axis = 1, inplace = True)
data.head()