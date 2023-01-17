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
from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import scipy

import numpy

import json

import sys

import csv

import os
print('matplotlib: {}'.format(matplotlib.__version__))

print('sklearn: {}'.format(sklearn.__version__))

print('scipy: {}'.format(scipy.__version__))

print('seaborn: {}'.format(sns.__version__))

print('pandas: {}'.format(pd.__version__))

print('numpy: {}'.format(np.__version__))

print('Python: {}'.format(sys.version))
sns.set(style='white', context='notebook', palette='deep')

pylab.rcParams['figure.figsize'] = 12,8

warnings.filterwarnings('ignore')

mpl.style.use('ggplot')

sns.set_style('white')

%matplotlib inline
# import train and test to play with it

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
# shape

print(df_train.shape)
#columns*rows

df_train.size
def check_missing_data(df):

    flag=df.isna().sum().any()

    if flag==True:

        total = df.isnull().sum()

        percent = (df.isnull().sum())/(df.isnull().count()*100)

        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        data_type = []

        # written by MJ Bahmani

        for col in df.columns:

            dtype = str(df[col].dtype)

            data_type.append(dtype)

        output['Types'] = data_type

        return(np.transpose(output))

    else:

        return(False)
check_missing_data(df_train)
check_missing_data(df_test)
print(df_train.info())
df_train['Age'].unique()
df_train["Pclass"].value_counts()
df_train.head() 
df_train.tail() 
df_train.sample(5) 
df_train.describe()
df_train.isnull().sum()
df_train.groupby('Survived').count()
df_train.columns
df_train.where(df_train['Age']==30).head(2)
df_train[df_train['Age']==30]
X = df_train.iloc[:, :-1].values

y = df_train.iloc[:, -1].values