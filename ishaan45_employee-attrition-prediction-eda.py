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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse

import seaborn as sns

import xgboost as xgb

from xgboost import plot_importance

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# Reading the train and test file

train_file = pd.read_csv('/kaggle/input/Train.csv')
train_file.info()
train_file['Gender'].value_counts() / len(train_file) * 100
null_count = train_file.isnull().sum().reset_index()

null_count.columns = ['feature_name','missing_count']

null_count = null_count[null_count['missing_count'] > 0].sort_values(by='missing_count',ascending=True)

null_value_count = pd.Series(null_count['missing_count'].values, index=null_count['feature_name'])
null_value_count / len(train_file) * 100
cols = [i for i in train_file.columns if train_file[i].dtype == 'int64' or train_file[i].dtype == 'float64']

corr_num = train_file[cols].corr()

sns.heatmap(corr_num)
def encode(data):

    cat_cols = []

    for col in data.columns:

        if data[col].dtype == 'object':

            data[col] = le.fit_transform(data[col].astype(str))

            cat_cols.append(col)

    return data , cat_cols
train_data, cat_cols = encode(train_file)
corr_obj = train_data[cat_cols].corr(method='spearman')

sns.heatmap(corr_obj)
sns.distplot(np.log1p(train_file['Attrition_rate']))
def highlight_cols(s):

    color = '#ADD8E6'

    return 'background-color: %s' % color
train_file.style.applymap(highlight_cols, subset=cat_cols)