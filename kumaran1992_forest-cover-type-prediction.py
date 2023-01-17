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

import seaborn as sns

from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,auc

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
df_train = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')

df_test = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')

df_train.head()
df_train.shape
df_test.shape
test_id =df_test['Id']

train_id = df_train ['Id']
df_train.isnull().sum()
df_test.isnull().sum()
df_train.columns
df_test.columns
df_train.dtypes
df_test.dtypes
df_train.shape
df_test.shape
# From both train and test data

df_train.drop(['Id'], axis = 1,inplace = True)

df_test.drop(['Id'], axis = 1,inplace = True)
df_train.shape
df_train.columns
df_train.shape
df_test.shape
sns.heatmap(df_train.isnull())
sns.heatmap(df_test.isnull())
df_test.isnull().sum()
corrmat = df_train.corr()

sns.heatmap(corrmat,vmax = 0.8,square = True)
data_corr = df_train.corr()

threshold = 0.5

corr_list = []
data_corr.head()
cols = df_train.columns.tolist()

cols
df_train.skew()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit_transform(df_train)
df_train.shape
df_train.columns
df_test.shape
x = df_train.drop(['Cover_Type'],axis = 1)

y = df_train['Cover_Type']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
xgb = XGBClassifier()

xgb.fit(x_train,y_train)
predics =xgb.predict(x_test)

predics

accuracy_score(y_test,predics)
df_test['Cover_Type'] = xgb.predict(df_test)

df_test['Cover_Type']
my_submission = pd.DataFrame({'Id':test_id,'Cover_Type': df_test['Cover_Type']})

my_submission.to_csv('submission.csv', index=False)
my_submission.to_csv(r'my_submission.csv')
from IPython.display import FileLink

FileLink(r'submission.csv')