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
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv',nrows=1459)

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv',nrows=1459)

df_test0=df_test

df_train.info()
df_train.corr()['SalePrice']
import matplotlib.pyplot as plt

plt.scatter(df_train['GrLivArea'],df_train['SalePrice'],s=5, cmap='viridis')

plt.show
target = df_train['SalePrice']

df_train = df_train.drop(['SalePrice'],axis=1)
df_tt = pd.concat([df_train,df_test],axis=0)

df_tt = df_tt.drop('Id',axis=1)

df_tt = df_tt.drop(['MiscFeature','Fence','PoolQC','Alley'],axis=1)
df_tt_obj = df_tt.select_dtypes('object')

df_tt_int = df_tt.select_dtypes('int64')
df_tt_obj.info()
xx = df_tt_obj.columns[df_tt_obj.isna().any()].tolist()
df_tt_obj = df_tt_obj.fillna('None')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ohe = OneHotEncoder()
ohe.fit(df_tt_obj)
df_tt_obj = ohe.transform(df_tt_obj).toarray()
df_tt_obj[:2]
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

mms.fit(df_tt_int)
df_tt_int = mms.transform(df_tt_int)
df_tt_int[:2]
feature = np.hstack([df_tt_obj,df_tt_int])
feature_train = feature[:(len(target)),:]

feature_test = feature[len((target)):,:]
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(feature_train,target)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(trainX,trainY)
rf.score(testX,testY)
rf.feature_importances_
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.ensemble import GradientBoostingRegressor

gbt = GradientBoostingRegressor()

gbt
gbt = gbt.fit(trainX,trainY)
gbt.score(testX,testY)
pred = rf.predict(testX)
ypred=gbt.predict(feature_test)
output = pd.DataFrame({'Id': df_test0.Id, 'SalePrice': ypred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")