# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
%matplotlib inline
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Any results you write to the current directory are saved as output.
print(os.getcwd())
df = pd.read_csv('../input/train.csv')
#df.head()
tf = pd.read_csv('../input/test.csv')
tf.head()

#sns.heatmap()
df.info()
#df.info()
test_id = tf['Id']
price_train = df['SalePrice']
#type(price_train)
#price_train.head()
df.drop('SalePrice',axis=1,inplace=True)
df.info()
data_id = df['Id']
df.drop('Id',axis=1,inplace=True)
tf.drop('Id',axis = 1,inplace=True)
df.get_dtype_counts()
df_objects = df.select_dtypes(include=['object']).copy()
tf_objects = tf.select_dtypes(include=['object']).copy()


df_objects.head()
df_objects.isnull().sum()
df_objects.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
tf_objects.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

df_objects.isnull().sum()
#df_objects = df_objects.fillna(df_objects.va)
#sns.heatmap(pd.concat([df['FireplaceQu'],price_train],axis=1).corr())
df_objects = df_objects.apply(lambda x:x.fillna(x.value_counts().index[0]))
tf_objects = tf_objects.apply(lambda x:x.fillna(x.value_counts().index[0]))

df_objects.isnull().sum()
encoder = LabelEncoder()
df_objects_v2 = df_objects.apply(lambda x: x.astype('category').cat.codes)
tf_objects_v2 = tf_objects.apply(lambda x: x.astype('category').cat.codes)
df_objects_v2.head()
df_non_obj = df.select_dtypes(exclude=['object'])
tf_non_obj = tf.select_dtypes(exclude=['object'])

#df_non_obj.info()
df_non_obj.info()
df_non_obj = df_non_obj.apply(lambda x:x.fillna(x.value_counts().index[0]))
tf_non_obj = tf_non_obj.apply(lambda x:x.fillna(x.value_counts().index[0]))
df_non_obj.info()
df_objects_v2.info()
train_data = pd.concat([df_non_obj,df_objects_v2],axis = 1)
test_data = pd.concat([tf_non_obj,tf_objects_v2],axis=1)
train_vals = train_data.values
test_vals = test_data.values


train_labels = price_train.values
print(train_vals.shape,train_labels.shape)

print(train_vals.shape,train_labels.shape)
print(test_vals.shape)


#train_test_split?
#X_train,val_train,X_labels,val_labels=train_test_split(train_vals,train_labels,random_state=42)
#xgb = XGBRegressor?
#xgb = XGBRegressor()
#xgb.fit(X_train,X_labels)
#mean_squared_log_error(val_labels,xgb.predict(val_train))
xgb_model = XGBRegressor()
#model.save_model('xgb_houses.model')
xgb_model.fit(train_vals,train_labels)
#xgb_model.save_model('xgb_houses.model')
pred_vals = xgb_model.predict(test_vals)
#print(os.listdir())
print(pred_vals.shape)
data = {'SalePrice':pred_vals}
data_pd = pd.DataFrame(data)
data_pd.head()
fin_vals = pd.concat([test_id,data_pd],axis = 1)
fin_vals.to_csv('result.csv',index = False)