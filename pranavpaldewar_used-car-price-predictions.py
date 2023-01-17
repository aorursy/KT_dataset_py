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
df=pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')

pd.set_option('max_columns',100)
df.head()
dr=['url','region_url','region','description','image_url','lat','long','vin','id','county','title_status',
    'model','state']
df_1=df.drop(dr,axis=1)
df_1.shape
df_1.describe()
df_1.info()
df_1=df_1.dropna()
df_1.head()
cat_df=df_1.select_dtypes(include='object')
cat_df.columns
cat_df.head()
import seaborn as sns
import matplotlib.pyplot as plt 
plt.figure(figsize=(10,6))
sns.heatmap(df_1.corr(),annot=True)
num_df=df_1.select_dtypes(exclude='object')
num_df.head()
plt.figure(figsize=(10,6))
plt.scatter('year','price',data=df_1,alpha=0.5)

plt.figure(figsize=(10,6))
plt.scatter(df_1.odometer,df_1.price)
plt.figure(figsize=(10,6))
cat_df.manufacturer.value_counts().plot(kind='bar')
plt.figure(figsize=(20,12))
df_1.year.value_counts().plot(kind='bar')

df_1.condition.value_counts().plot(kind='bar')
df_1.paint_color.value_counts().plot(kind='bar')
df_1.groupby('paint_color')['price'].median().plot(kind='bar')
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
for col in cat_df.columns:
    cat_df[col]=label.fit_transform(cat_df[col].astype('str'))
cat_df.head()
cat_df.shape
df_1.head()
df_1.groupby('transmission')['price'].median().plot()
plt.figure(figsize=(10,6))
df_1.groupby('cylinders')['price'].median().plot()
df_1.transmission.value_counts().plot(kind='bar')
final = cat_df.merge(num_df,on=cat_df.index)
from sklearn.model_selection import train_test_split
y=final['price']
x=final.drop('price',axis=1)
train_x,valid_x,train_y,valid_y=train_test_split(x,y,test_size=0.2,random_state=1)
train_x.shape
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=100)
rfr.fit(train_x,train_y)
rfr.score(valid_x,valid_y)
predict_1=rfr.predict(valid_x)
predict_1=rfr.predict(valid_x)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(valid_y,predict_1)

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(train_x,train_y)
predict_2=dtr.predict(valid_x)
mean_absolute_error(valid_y,predict_2)
from sklearn.linear_model import Ridge
r=Ridge(alpha=1)
r.fit(train_x,train_y)
pre=r.predict(valid_x)
mean_absolute_error(valid_y,pre)
import tensorflow as tf
model_1=tf.keras.Sequential([tf.keras.layers.Dense(12,input_dim=12,activation='relu'),
                          tf.keras.layers.Dense(256,activation='relu'),
                          tf.keras.layers.Dense(256,activation='relu'),
                          tf.keras.layers.Dense(1,activation='linear')])
model_1.compile(loss='mae',optimizer='adam',metrics=['accuracy'])
model_1.fit(train_x,train_y,epochs=200)
