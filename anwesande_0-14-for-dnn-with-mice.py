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
df_train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

y_train=df_train.SalePrice.values

df_test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

df=df_train.append(df_test)

df=df.drop(["SalePrice","Id"],axis=1)

df.head()
df.info()
missingp=df.isna().sum()*100/df.shape[0]

missingp.value_counts()
df = df.loc[:, df.isnull().sum() < 0.8*df.shape[0]]
df.info()
df_dum=pd.get_dummies(df)

df_dum.head()
from fancyimpute import IterativeImputer

MICE_imputer = IterativeImputer()

df_MICE = df_dum.copy(deep=True)

df_MICE.iloc[:, :] = MICE_imputer.fit_transform(df_MICE)
missingp1=df_MICE.isna().sum()*100/df_MICE.shape[0]

missingp1.unique()
df_MICE.head()
y_train.shape

def split_combined():

    global df_MICE

    train = df_MICE[:1460]

    test = df_MICE[1460:]



    return train , test 

  

train, test = split_combined()
test.shape
import tensorflow as tf

model=tf.keras.models.Sequential([

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,kernel_initializer="normal",activation="relu"),

    tf.keras.layers.Dense(256,kernel_initializer="normal",activation="relu"),

    tf.keras.layers.Dense(256,kernel_initializer="normal",activation="relu"),

    tf.keras.layers.Dense(256,kernel_initializer="normal",activation="relu"),

    tf.keras.layers.Dense(1,kernel_initializer='normal',activation='linear')



])
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.fit(train,y_train, epochs=500)
predictions=model.predict(test)
predictions
testpred=pd.DataFrame(predictions)

sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],testpred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission.csv',index=False)