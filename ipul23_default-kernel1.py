#Sample kernel

from sklearn.model_selection import KFold, cross_val_score, train_test_split

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor
df_sample = pd.read_csv("../input/sample_airbnb.csv")

df_train = pd.read_csv("../input/train_airbnb.csv")

df_test = pd.read_csv("../input/test_airbnb.csv")
df_train.head()
#check missing value

for i in df_train.columns:

    print(i,' ',df_train[i].isnull().sum())

#check missing value

for i in df_test.columns:

    print(i,' ',df_test[i].isnull().sum())
#drop categorical

for i in df_train.columns:

    if (df_train[i].dtypes.name == 'object'):

        df_train.drop(i,axis=1,inplace=True)

        df_test.drop(i,axis=1,inplace=True)
#drop missing value

for i in df_train.columns:

    if (df_train[i].isnull().sum() > 0):

        df_train.drop(i,axis=1,inplace=True)

#drop missing value

for i in df_test.columns:

    if (df_test[i].isnull().sum() > 0):

        df_test.drop(i,axis=1,inplace=True)
model = RandomForestRegressor()

X = df_train.drop('price',axis=1)

y = df_train['price']

model.fit(X,y)

predict = model.predict(df_test)
submit = pd.read_csv("../input/sample_airbnb.csv")

submit['price'] = predict

submit.to_csv("answer.csv",index=False)