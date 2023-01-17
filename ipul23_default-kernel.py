#Sample kernel

from sklearn.model_selection import KFold, cross_val_score, train_test_split

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor
df_sample = pd.read_csv("../input/sample_black.csv")

df_train = pd.read_csv("../input/train_black.csv")

df_test = pd.read_csv("../input/test_black.csv")
df_train.head()
#check missing value

for i in df_train.columns:

    print(i,' ',df_train[i].isnull().sum())

#check missing value

for i in df_test.columns:

    print(i,' ',df_test[i].isnull().sum())
#Fill missing value

df_train['Product_Category_2'].fillna(df_train['Product_Category_2'].mean(),inplace=True)

df_train['Product_Category_3'].fillna(df_train['Product_Category_3'].mean(),inplace=True)



df_test['Product_Category_2'].fillna(df_test['Product_Category_2'].mean(),inplace=True)

df_test['Product_Category_3'].fillna(df_test['Product_Category_3'].mean(),inplace=True)
df_train.dtypes
#drop categorical

for i in df_train.columns:

    if (df_train[i].dtypes.name == 'object'):

        df_train.drop(i,axis=1,inplace=True)

        df_test.drop(i,axis=1,inplace=True)
model = RandomForestRegressor()

X = df_train.drop('Purchase',axis=1)

y = df_train['Purchase']

model.fit(X,y)

predict = model.predict(df_test)
submit = pd.read_csv("../input/sample_black.csv")

submit['Id'] = submit['Id']

submit['Purchase'] = predict

submit.to_csv("answer.csv",index=False)