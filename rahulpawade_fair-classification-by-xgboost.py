import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/fair-classification/train.csv")
train.info()
sns.heatmap(train.isna(),yticklabels=0)
train.head()
train_object = train.select_dtypes(include=object)

train_num = train.select_dtypes(include=np.number)
for i in train_object.columns:

    train_object[i]=pd.get_dummies(train_object[i],drop_first=1)
df_train = pd.concat([train_object,train_num],axis=1)
df_train.info()
sns.pairplot(df_train)
test = pd.read_csv("../input/fair-classification/test_no_income.csv")
test.info()
sns.heatmap(test.isna(),yticklabels=0)
test_object = test.select_dtypes(include=object)

test_num = test.select_dtypes(include=np.number)
for i in test_object.columns:

    test_object[i]=pd.get_dummies(test_object[i],drop_first=1)
df_test = pd.concat([test_object,test_num],axis=1)
df_test.info()
df_test.columns
x_train = df_train[['workclass', 'education', 'marital-status', 'occupation',

       'relationship', 'native-country', 'Age', 'fnlwgt', 'education-num',

       'race', 'gender', 'capital gain', 'capital loss', 'hours per week']]

y_train = df_train["income"]
x_test = df_test[['workclass', 'education', 'marital-status', 'occupation',

       'relationship', 'native-country', 'Age', 'fnlwgt', 'education-num',

       'race', 'gender', 'capital gain', 'capital loss', 'hours per week']]
x_train.shape,x_test.shape,y_train.shape
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
s = pd.read_csv("../input/fair-classification/test_sample1.csv")
f = {"Id":s["Id"],"income":y_pred}
f = pd.DataFrame(f)
f.to_csv("submission.csv",index=False)
f.head()