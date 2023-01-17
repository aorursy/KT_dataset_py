import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
train = pd.read_csv('../input/leaf-classification/train.csv.zip', compression='zip', header=0, sep=',', quotechar='"')
train.head()
train.info()
train.isnull().sum()
train["species"].value_counts()
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
#train["species"] = l.fit_transform(train["species"])
df_train = train.drop(columns=["species","id"],axis=1)
df_train.shape
test = pd.read_csv('../input/leaf-classification/test.csv.zip')
test.head()
df_test = test.drop(columns="id",axis=1)
df_test.shape
x_train = df_train

y_train = train["species"]

x_test = df_test
x_train.shape,x_test.shape,y_train.shape
from  xgboost import XGBClassifier

model = XGBClassifier()
s = model.fit(x_train,y_train).predict_proba(x_test)
s
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

lb.fit(y_train)

print(lb.classes_)
classes = np.unique(y_train)

"Number of unique classes: {0}".format(len(classes))
model.classes_
result = pd.DataFrame(index= test.id, columns=model.classes_, data=model.predict_proba(x_test))

#result = pd.DataFrame(result)

result.to_csv("submission.csv", index_label='id')
result.head()