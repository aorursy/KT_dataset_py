import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

train = pd.read_csv("../input/digit-recognizer/train.csv")
train.shape
t = train.iloc[:,1:]
train["label"][2]
plt.imshow(t.iloc[2:3,:].values.reshape(28,28))
x_train = train.drop(columns="label",axis=1)

y_train = train["label"]
print(x_train.shape)

print(y_train.shape)
test = pd.read_csv("../input/digit-recognizer/test.csv")
test.shape
x_test = test
x_test.shape
from xgboost import XGBClassifier

m = XGBClassifier()
m.fit(x_train,y_train)
y_pred = m.predict(x_test)
dfs = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
f = {"ImageId":dfs["ImageId"],"Label":y_pred}

f = pd.DataFrame(f)

f.to_csv("submission.csv",index=False)