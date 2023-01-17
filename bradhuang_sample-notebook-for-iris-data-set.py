#Common import

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt
df_train = pd.read_csv("/kaggle/input/2401-iris-data-set/train.csv")

df_test = pd.read_csv("/kaggle/input/2401-iris-data-set/test.csv")

df_train.head(5)
df_train.info() #Show the info of all columns
df_train_x = df_train.drop(columns=['class'])

df_train_y = df_train["class"].copy()
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)

clf.fit(df_train_x, df_train_y)
df_test_predicted = clf.predict(df_test)
df_test_submit = pd.DataFrame({"Id": df_test['Id'], "class":df_test_predicted})

df_test_submit.head(5)
df_test_submit.to_csv("submit.csv",index=False)