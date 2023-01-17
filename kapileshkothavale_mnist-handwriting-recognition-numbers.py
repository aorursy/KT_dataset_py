# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import training and test datasets

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_test.shape

df_train.head()
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

df_train.shape
x_train = df_train.iloc[: 24000,1:]

y_train = df_train.iloc[: 24000,0]

x_test = df_train.iloc[24000:,1:]

y_test = df_train.iloc[24000:,0]

y_test.shape
dtc.fit(x_train,y_train)
st = df_test.iloc[52].as_matrix()

st = st.reshape(28,28)

plt.imshow(255-st,cmap="gray")

plt.show()

print(dtc.predict([df_test.iloc[52]]))
st = df_test.iloc[24].as_matrix()

st = st.reshape(28,28)

plt.imshow(255-st,cmap="gray")

plt.show()

print(dtc.predict([df_test.iloc[24]]))
st = df_test.iloc[16].as_matrix()

st = st.reshape(28,28)

plt.imshow(255-st,cmap="gray")

plt.show()

print(dtc.predict([df_test.iloc[16]]))
pred = dtc.predict(x_test)

print(dtc.score(x_test,y_test)*100)