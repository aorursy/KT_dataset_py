# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('/kaggle/input/iris/Iris.csv')

df.head()
sns.catplot(x="Species", y="SepalLengthCm", data = df)

sns.catplot(x="Species", y="SepalWidthCm", data = df)

sns.catplot(x="Species", y="PetalLengthCm", data = df)

sns.catplot(x="Species", y="PetalWidthCm", data = df)
corrmat = df.corr()

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(corrmat, vmax=.5, square=True, cmap="RdYlGn", fmt='.1f', linewidth=0.1)
sns.pairplot(df)
#label encode the target variable

encode = LabelEncoder()

df.Species = encode.fit_transform(df.Species)
df.head()
train, test = train_test_split(df, test_size=0.2, random_state=0)
train.shape, test.shape
train_x=train.drop(columns=['Species'], axis=1)

train_y= train['Species']
test_x=test.drop(columns=['Species'], axis=1)

test_y= test['Species']
model=LogisticRegression()

model.fit(train_x, train_y)
predict = model.predict(test_x)
print(test_y[0:10])

print(predict[0:10])
print(accuracy_score(test_y, predict))