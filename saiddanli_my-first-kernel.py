# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.info()
data.head(10)
data.columns
# Datamızdan şuan için kullanmayacağımız sütunları çıkartarak datamızı sadeleştiriyoruz.

data.drop(['restecg','exang','oldpeak', 'slope', 'ca', 'thal'], axis = 1, inplace= True)

data
data.corr()
f,ax = plt.subplots(figsize=(12,12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".1f", ax=ax)

plt.show()
data.plot(kind="scatter", x="age",y="thalach",color="red", alpha=.5)

plt.xlabel("age")

plt.ylabel("max_heart_rate")

plt.show()
data.target.value_counts()
sns.countplot(x="target", data=data, palette="bwr" )

plt.show()