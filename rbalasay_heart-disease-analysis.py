# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import seaborn as sns

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
# read the data

heart_data = pd.read_csv('../input/heart.csv', sep=',')
# view all column names and few rows of data

heart_data.describe()
sns.set()

# people with heart attack based on sex,age,cp

sns.catplot(x="sex", y="age",col="target",hue="cp", kind="swarm",data=heart_data)
# comparison of people with heart attack on basis of sex

sns.countplot(x="target", hue="sex",data=heart_data)

plt.show()
# comparison of fbs on basis of sex

sns.countplot(x="fbs", hue="sex",data=heart_data)

plt.show()
# comparison of fbs on basis of people who had a heart attack

sns.countplot(x="fbs", hue="target",data=heart_data)

plt.show()
# comparison of restecg on basis of sex

sns.countplot(x="restecg", hue="sex",data=heart_data)

plt.show()
# comparison of restecg on basis of people who had a heart attack

sns.countplot(x="restecg", hue="target",data=heart_data)

plt.show()
# comparison of thal on basis of sex

sns.countplot(x="thal", hue="sex",data=heart_data)

plt.show()
# comparison of thal on basis of people who had a heart attack

sns.countplot(x="thal", hue="target",data=heart_data)

plt.show()