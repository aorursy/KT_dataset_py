# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

total=train.append(test)



total.shape

total.describe()
total.head()
total.Age=total.Age.fillna(total.Age.mean())

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns
# Data Understanding

plt.subplot(2,2,1)

train.Pclass.value_counts().plot(kind="bar")

plt.ylabel(u"Num of people")

plt.title(u"Rank of Pclass")



plt.subplot(2,2,2)

plt.scatter(train.Survived, train.Age)

plt.ylabel(u"Age")                         # 设定纵坐标名称

plt.grid(b=True, which='major', axis='y') 

plt.title(u"Survived distribution by age")



plt.subplot(2,2,3)

train.Age[train.Pclass == 1].plot(kind='kde')   

train.Age[train.Pclass == 2].plot(kind='kde')

train.Age[train.Pclass == 3].plot(kind='kde')

plt.xlabel(u"age")# plots an axis lable

plt.ylabel(u"density") 

plt.title(u"Rank")

plt.legend((u'1', u'2',u'3'),loc='best') # sets our legend for our graph.





help(train.plot)