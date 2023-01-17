# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcard.csv")



data.head(10)
data.describe()
data.columns
data.isnull().sum()
"We will find the correlation among the various features and the class"



corr = data.corr()

corr
import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(data=corr,vmax=1)

plt.show()

data = data.drop(['Time'],axis=1)
#execution of this section will require some time since there are a lot of computations.

#execute this code if you want to have a better insight into the distribution of data

"""data['Class'].unique()

fig, ax = plt.subplots(5,6,sharex=False, sharey=False, figsize=(20,24))

i=1

for column in data.columns:

    if(column != 'Class'):

        data0 = data.loc[data['Class']==0,column]

        data1 = data.loc[data['Class']==1,column]

        plt.subplot(5,6,i)

        i = i + 1

        data0.plot(kind='density',label='class 0')

        data1.plot(kind='density',label='class 1')

        plt.ylabel(column)

        plt.legend()"""
target = data['Class']

features = data.drop(['Class'], axis = 1)
from sklearn.model_selection import train_test_split

train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size = 0.25, random_state = 43)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(train_X,train_Y)

pred_class = model.predict(test_X)

pred_class
from sklearn.metrics import confusion_matrix

con_matrix = confusion_matrix(test_Y, pred_class)

tot = test_Y.count()

con_matrix/tot