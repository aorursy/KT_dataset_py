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
import pandas as pd 

import matplotlib.pyplot as plt

from pylab import *

import numpy as np



from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

import seaborn as sns
dataset = pd.read_csv("../input/glass.csv")

print(dataset.shape)

dataset.head(3)
sns.pairplot(dataset, hue = 'Type')
dataset['Type'].groupby(dataset['Type']).count()
(dataset.iloc[:,0:9]).describe()
plt.figure(figsize=(10,10))

subplot(9,1,1)

sns.boxplot('Type','Na',data =dataset)



subplot(9,1,2)

sns.boxplot('Type','Mg',data =dataset)



subplot(9,1,3)

sns.boxplot('Type','Al',data =dataset)



subplot(9,1,4)

sns.boxplot('Type','Si',data =dataset)



subplot(9,1,5)

sns.boxplot('Type','K',data =dataset)



subplot(9,1,6)

sns.boxplot('Type','Ca',data =dataset)



subplot(9,1,7)

sns.boxplot('Type','Ba',data =dataset)



subplot(9,1,8)

sns.boxplot('Type','Fe',data =dataset)



subplot(9,1,9)

sns.boxplot('Type','RI',data =dataset)
from sklearn.neighbors import KNeighborsClassifier

x=dataset.iloc[:,0:9].values

y=dataset['Type'].values



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

knn=KNeighborsClassifier()

knn.fit(x_train,y_train)

knn.score(x_test,y_test)