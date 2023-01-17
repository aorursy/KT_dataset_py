# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.cross_validation import train_test_split

import xlrd
xlrd.__VERSION__

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
F_Path = "../input/diabetes.csv"
df = pd.read_csv(F_Path)
#Get Rid of any NaN
df =df.dropna(thresh=1)

#Plot an analyze the dataset for any correlation
correlations = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels('DATA')
ax.set_yticklabels('DATA')
plt.show()

#Remove Skin Column
df =df.drop('SkinThickness', 'columns')

#The Amount of True
true =(df['Outcome']==1).sum()
false =(df['Outcome']==0).sum()
percentTrue = (true / (true + false))*100
print('The amount of true: ')
print(true)
print('The amount of false:')
print(false)
print('percent true: ')
print(percentTrue)

train,test = train_test_split(df,test_size=0.3)
print(train)
print(test)
X_test = test.drop('Outcome','columns')
Y_test = test['Outcome']

X_train = train.drop('Outcome', 'columns')
Y_train = train['Outcome']

df =df.drop(df.loc[df['Insulin']==0].index)
# Any results you write to the current directory are saved as output.
df =df.drop(df.loc[df['Insulin']==0].index)
df =df.drop(df.loc[df['BloodPressure']==0].index)
df =df.drop(df.loc[df['Age']==0].index)
df =df.drop(df.loc[df['Glucose']==0].index)

tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
tree.fit(X_train, Y_train)
tree.predict([[5, 121, 72, 112, 26.2, 0.245, 30]])
#it predicted a false and false was what was expected


