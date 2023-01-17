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
from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import scipy

import numpy

import json

import sys

import csv

import os
print('matplotlib: {}'.format(matplotlib.__version__))

print('sklearn: {}'.format(sklearn.__version__))

print('scipy: {}'.format(scipy.__version__))

print('seaborn: {}'.format(sns.__version__))

print('pandas: {}'.format(pd.__version__))

print('numpy: {}'.format(np.__version__))

print('Python: {}'.format(sys.version))
sns.set(style='white', context='notebook', palette='deep')

pylab.rcParams['figure.figsize'] = 12,8

warnings.filterwarnings('ignore')

mpl.style.use('ggplot')

sns.set_style('white')

%matplotlib inline
# import train and test to play with it

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
type(df_train)
type(df_test)
df_train.head()
# Modify the graph above by assigning each species an individual color.

g = sns.FacetGrid(df_train, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"seagreen", 0:"gray"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
df_train.plot(kind='scatter', x='Age', y='Fare',alpha = 0.5,color = 'red')
#show scatter plot with using Matplotlib

plt.figure(figsize=(8,6))

plt.scatter(range(df_train.shape[0]), np.sort(df_train['Age'].values))

plt.xlabel('index')

plt.ylabel('Age')

plt.title('Explore: Age')

plt.show()
ax= sns.boxplot(x="Pclass", y="Age", data=df_train)

ax= sns.stripplot(x="Pclass", y="Age", data=df_train, jitter=True, edgecolor="gray")

plt.show()
# histograms

df_train.hist(figsize=(15,20));

plt.figure();
df_train["Age"].hist();
df_train.Age.plot(kind = 'hist',bins = 5);
f,ax=plt.subplots(1,2,figsize=(20,10))

df_train[df_train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('Survived= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

df_train[df_train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')

ax[1].set_title('Survived= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=df_train,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

df_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
sns.countplot('Pclass', hue='Survived', data=df_train)

plt.title('Pclass: Sruvived vs Dead')

plt.show()
# scatter plot matrix

pd.plotting.scatter_matrix(df_train,figsize=(10,10))

plt.figure();
# violinplots on petal-length for each species

sns.violinplot(data=df_train,x="Sex", y="Age")
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=df_train,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=df_train,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
# Using seaborn pairplot to see the bivariate relation between each pair of features

sns.pairplot(data=df_train[["Fare","Survived","Age","Pclass"]],

             hue="Survived", dropna=True);
sns.FacetGrid(df_train, hue="Survived", size=5).map(sns.kdeplot, "Fare").add_legend()

plt.show();
sns.jointplot(x='Fare',y='Age',data=df_train);
sns.jointplot(x='Fare',y='Age' ,data=df_train, kind='reg');
sns.swarmplot(x='Pclass',y='Age',data=df_train);
plt.figure(figsize=(7,4)) 

sns.heatmap(df_train.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

plt.show();
plt.imshow(df_train.corr(), cmap='hot', interpolation='nearest')

plt.show()
df_train['Pclass'].value_counts().plot(kind="bar");
sns.factorplot('Pclass','Survived',hue='Sex',data=df_train)

plt.show();
sns.factorplot('SibSp','Survived',hue='Pclass',data=df_train)

plt.show()
#let's see some others factorplot

f,ax=plt.subplots(1,2,figsize=(20,8))

sns.barplot('SibSp','Survived', data=df_train,ax=ax[0])

ax[0].set_title('SipSp vs Survived in BarPlot')

sns.factorplot('SibSp','Survived', data=df_train,ax=ax[1])

ax[1].set_title('SibSp vs Survived in FactorPlot')

plt.close(2)

plt.show();
f,ax=plt.subplots(1,3,figsize=(20,8))

sns.distplot(df_train[df_train['Pclass']==1].Fare,ax=ax[0])

ax[0].set_title('Fares in Pclass 1')

sns.distplot(df_train[df_train['Pclass']==2].Fare,ax=ax[1])

ax[1].set_title('Fares in Pclass 2')

sns.distplot(df_train[df_train['Pclass']==3].Fare,ax=ax[2])

ax[2].set_title('Fares in Pclass 3')

plt.show()