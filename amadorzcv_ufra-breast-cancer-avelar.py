# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/data.csv')
train_df.head()
list(train_df.columns.values)
train_df.isna().sum()
train_df=train_df.drop(['Unnamed: 32'],axis=1)
train_df.isna().sum()
from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()

train_df.diagnosis = labelEncoder.fit_transform(train_df.diagnosis)

train_df.head()
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'symmetry_se', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'symmetry_mean', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'radius_mean', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'radius_worst', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'radius_se', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'fractal_dimension_worst', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'fractal_dimension_se', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'area_mean', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'area_worst', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'area_se', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'texture_mean', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'texture_worst', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'texture_se', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'perimeter_mean', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'perimeter_worst', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'perimeter_se', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'smoothness_mean', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'smoothness_worst', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'smoothness_se', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'compactness_mean', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'compactness_worst', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'compactness_se', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'concavity_mean', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'concavity_worst', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'concavity_se', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'concave points_mean', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'concave points_worst', bins=20)
g = sns.FacetGrid( train_df, col='diagnosis')

g.map(plt.hist, 'concave points_se', bins=20)
train_df=train_df.drop(

['id',

  'symmetry_mean',

  'symmetry_se',

  'texture_se',

  'compactness_se',

  'concavity_se',

  'smoothness_mean',

  'smoothness_worst',

  'smoothness_se',

  'fractal_dimension_se',

  'radius_se'],axis=1)



train_df.head()
classe = train_df['diagnosis']

atributos = train_df.drop('diagnosis', axis=1)

atributos.head()
from sklearn.model_selection import train_test_split

atributos_train, atributos_test, class_train, class_test = train_test_split(atributos, classe, test_size = 0.25 )



atributos_train.describe()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, random_state =0)

model = dtree.fit(atributos_train, class_train)
from sklearn.metrics import accuracy_score

classe_pred = model.predict(atributos_test)

acc = accuracy_score(class_test, classe_pred)

print("My Decision Tree acc is {}".format(acc))