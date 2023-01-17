# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

test= pd.read_csv("../input/test.csv")

train= pd.read_csv("../input/train.csv")

train = np.array(train) 



train

number_passengers = np.size(train[0::,1].astype(np.float))

number_passengers 
thirdClass = train[0::,2].astype(np.float) == 3

thirdClasssize = np.size(train[thirdClass,2].astype(np.float) == 3)

thirdClasssize

thirdclassSurvived=np.sum(train[thirdClass,1].astype(np.float) == 1)

thirdclassSurvived
survived = np.sum(train[0::,1].astype(np.float))

survived
ThirdClassPercentageSurvived = thirdclassSurvived / thirdClasssize * 100

ThirdClassPercentageSurvived
firstClass = train[0::,2].astype(np.float) == 1

firstClasssize = np.size(train[firstClass,2].astype(np.float) == 1)

firstClasssize
firstclassSurvived=np.sum(train[firstClass,1].astype(np.float) == 1)

firstclassSurvived
FirstClassPercentageSurvived = firstclassSurvived / firstClasssize * 100

train= pd.read_csv("../input/train.csv")

FirstClassPercentageSurvived
train_df= pd.read_csv("../input/train.csv")

print(pd.crosstab(train_df.Pclass, train_df.Survived, margins=True))

print(pd.crosstab(train_df.Pclass, train_df.Survived, margins=True).apply(lambda r: 2*r/r.sum(), axis=1))
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

g = sns.FacetGrid(train_df, col='Pclass')

g.map(plt.hist, 'Survived', bins=10)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', hue_order=["female", "male"])

grid.add_legend()