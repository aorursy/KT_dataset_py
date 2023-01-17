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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm
from pandas.tools.plotting import parallel_coordinates
%matplotlib inline
train_data = pd.read_csv("../input/train.csv")
train_data.columns
train_data.head()
train_data.dtypes
train_data.drop(['PassengerId','Ticket','Cabin'], axis=1, inplace = True)

train_data.isnull().sum()
 #complete missing age with mean
train_data['Age'].fillna(train_data['Age'].mean(), inplace = True)
#complete embarked with mode
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace = True)
train_data.isnull().sum()
train_data['survived_dead'] = train_data['Survived'].apply(lambda x : 'Survived' if x == 1 else 'Dead')
train_data.describe()
sns.clustermap(data = train_data.corr().abs(),annot=True, fmt = ".2f", cmap = 'Blues')
sns.countplot('survived_dead', data = train_data)
sns.countplot( train_data['Sex'],data = train_data, hue = 'survived_dead', palette='coolwarm')
sns.countplot( train_data['Pclass'],data = train_data, hue = 'survived_dead')
sns.barplot(x = 'Pclass', y = 'Fare', data = train_data)
sns.pointplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = train_data);
sns.barplot(x  = 'Embarked', y = 'Fare', data = train_data)
g = sns.FacetGrid(train_data, hue='Survived')
g.map(sns.kdeplot, "Age",shade=True)
sns.catplot(x="Embarked", y="Survived", hue="Sex",
            col="Pclass", kind = 'bar',data=train_data, palette = "rainbow")
sns.catplot(x='SibSp', y='Survived',hue = 'Sex',data=train_data, kind='bar')
sns.catplot(x='Parch', y='Survived',hue = 'Sex',data=train_data, kind='point')
g= sns.FacetGrid(data = train_data, row = 'Sex', col = 'Pclass', hue = 'survived_dead')
g.map(sns.kdeplot, 'Age', alpha = .75, shade = True)
plt.legend()