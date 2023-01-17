# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report , roc_auc_score, roc_curve, precision_recall_curve

from sklearn.feature_selection import RFE

import time

from pprint import pprint

from tabulate import tabulate

from sklearn.tree import export_graphviz

import eli5

from eli5.sklearn import PermutationImportance
df=pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df['age'].value_counts()
df = df[df['thal'] != 0]

df = df[df['age'] != 29]

df.head()    
df['thal'].value_counts()
df['thal'] = df['thal'].replace(1, 'fixed defect')

df['thal'] = df['thal'].replace(2, 'normal')

df['thal'] = df['thal'].replace(3, 'reversable defect')

df['cp'] = df['cp'].replace(0, 'asymptomatic')

df['cp'] = df['cp'].replace(1, 'atypical angina')

df['cp'] = df['cp'].replace(2, 'non-anginal pain')

df['cp'] = df['cp'].replace(3, 'typical angina')

df['restecg'] = df['restecg'].replace(0, 'ventricular hypertrophy')

df['restecg'] = df['restecg'].replace(1, 'normal')

df['restecg'] = df['restecg'].replace(2, 'ST-T wave abnormality')

df['slope'] = df['slope'].replace(0, 'downsloping')

df['slope'] = df['slope'].replace(1, 'flat')

df['slope'] = df['slope'].replace(2, 'upsloping')

temp = pd.get_dummies(df[['cp', 'restecg', 'slope', 'thal']])

df = df.join(temp, how='left')

df = df.drop(columns = ['cp','restecg', 'slope', 'thal'], axis=1)

df.head()
df=df.drop_duplicates()
df.corr()
sns.heatmap(df)
sns.catplot('age',kind='count',hue='target',data=df,height=10)
sns.countplot(df.target)

df.target.value_counts()