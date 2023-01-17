import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.cluster import KMeans

from sklearn.pipeline import Pipeline

import gzip
data = pd.read_csv('../input/StudentsPerformance.csv')
data.info()
data.head()
data.describe()
sns.countplot(x = 'gender', data = data)
sns.countplot(x = 'race/ethnicity', data = data)
sns.countplot(x = 'parental level of education', data = data)
sns.countplot(x = 'lunch', data =data)
sns.countplot(x = 'test preparation course', data = data)
sns.countplot(x = 'test preparation course', hue = 'gender', data = data)
sns.countplot(x = 'test preparation course', hue = 'race/ethnicity', data = data)
sns.countplot(x = 'test preparation course', hue = 'lunch', data = data)
sns.boxplot(x='gender',y='math score',data=data)
sns.boxplot(x='race/ethnicity',y='math score',data=data)
sns.boxplot(x='parental level of education',y='math score',data=data)
sns.boxplot(x='lunch',y='math score',data=data)
sns.boxplot(x='test preparation course',y='math score',data=data)
sns.countplot(x = 'race/ethnicity', hue = 'lunch', data = data)