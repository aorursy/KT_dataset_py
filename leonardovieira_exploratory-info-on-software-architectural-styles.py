import os

import pandas as pd

from pandas import DataFrame

import numpy as np



data = pd.read_csv('../input/Data Set Aquisition (3).csv')

data.head()
from matplotlib.pyplot import pie, axis, show



JobExperience = data['Job Experience ?'].value_counts()

print(JobExperience)



pie(JobExperience, labels=JobExperience.index, autopct='%1.1f%%', radius=2);

show()
LastDegree = data['Last Degree?'].value_counts()

print(LastDegree)



pie(LastDegree, labels=LastDegree.index, autopct='%1.1f%%', radius=2);

show()
import seaborn.apionly as sns

%matplotlib inline

import matplotlib.pyplot as plt



plt.figure(figsize=(10,12))

Organization = sns.countplot(y="Organization?", data=data,

              order=data['Organization?'].value_counts().nlargest(50).index,

              palette='GnBu_d')

plt.show()
nonImportant = ['Timestamp', 'Your Good Name?', 'Organization?']

data.drop(nonImportant, axis=1, inplace=True)
data['Last Degree?'].replace(['BSC (CS or SE)', 'MS(CS or SE)', 'Phd (CS or SE)'], [1, 2, 3], inplace=True)

data['Job Experience ?'].replace(['Education', 'Software Industry', 'Other', 'Software Industry;Education', 'Education;Other', 'Software Industry;Education;Other', 'Software Industry;Other'], [1, 2, 3, 4, 5, 6, 7], inplace=True)

data.head()
def normalize(df):

    result = df.copy()

    for featureName in df.columns:

        maxValue = df[featureName].max()

        minValue = df[featureName].min()

        result[featureName] = (df[featureName] - minValue) / (maxValue - minValue)

    return result
nData = normalize(data)

nData.head()
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(nData.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)