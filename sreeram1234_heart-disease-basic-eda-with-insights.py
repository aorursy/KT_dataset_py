# importing libraries.!!



import pandas as pd

import numpy as np

import os

import seaborn as sns

from matplotlib import pyplot as plt



import warnings

warnings.filterwarnings('ignore')
# loading data



data = pd.read_csv('../input/heart.csv')
# Let's take a look into the data



data.head()
print(f'Data have {data.shape[0]} rows and {data.shape[1]} columns')
# distribution of the features in the dataset



data.describe().T
# check the target distribution



plt.rcParams['figure.figsize'] = (8, 6)



sns.countplot(x='target', data=data);
data.groupby(by=['sex', 'target'])['target'].count()
pd.crosstab(data['sex'], data['target'])
sns.catplot(x='sex', col='target', kind='count', data=data);
print("% of women suffering from heart disease: " , data.loc[data.sex == 0].target.sum()/data.loc[data.sex == 0].target.count())

print("% of men suffering from heart disease:   " , data.loc[data.sex == 1].target.sum()/data.loc[data.sex == 1].target.count())
f,ax=plt.subplots(1,2,figsize=(16,7))



data.loc[data['sex']==1, 'target'].value_counts().plot.pie(explode=[0,0.10],autopct='%1.1f%%',ax=ax[0],shadow=True)

data.loc[data['sex']==0, 'target'].value_counts().plot.pie(explode=[0,0.10],autopct='%1.1f%%',ax=ax[1],shadow=True)



ax[0].set_title('Patients (male)')

ax[1].set_title('Patients (female)')



plt.show()
data.groupby(by=['cp', 'target'])['target'].count()
pd.crosstab(data['cp'], data['target']).style.background_gradient(cmap='autumn_r')
sns.catplot(x='cp', col='target', kind='count', data=data);
data.groupby(by=['fbs', 'target'])['target'].count()
sns.catplot(x='fbs', col='target', kind='count', data=data);
data.groupby(by=['restecg', 'target'])['target'].count()
sns.catplot(x='restecg', col='target', kind='count', data=data);
data.groupby(by=['exang', 'target'])['target'].count()
sns.catplot(x='exang', col='target', kind='count', data=data);
data.groupby(by=['slope', 'target'])['target'].count()
sns.catplot(x='slope', col='target', kind='count', data=data);
data.groupby(by=['ca', 'target'])['target'].count()
sns.catplot(x='ca', col='target', kind='count', data=data);
data.groupby(by=['thal', 'target'])['target'].count()
sns.catplot(x='thal', col='target', kind='count', data=data);
sns.distplot(a=data['age'], color='black');
sns.boxplot(x=data['target'], y=data['age']);
sns.distplot(data['trestbps']);
sns.boxplot(x=data['target'], y=data['trestbps']);
sns.distplot(data['thalach'], color='black');
sns.boxplot(x=data['target'], y=data['thalach']);
sns.distplot(data['chol']);
sns.boxplot(x='target', y='chol', data=data);
sns.scatterplot(x='chol', y='thalach', data=data, hue='target');
sns.scatterplot(x='chol', y='age', data=data, hue='target');
sns.scatterplot(x='chol', y='trestbps', data=data, hue='target');
sns.scatterplot(x='chol', y='oldpeak', data=data, hue='target');
sns.pairplot(data[['chol', 'age', 'trestbps', 'thalach', 'target']], hue='target');
corr = data[['chol', 'age', 'trestbps', 'thalach', 'target']].corr()



cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, annot=True, linewidths=1.7, linecolor='white');