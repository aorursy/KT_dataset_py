import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as mpl

from matplotlib import pyplot as plt
%matplotlib inline
df_iris = sns.load_dataset('iris')

df_iris.head()
sns.set_style('dark')
sns.relplot(x='sepal_length', y='sepal_width', data=df_iris)
df_iris['species'].unique()
sns.relplot(x='sepal_length', y='sepal_width', data=df_iris, hue='species'); # hint: semicolon
sns.set_style('whitegrid')
fmri = sns.load_dataset("fmri")

fmri.head()
fmri.region.unique()
sns.relplot(x='timepoint', y='signal', data=fmri, hue='region', kind='line', height=10);
sns.lineplot(x='timepoint', y='signal', data=fmri, hue='region') # No height parameter because it's too big

plt.legend(bbox_to_anchor=(1, 1), loc=2);
plt.rcParams["figure.figsize"] = (15,9) #(width, height)
sns.catplot(kind='box', data=df_iris);
sns.barplot(x='species', y='petal_width', data=df_iris)



plt.title('Average Petal Width of Irises')
sns.countplot(x='petal_width', data=df_iris)



plt.title('Petal Width Distribution');
sns.set_style('whitegrid')

sns.distplot(df_iris['sepal_length'], kde=True);
sns.barplot(x='species', y='petal_width', data=df_iris)

sns.despine() # This must be run after creating chart
sns.set_palette(sns.color_palette('colorblind'))

sns.barplot(x='species', y='petal_width', data=df_iris)