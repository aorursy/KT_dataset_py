import seaborn as sns

from sklearn import preprocessing

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt  

matplotlib.style.use('ggplot')

%matplotlib inline

import math

import matplotlib as mpl

import plotly

import colorsys

plt.style.use('seaborn-talk')

from mpl_toolkits.mplot3d import Axes3D

from __future__ import division

import pylab

import plotly.plotly as py

import plotly.graph_objs as go

from matplotlib import colors as mcolors

from scipy import stats

from sklearn import datasets

from sklearn import metrics

import types

from sklearn.manifold import TSNE

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline





import seaborn as sns

sns.set(style="whitegrid", palette="muted")

current_palette = sns.color_palette()
df =  pd.read_csv('../input/human-resources-data-set/core_dataset.csv')

df.describe()
df.info()
g = sns.factorplot("Age", data=df, aspect=4, kind="count")

g.set_xticklabels(rotation=90)

g = plt.title("Distribution of Ages Among Dental Magic Employees")
sns.countplot(df["Sex"])

g = plt.title("Gender")
df['Department'].value_counts()[0:20].plot(kind='bar',figsize=(10,8))
df['Reason For Term'].value_counts()[0:20].plot(kind='bar',figsize=(10,8))
df['Employee Source'].value_counts()[0:20].plot(kind='bar',figsize=(10,8))
pd.pivot_table(df,index=["Department"])
pd.pivot_table(df,index=["Employment Status"])
pd.pivot_table(df,index=["Reason For Term"])
employment_status = df.groupby(by='Employment Status').size().sort_values(ascending=False).head(10)
print (employment_status)
labels = 'Voluntarily Terminated', 'Terminated for Cause', 'Leave of Absence'

sizes = [88, 14, 14]

colors = ['skyblue', 'lightgreen', 'lightcoral']



plt.pie(sizes,               

        labels=labels,      

        colors=colors,      

        autopct='%1.1f%%',  

        startangle=30       

        )



plt.axis('equal')



plt.show()
plt.figure(figsize=(16,5))

sns.countplot('Manager Name', data = df)

plt.xticks(rotation = 45)

plt.tight_layout()
plt.figure(figsize=(16,5))

sns.countplot('Position', data = df)

plt.xticks(rotation = 45)

plt.tight_layout()
clarity_color_table = pd.crosstab(index=df["Department"], 

                          columns=df["MaritalDesc"])



clarity_color_table.plot(kind="bar", 

                 figsize=(10,5),

                 stacked=True)
clarity_color_table = pd.crosstab(index=df["Department"], 

                          columns=df["Sex"])



clarity_color_table.plot(kind="bar", 

                 figsize=(10,5),

                 stacked=True)
plt.figure(figsize=(16,5))

sns.countplot('Date of Hire', data = df)

plt.xticks(rotation = 45)

plt.tight_layout()