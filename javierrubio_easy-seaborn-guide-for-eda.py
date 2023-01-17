# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt # Seaborn is build over matplotlib and you need some adjustments from it because is more flexible.
from matplotlib.pyplot import figure, show  #To set the figure size
sns.set(style='darkgrid', context='notebook', palette='deep', # Here you can set up the style of your graphs, this is my prefered one.
        font_scale=1.3) # This last parameter is important control the font size
%matplotlib inline

import warnings # To avoid warnings
warnings.filterwarnings('ignore')
# import train and test to play with it
df = pd.read_csv('../input/train.csv')
df = df.dropna(axis=0, subset=['Age']) #drop Age nulls for representation propose
# seaborn histogram
figure(figsize=(10,5)) # figsize=(width,height)
ax = sns.distplot(df.Fare, hist=True, kde=False,bins=100,color = 'blue' ,hist_kws={'edgecolor':'black'})

# Add labels
ax.set(xlabel='Fare', ylabel='Number of samples',title="Fare distribution")
show()
## logx=True to provide logaritmic axis
# An easy way to provide everything in a single command
df.hist(figsize=(15,15))
show;
figure(figsize=(10,5))
ax = sns.countplot(data=df,x=df.Pclass)
ax.set(ylabel='samples',title="Passengers class distribution");
figure(figsize=(10,6))
ax = sns.regplot(x="Age", y="Fare", color= 'green',data=df)
ax.set(title="Relation between Age and Fare");

## logx=True to provide logaritmic axis
ax = sns.catplot(x="Sex", y="Survived", kind="bar", data=df, height=6, aspect=1)
ax.set(title="Relation between Gender and Survived", xlabel='Gender');
sns.catplot(x="Pclass", y="Age", data=df, height=6, aspect=1);
grid = sns.FacetGrid(df, col='Survived', row='Sex', size=3.5, aspect=1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
sns.catplot(x="Pclass", y="Age", hue="Sex", kind="swarm", data=df, height=6, aspect=1);
figure(figsize=(15,6))
sns.heatmap(df.corr(), annot=True);