import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
%matplotlib inline
df = pd.read_csv('../input/Iris.csv')
df.sample(10).sort_index()
df.drop('Id',axis=1, inplace=True)
df.shape
sns.pairplot(df, hue = 'Species')
g = sns.PairGrid(df, hue = 'Species',palette='Set2')
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.regplot)
g.map_upper(sns.kdeplot)
sns.pairplot(df,x_vars=['SepalLengthCm','SepalWidthCm'], 
             y_vars=['PetalLengthCm','PetalWidthCm'],hue='Species',palette='Set2')
sns.heatmap(df.select_dtypes(include='float'))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',center=0)
