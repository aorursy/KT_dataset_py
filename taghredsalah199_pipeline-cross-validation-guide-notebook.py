import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_happ2019= pd.read_csv('../input/world-happiness/2019.csv')

df_happ2019.head()
plt.figure(figsize=(10,10))

sns.heatmap(df_happ2019.corr(),annot=True)
sns.clustermap(df_happ2019.corr())
sns.lmplot(x='Perceptions of corruption',y='Healthy life expectancy',data=df_happ2019)
x= df_happ2019['Freedom to make life choices']

y= df_happ2019['Score']

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.kdeplot(x, y, cmap=cmap, shade=True);
sns.jointplot(x='GDP per capita',y='Social support',data=df_happ2019,kind='hex')
sns.jointplot(x='Healthy life expectancy',y='Social support',data=df_happ2019,kind='reg')
df_happ2019=df_happ2019.drop('Country or region', axis=1)
plt.figure(figsize=(10,10))

sns.heatmap(df_happ2019.isnull(),cmap="YlGnBu")
X= df_happ2019.drop('Score',axis=1)

y=df_happ2019['Score']
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVR

from sklearn.decomposition import PCA

my_pipeline= make_pipeline(PCA(),SVR())
from sklearn.model_selection import cross_val_score

scores= cross_val_score(my_pipeline,X,y,scoring='neg_mean_absolute_error')

scores