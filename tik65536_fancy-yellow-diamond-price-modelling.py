%config IPCompleter.greedy=True

%matplotlib inline

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

from statsmodels.formula.api import ols

from statsmodels.stats.anova import anova_lm

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
raw_data=pd.read_csv('../input/allbatch_cost_modify.csv')

option=['NFY','NFLY']

raw_data=raw_data.loc[raw_data['COLOR_SHORT'].isin(option)]
#separate the column by numerical and category in list for future use

cat_column=['SHAPE','COLOR_SHORT','CLARITY','POL','SYM','FLOURO']

num_column=['Cash','SIZE']

#check each category variable 

raw_data.SHAPE.unique()

raw_data.COLOR_SHORT.unique()

raw_data.CLARITY.unique()

raw_data.POL.unique()

raw_data.SYM.unique()

raw_data.FLOURO.unique()

raw_data.describe() # description statistic for numerical data
b=sns.pairplot(raw_data)
#raw_data=raw_data.drop(columns=['VISA','Unionpay','ListPrice'])

raw_data['Cash']= np.log(raw_data['Cash'])

raw_data['SIZE']= np.log(raw_data['SIZE'])

b=sns.pairplot(raw_data)

corr=raw_data.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr,mask=mask, cmap=cmap,linewidths=.5 ,center=0,square=True, cbar_kws={"shrink": 1})
fig, ax = plt.subplots(6,2,figsize=(15,15))

ax[0][0]=sns.catplot('SHAPE',data=raw_data,kind='count',ax=ax[0][0])

ax[0][1]=sns.catplot('COLOR_SHORT',data=raw_data,kind='count',ax=ax[0][1])

ax[1][0]=sns.catplot('CLARITY',data=raw_data,kind='count',ax=ax[1][0])

ax[1][1]=sns.catplot('POL',data=raw_data,kind='count',ax=ax[1][1])

ax[2][0]=sns.catplot('SYM',data=raw_data,kind='count',ax=ax[2][0])

ax[2][1]=sns.catplot('FLOURO',data=raw_data,kind='count',ax=ax[2][1])

ax[3][0]=sns.boxplot(x="COLOR_SHORT", y="Cash", data=raw_data,ax=ax[3][0])

ax[3][1]=sns.boxplot(x="CLARITY", y="Cash", data=raw_data,ax=ax[3][1])

ax[4][0]=sns.boxplot(x="POL", y="Cash", data=raw_data,ax=ax[4][0])

ax[4][1]=sns.boxplot(x="SYM", y="Cash", data=raw_data,ax=ax[4][1])

ax[5][0]=sns.boxplot(x="FLOURO", y="Cash", data=raw_data,ax=ax[5][0])

ax[5][1]=sns.boxplot(x="SHAPE", y="Cash", data=raw_data,ax=ax[5][1])

fig.tight_layout()

plt.close(2);plt.close(3);plt.close(4);plt.close(5);plt.close(6);plt.close(7)
b=sns.relplot(x="SIZE", y="Cash", hue="COLOR_SHORT",alpha=.9, palette="muted", height=8, data=raw_data)
b=sns.relplot(x="SIZE", y="Cash", hue="SHAPE",alpha=.9, palette="muted", height=8, data=raw_data)
b=sns.relplot(x="SIZE", y="Cash", hue="CLARITY",alpha=.9, palette="muted", height=8, data=raw_data)
b=sns.relplot(x="SIZE", y="Cash", hue="POL",alpha=.9, palette="muted", height=8, data=raw_data)
b=sns.relplot(x="SIZE", y="Cash", hue="SYM",alpha=.9, palette="muted", height=8, data=raw_data)
b=sns.relplot(x="SIZE", y="Cash", hue="FLOURO",alpha=.9, palette="muted", height=8, data=raw_data)
model1=ols('Cash~SIZE+C(COLOR_SHORT)+C(SHAPE)+C(CLARITY)+C(FLOURO)+C(SYM)+C(POL)',data=raw_data).fit()

model1.summary()

b=sns.lmplot(x="SIZE", y="Cash",hue="COLOR_SHORT", data=raw_data)
model2=ols('Cash~SIZE+C(COLOR_SHORT)+C(SHAPE)+C(CLARITY)+C(POL)',data=raw_data).fit()

model2b=ols('Cash~SIZE+C(COLOR_SHORT)+C(SHAPE)+C(CLARITY)+C(POL)+C(SYM)+C(FLOURO)',data=raw_data).fit()

anova_lm(model2,model2b)
model2=ols('Cash~SIZE+C(COLOR_SHORT)+C(SHAPE)+C(CLARITY)+C(POL)',data=raw_data).fit()

model2.summary()
from sklearn.cluster import KMeans

X=np.array([0.6516,0.5748,0.5310,0.4927,0.4928,0.4548,0.3668])

kmeans = KMeans(n_clusters=3)

kmeans.fit_predict(X.reshape(-1,1))

kmeans.cluster_centers_[:, 0]
fig = plt.figure()

ax = plt.axes()

x = np.linspace(0.01 , 5 ,100)

ax.plot(x, 1.5678*np.power(x,0.5678))