import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import scipy.stats as stats

train_full= pd.read_csv('../input/train.csv',index_col='PassengerId')

quantitative = [f for f in train_full.columns if train_full.dtypes[f] != 'object']
quantitative.remove('Survived')#Survived is target label
qualitative = [f for f in train_full.columns if train_full.dtypes[f] == 'object']
train_full.head()
train_full.info()
sns.distplot(train_full['Survived'],kde=False)
train_full['Survived'].describe()
f = pd.melt(train_full, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
sns.barplot(y='Survived',x='Pclass',data=train_full)
sns.barplot(y='Survived',x='Sex',data=train_full)
sns.barplot(y='Survived',x='Sex',hue='Pclass',data=train_full)
sns.barplot(y='Survived',x='Embarked',data=train_full)
sns.barplot(y='Survived',x='SibSp',data=train_full)
sns.barplot(y='Survived',x='Parch',data=train_full)
sns.boxplot(x='Survived',y='Age',hue='Sex',data=train_full)
sns.boxplot(x='Survived',y='Fare',hue='Sex',data=train_full)
plt.figure(figsize=(20, 7))
plt.subplot(1,2,1)
sns.barplot(x="Pclass", y="Survived", hue="SibSp", data=train_full)
plt.subplot(1,2,2)
sns.barplot(x="Pclass", y="Survived", hue="Parch", data=train_full)
def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['Survived'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    
features = quantitative
spearman(train_full, features)
def anova(frame):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():#for each level of the category c
            s = frame[frame[c] == cls]['Survived'].values
            samples.append(s)#Get all the values of SalePrice for the level cls of the categorical variable c, and append
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

a = anova(train_full)
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
sns.boxplot(x='Pclass',y='Age',hue='Sex',data=train_full)
sns.boxplot(x='Pclass',y='Fare',hue='Sex',data=train_full)
sns.heatmap(train_full.drop('Survived',axis=1).corr(),cmap=sns.diverging_palette(220, 20, n=10),annot=True,linewidths=0.1)
