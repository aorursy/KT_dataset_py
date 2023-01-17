import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

data = pd.read_csv('../input/mobile-price-classification/train.csv')

data.head()
data.columns
X = data.iloc[:,0:20]

y = data.iloc[:,-1]
X
y
bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)
fit
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)
dfcolumns
dfscores
#Concat two dataframes for better visualization

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']
featureScores
print(featureScores.nlargest(10,'Score'))
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
#Correlation matrix

import seaborn as sns

corrmat = data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20, 20))



g = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap='RdYlGn')