import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

data = pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")

X = data.iloc[:,0:20]  #independent columns

y = data.iloc[:,-1]    #target column i.e price range

data.head()
data.shape
# apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2 , k=10)

fit = bestfeatures.fit(X, y )
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)
# concatenate two dataframes for better visualizaton

featureScores = pd.concat([dfcolumns , dfscores] ,axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores
featureScores.nlargest(10 , 'Score')
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
import seaborn as sns

#get correlations of each features in dataset

corrmat = data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")