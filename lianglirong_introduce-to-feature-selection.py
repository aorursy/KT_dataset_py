import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import os
print(os.listdir("../input"))

data = pd.read_csv("../input/pima_data.csv")
array = data.values
X = array[:,0:8]
Y = array[:,8] 
#selcet the best four features
test = SelectKBest(score_func=chi2,k=4)
fit = test.fit(X,Y)
print(fit.scores_)
features = fit.transform(X)
print(features)
#RFE
model = LogisticRegression()
rfe = RFE(model,3)
fit = rfe.fit(X,Y)
print("n_features_：",fit.n_features_)
print("support_:",fit.support_)
print("ranking:",fit.ranking_)

#PCA
pca = PCA(n_components=3)
fig = pca.fit(X)
fig.components_.shape
