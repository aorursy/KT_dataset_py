import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest

import os
print(os.listdir("../input"))
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv("../input/pima_data.csv",names=names)

array = data.values
X = array[:,0:8]
Y = array[:,8]
steps = []
steps.append(('Standardize', StandardScaler()))
steps.append(('logistic', LogisticRegression()))
model = Pipeline(steps)

kfold = KFold(n_splits=10,random_state=2)

result = cross_val_score(model,X,Y,cv=kfold)
print("result:",result.mean())
steps = []
features = []
features.append(('pca', PCA()))
features.append(('select_best', SelectKBest(k=6)))

steps.append(('feature_union',FeatureUnion(features)))
steps.append(('logistic',LogisticRegression()))
model = Pipeline(steps)

kfold = KFold(n_splits=10,random_state=2)
result = cross_val_score(model,X,Y,cv=kfold)
print("result:",result.mean())

