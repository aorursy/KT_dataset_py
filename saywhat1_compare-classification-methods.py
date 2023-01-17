#Seaborn throws a lot of warnings, so we're going to ignore them.
import warnings
warnings.filterwarnings("ignore");

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../input/Iris.csv', index_col = 'Id')
df['SepalAreaCm'] = df.SepalLengthCm * df.SepalWidthCm
df['PetalAreaCm'] = df.PetalLengthCm * df.PetalWidthCm
df.Species = df.Species.astype('category')
sns.lmplot(x = 'SepalAreaCm', y = 'PetalAreaCm', data = df, hue = 'Species', fit_reg=False);
df[['SepalAreaCm', 'PetalAreaCm', 'Species']].groupby('Species').mean()
X = df.iloc[:,[0,1,2,3,5,6]]
Y = df.Species.cat.codes

logreg = LogisticRegression(C=1e5).fit(X,Y)

kfold = cross_validation.KFold(len(Y), n_folds=30)
lrScores = cross_validation.cross_val_score(logreg, X, Y, cv=kfold)
print('Score for cross validation: {:.2%}'.format(lrScores.mean()))
X_sep = df.iloc[:,[0,1]]

logregS = LogisticRegression(C=1e5).fit(X_sep,Y)

lrsScores = cross_validation.cross_val_score(logregS, X_sep, Y, cv=kfold)
print('Score for cross validation: {:.2%}'.format(lrsScores.mean()))
X_ped = df.iloc[:,[2,3]]

logregP = LogisticRegression(C=1e5).fit(X_ped,Y)

lrpScores = cross_validation.cross_val_score(logregP, X_ped, Y, cv=kfold)
print('Score for cross validation: {:.2%}'.format(lrpScores.mean()))
X_are = df.iloc[:,[5,6]]

logregA = LogisticRegression(C=1e5).fit(X_are,Y)

lraScores = cross_validation.cross_val_score(logregA, X_sep, Y, cv=kfold)
print('Score for cross validation: {:.2%}'.format(lraScores.mean()))
rf = RandomForestClassifier().fit(X,Y)

rfScores = cross_validation.cross_val_score(rf, X, Y, cv=kfold)
print('Score for cross validation where: {:.2%}'.format(rfScores.mean()))