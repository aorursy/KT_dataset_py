import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib

import warnings

warnings.filterwarnings('ignore')



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



from sklearn.linear_model import LassoCV, Lasso



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

dataset = pd.read_csv('/kaggle/input/london-bike-sharing-dataset/london_merged.csv').drop(['timestamp', 't1', 't2'], axis=1) # Dropping columns with negative values

dataset.head()
dataset_values = dataset.values # Extract values from our dataset.



X = dataset_values[:,0:7] # Input

Y = dataset_values[:,6] # Target



test = SelectKBest(score_func=chi2, k=3) # Extract features, setting k value equal to 3. 

fit = test.fit(X, Y) 

print(fit.scores_) # Scores for each feature

features = fit.transform(X) # Apply the transformation

print(features[0:5,:])
model = LogisticRegression(solver='lbfgs')

rfe = RFE(model, 3) # Select 3 features

fit = rfe.fit(X, Y)

print("Selected Features: %s" % fit.support_)

print("Feature Ranking: %s" % fit.ranking_)
reg = LassoCV()

reg.fit(X, Y)

print("Best alpha: %f" % reg.alpha_)

print("Best score: %f" %reg.score(X,Y))

coef = pd.Series(reg.coef_, index = list(dataset.columns.values))
imp_coef = coef.sort_values()

matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso")