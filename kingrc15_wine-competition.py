import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import os
raw_wine_data = pd.read_csv('/kaggle/input/tdac-wine/Test_Data.csv')
raw_wine_data.head()
wine_target = raw_wine_data['type']

raw_wine_data = raw_wine_data.drop(['index','type'], axis=1)
vals = raw_wine_data.values

min_max_scaler = preprocessing.MinMaxScaler()

vals_scaled = min_max_scaler.fit_transform(vals)

processed_wine_data = pd.DataFrame(vals_scaled)
from sklearn.manifold import TSNE



decomp_wine = TSNE(n_components=2, early_exaggeration=2.0).fit_transform(processed_wine_data)
decomp_wine = pd.DataFrame(decomp_wine)
red_wine = decomp_wine[wine_target == 0]

white_wine = decomp_wine[wine_target == 1]



fig,ax=plt.subplots(1,1,figsize=(10, 10))

red_wine.plot.scatter(0,1, color='red', ax=ax, label='Red Wine')

white_wine.plot.scatter(0,1, color='blue', ax=ax, label='White Wine')
parameters_logit= [{'C':[0.1,0.2,0.5],'solver':['liblinear'],'penalty':['l1','l2'],'max_iter':[1000]},

                   {'C':[0.1,0.2,0.5,1],'solver':['lbfgs'],'penalty':['l2'],'max_iter':[1000]}]
X_train, X_test, y_train, y_test = train_test_split(processed_wine_data, wine_target)

LR = LogisticRegression()

grid_search_logit=GridSearchCV(estimator=LR, param_grid=parameters_logit,scoring='accuracy',cv=10)

grid_search_logit.fit(X_train,y_train)

grid_search_logit.score(X_test, y_test)
raw_wine_val_data = pd.read_csv('/kaggle/input/tdac-wine/Val_Data.csv')
del raw_wine_val_data['Index']

vals = raw_wine_val_data.values

min_max_scaler = preprocessing.MinMaxScaler()

vals_scaled = min_max_scaler.fit_transform(vals)

processed_wine_val_data = pd.DataFrame(vals_scaled)

processed_wine_val_data.head()
guess = grid_search_logit.predict(processed_wine_val_data)

np.savetxt("val.csv", guess, delimiter=",")