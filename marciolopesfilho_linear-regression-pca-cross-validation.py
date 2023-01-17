import numpy as np

import pandas as pd 

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor as NN

from sklearn.model_selection import train_test_split,KFold

from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

import statsmodels.formula.api as smf
fl_moscow = pd.read_csv('../input/price-of-flats-in-moscow/flats_moscow.csv')

fl_moscow.dtypes

fl_moscow = fl_moscow.drop("Unnamed: 0", axis=1)

fl_moscow.head()
fl_moscow.iloc[:,6:10] = fl_moscow.iloc[:,6:10].astype('category')

fl_moscow.iloc[:,0:6] = fl_moscow.iloc[:,0:6].astype('float')

fl_moscow.dtypes

price = fl_moscow['price']

Xs = fl_moscow.drop('price', axis=1)

scal = StandardScaler()

scal.fit(Xs[['totsp','livesp','kitsp','dist','metrdist']])

Xs[['totsp','livesp','kitsp','dist','metrdist']] = scal.transform(Xs[['totsp','livesp','kitsp','dist','metrdist']])

Xs_train, Xs_test, price_train, price_test = train_test_split(Xs, price, test_size=.7, random_state = 49)

price_fl_moscow = linear_model.LinearRegression()

price_fl_moscow.fit(Xs_train,price_train)

price_fl_moscow.score(Xs_test,price_test)

forc = smf.ols('price ~ totsp + livesp + kitsp + dist + metrdist + floor + code + walk', data = fl_moscow).fit()

print(forc.summary())
#Scorers

r2 = make_scorer(r2_score, greater_is_better = True)

mae = make_scorer(mean_absolute_error, greater_is_better = False)

mqe = make_scorer(mean_squared_error, greater_is_better = False)
flow = Pipeline([('zscores', StandardScaler()),

                 ('pca', PCA()),

                 ('regNN', NN())])

cv_nn = GridSearchCV(

    estimator= flow,

    scoring = r2,

    param_grid = {

        'regNN__n_neighbors':range(1,35),

        'regNN__weights':['uniform','distance'],

        'regNN__p':[1,2],

        'pca__n_components':[1,2,3]

    }

)

cv_nn.fit(Xs_train,price_train)

print(cv_nn.best_estimator_.get_params())
NN_test = cv_nn.predict(Xs_test)
lr_pred = price_fl_moscow.predict(Xs_test)
price_test
r2_score(price_test,NN_test)
r2_score(price_test,lr_pred)
#Without PCA

flow2 = Pipeline([('zscores', StandardScaler()),

                 ('regNN', NN())])

cv_nn2 = GridSearchCV(

    estimator= flow2,

    scoring = mqe,

    param_grid = {

        'regNN__n_neighbors':range(1,35),

        'regNN__weights':['uniform','distance'],

        'regNN__p':[1,2]

    }

)

cv_nn2.fit(Xs_train,price_train)

print(cv_nn2.best_estimator_.get_params())
flow = Pipeline([('zscores', StandardScaler()),

                 ('pca', PCA()),

                 ('LinReg', linear_model.LinearRegression())])

cv_lr = GridSearchCV(

    estimator= flow,

    scoring = r2,

    param_grid = {

        'LinReg__normalize':[0,1],

        'pca__n_components':[1,2,3]

    }

)
cv_lr.fit(Xs_train,price_train)
cv_lr.predict(Xs_test)
cv_lr.best_score_