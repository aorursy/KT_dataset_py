import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv("../input/kag_risk_factors_cervical_cancer.csv")





# process columns, apply object > numeric

for c in train.columns:

    if train[c].dtype == 'object':

        train[c] =pd.to_numeric(train[c], errors='coerce')

def plot_corr(df,size=10):

    import matplotlib.pyplot as plt

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot'''



    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns);

    plt.yticks(range(len(corr.columns)), corr.columns);



print(train.describe().T)

plot_corr(train)

new_col= train.groupby('Dx:Cancer').mean()

print(new_col.head().T)
def dddraw(X_reduced,name):

    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D

    # To getter a better understanding of interaction of the dimensions

    # plot the first three PCA dimensions

    fig = plt.figure(1, figsize=(8, 6))

    ax = Axes3D(fig, elev=-150, azim=110)

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired)

    titel="First three directions of "+name 

    ax.set_title(titel)

    ax.set_xlabel("1st eigenvector")

    ax.w_xaxis.set_ticklabels([])

    ax.set_ylabel("2nd eigenvector")

    ax.w_yaxis.set_ticklabels([])

    ax.set_zlabel("3rd eigenvector")

    ax.w_zaxis.set_ticklabels([])



    plt.show()
from sklearn.decomposition import PCA, FastICA,SparsePCA,NMF, LatentDirichletAllocation,FactorAnalysis

from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection

from sklearn.cluster import KMeans,Birch

import statsmodels.formula.api as sm

from scipy import linalg

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

import matplotlib.pyplot as plt



n_col=5

X = train.drop(['Dx:Cancer'],axis=1) 



def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



Y=train['Dx:Cancer']

X=X.fillna(value=0)       # those ? converted to NAN are bothering me abit...        

scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

poly = PolynomialFeatures(2)

X=poly.fit_transform(X)





names = [

         'PCA',

         'FastICA',

         'Gauss',

         'KMeans',

         'SparsePCA',

         'SparseRP',

         'Birch',

         'NMF',    

         'LatentDietrich',    

        ]



classifiers = [

    

    PCA(n_components=n_col),

    FastICA(n_components=n_col),

    GaussianRandomProjection(n_components=3),

    KMeans(n_clusters=24),

    SparsePCA(n_components=n_col),

    SparseRandomProjection(n_components=n_col, dense_output=True),

    Birch(branching_factor=10, n_clusters=12, threshold=0.5),

    NMF(n_components=n_col),    

    LatentDirichletAllocation(n_topics=n_col),

    

]

correction= [1,1,0,0,0,0,0,0,0]



temp=zip(names,classifiers,correction)

print(temp)



for name, clf,correct in temp:

    Xr=clf.fit_transform(X,Y)

    dddraw(Xr,name)

    res = sm.OLS(Y,Xr).fit()

    #print(res.summary())  # show OLS regression

    #print(res.predict(Xr).round()+correct)  #show OLS prediction

    #print('Ypredict',res.predict(Xr).round()+correct)  #show OLS prediction



    #print('Ypredict *log_sec',res.predict(Xr).round()+correct*Y.mean())  #show OLS prediction

    print(name,'%error',procenterror(res.predict(Xr)+correct*Y.mean(),Y),'rmsle',rmsle(res.predict(Xr)+correct*Y.mean(),Y))
from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.preprocessing import MinMaxScaler



# import some data to play with

n_col=36

X = train.drop(['Dx:Cancer'],axis=1) 

Y=train['Dx:Cancer']

X=X.fillna(value=0)       # those ? converted to NAN are bothering me abit...        



scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

poly = PolynomialFeatures(2)   #creating some polynomial features

X=poly.fit_transform(X)





def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,2)



names = [

         #'ElasticNet',

         'HuberRegressor',

         'Ridge',

         'Lasso',

         'LassoCV',

         'Lars',

         'BayesianRidge',

         'SGDClassifier',

         'RidgeClassifier',

         'LogisticRegression',

         'OrthogonalMatchingPursuit',

         #'RANSACRegressor',

         ]



classifiers = [

    #ElasticNetCV(cv=10, random_state=0),

    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),

    Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),

    Lasso(alpha=0.05),

    LassoCV(),

    Lars(n_nonzero_coefs=10),

    BayesianRidge(),

    SGDClassifier(),

    RidgeClassifier(),

    LogisticRegression(),

    OrthogonalMatchingPursuit(),

    #RANSACRegressor(),

]

correction= [0,0,0,0,0,0,0,0,0,0,0,0]



temp=zip(names,classifiers,correction)

print(temp)



for name, clf,correct in temp:

    regr=clf.fit(X,Y)

    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)

    print(name,'%error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))

    if procenterror(regr.predict(X),Y)<5:

        print(regr.predict(X).round())
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans



X = train.drop(['Dx:Cancer'],axis=1) 

Y=train['Dx:Cancer']

X=X.fillna(value=0)       # those ? converted to NAN are bothering me abit...        



scaler = MinMaxScaler()

scaler.fit(X)

poly = PolynomialFeatures(2)   #creating some polynomial features

X=poly.fit_transform(X)



Xtr, Xv, ytr, yv = train_test_split(X, Y, test_size=0.2, random_state=2017)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

#dtest = xgb.DMatrix(test1[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



#

xgb_pars = {'min_child_weight': 100, 'eta': 0.03, 'colsample_bytree': 0.3, 'max_depth': 3,

            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:logistic'}



# You could try to train with more epoch

model = xgb.train(xgb_pars, dtrain, 2000, watchlist, early_stopping_rounds=25,

                  maximize=False, verbose_eval=50)



 
#!/usr/bin/env python3

# -*- coding: utf-8 -*-



# Check this gist for xgboost wrapper: https://gist.github.com/slaypni/b95cb69fd1c82ca4c2ff

 

import sys

import math

 

import numpy as np

from sklearn.grid_search import GridSearchCV

 

sys.path.append('xgboost/wrapper/')

import xgboost as xgb

 

 

class XGBoostClassifier():

    def __init__(self, num_boost_round=10, **params):

        self.clf = None

        self.num_boost_round = num_boost_round

        self.params = params

        self.params.update({'objective': 'multi:softprob'})

 

    def fit(self, X, y, num_boost_round=None):

        num_boost_round = num_boost_round or self.num_boost_round

        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}

        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])

        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

 

    def predict(self, X):

        num2label = {i: label for label, i in self.label2num.items()}

        Y = self.predict_proba(X)

        y = np.argmax(Y, axis=1)

        return np.array([num2label[i] for i in y])

 

    def predict_proba(self, X):

        dtest = xgb.DMatrix(X)

        return self.clf.predict(dtest)

 

    def score(self, X, y):

        Y = self.predict_proba(X)

        return 1 / logloss(y, Y)

 

    def get_params(self, deep=True):

        return self.params

 

    def set_params(self, **params):

        if 'num_boost_round' in params:

            self.num_boost_round = params.pop('num_boost_round')

        if 'objective' in params:

            del params['objective']

        self.params.update(params)

        return self

    

    

def logloss(y_true, Y_pred):

    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))

    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)





def main():

    clf = XGBoostClassifier(

        eval_metric = 'auc',

        num_class = 2,

        nthread = 4,

        silent = 0,

        )

    parameters = {

        'num_boost_round': [100, 250, 500],

        'eta': [0.05, 0.1, 0.3],

        'max_depth': [6, 9, 12],

        'subsample': [0.9, 1.0],

        'colsample_bytree': [0.9, 1.0],

    }

    clf = GridSearchCV(clf, parameters, n_jobs=4, cv=2)

    X = train.drop(['Dx:Cancer'],axis=1) 

    y=train['Dx:Cancer']



    clf.fit(X,y)

    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])

    print('score:', score)

    for param_name in sorted(best_parameters.keys()):

        print("%s: %r" % (param_name, best_parameters[param_name]))

    print('Xgb %error',procenterror(clf.predict(X),y),'rmsle',rmsle(clf.predict(X),y))







if __name__ == '__main__':

    main()