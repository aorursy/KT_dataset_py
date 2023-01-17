import numpy as np

import xgboost as xgb

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression, TheilSenRegressor,RANSACRegressor,HuberRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

import seaborn as sns

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')





total = train.append(test)





# process columns, apply LabelEncoder to categorical features

for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder() 

        lbl.fit(list(train[c].values) + list(test[c].values)) 

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))



# find simply the 'autocorrelation' or say the basic regressive explained mean variation 

estimators = [

              ('TSR', TheilSenRegressor(random_state=42)), #0.11

              ('OLS', LinearRegression()),   # 0.53

              ('RSR', RANSACRegressor(random_state=42)),   #0.509 AR#0.03

              ('HBR', HuberRegressor())]   #AR# 0.01  

lw = 2

#cellen=['X'+str(w) for w in range(10,385) if str(w) not in ['25','72','121','149','188','193','303','381']]

cellen=['X47','X95','X314','X315','X232','X119','X311','X76','X329','X238','X340','X362','X137']

cellen=['X47','X95','X314','X315','X232','X119','X311','X76','X329','X238','X340','X362','X137','ID'] # with ID

X_=train[cellen]

y_=train['y']

for name, estimator in estimators:

    answ=estimator.fit(X_, y_)

    print(name,estimator.score(X_,y_),estimator.get_params())

    train[name]=estimator.predict(X_)

    y_=estimator.predict(X_)

        

# now we simply substract the explained variation and stay with the 'rest' 

#we will try to forecast with xgboost

test['TSR']=estimator.predict(test[cellen])

train_mem=train

kolom=train.columns

kolom=[k for k in kolom if k not in ['y','OLS','HBR','RSR','TSR','rest']+cellen]  #'ID'

train['rest']=train['y']-train['TSR']  #print(train['rest'].T)



#we use that noise detecting cluster analysis

for adc in range (8,9):

    train=train_mem

    print('n_comp',adc)

    ##Add decomposed components: PCA / ICA etc.

    n_comp = adc



    # PCA

    pca = PCA(n_components=n_comp, random_state=420)

    pca2_results_train = pca.fit_transform(train[kolom])

    pca2_results_test = pca.transform(test[kolom])

    

    # ICA

    ica = FastICA(n_components=n_comp, random_state=420)

    ica2_results_train = ica.fit_transform(train[kolom])

    ica2_results_test = ica.transform(test[kolom])



    # GRP

    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)

    grp_results_train = grp.fit_transform(train[kolom])

    grp_results_test = grp.transform(test[kolom])



    # SRP

    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

    srp_results_train = srp.fit_transform(train[kolom])

    srp_results_test = srp.transform(test[kolom])



    # Append decomposition components to datasets

    pcakol=[]

    icakol=[]

    grpkol=[]

    srpkol=[]    

    for i in range(1, n_comp+1):

        train['pca_' + str(i)] = pca2_results_train[:,i-1]

        test['pca_' + str(i)] = pca2_results_test[:, i-1]

        pcakol+=['pca_' + str(i)]

        

        train['ica_' + str(i)] = ica2_results_train[:,i-1]

        test['ica_' + str(i)] = ica2_results_test[:, i-1]

        icakol+=['ica_' + str(i)]



        train['grp_' + str(i)] = grp_results_train[:,i-1]

        test['grp_' + str(i)] = grp_results_test[:, i-1]

        grpkol+=['grp_' + str(i)] 

        

        train['srp_' + str(i)] = srp_results_train[:,i-1]

        test['srp_' + str(i)] = srp_results_test[:, i-1]

        srpkol+=['srp_' + str(i)]

        

    y_train = train['rest']

    y_mean = np.mean(y_train)

    ### Regressor

    import xgboost as xgb



    # prepare dict of params for xgboost to run with

    xgb_params = {

        'n_trees': 1500, 

        'eta': 0.0045,

        'max_depth': 5,

        'subsample': 0.93,

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        'base_score': y_mean, # base prediction = mean(target)

        'silent': 1

    }





    # form DMatrices for Xgboost training

    totkol=kolom+pcakol+icakol+grpkol+srpkol

    #print(totkol)

    dtrain = xgb.DMatrix(train[totkol], train['rest'])

    dtest = xgb.DMatrix(test[totkol])



    num_boost_rounds = 1250

    # train model

    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)





    # check f2-score (to get higher score - increase num_boost_round in previous cell)



    print(r2_score(dtrain.get_label(),model.predict(dtrain)))



    # make predictions and save results

    y_pred = model.predict(dtest)

    print(y_pred)



output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

output.to_csv('superScript AR_ %f.csv' % adc, index=False)

    

fig, ax = plt.subplots(figsize=(10,50))

xgb.plot_importance(model, height=0.9, ax=ax)

plt.show()

    

print(r2_score(dtrain.get_label(),model.predict(dtrain)))



# reconstructing the prediction with  the combination of TSR + XGB

# make predictions and save results #print(dtest.feature_names) #print(dtrain.feature_names)

y_pred = model.predict(dtest)+test['TSR']

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

output.to_csv('superScript ARenPCA_XGB_ %f.csv' % adc, index=False)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

plt.figure(figsize=(12,8))

sns.distplot(output.y.values, bins=50, kde=False)

plt.xlabel('Predicted AVG Time on Test platform', fontsize=12)

plt.show()

import numpy as np

from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import ElasticNetCV, LassoLarsCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline, make_union

from sklearn.utils import check_array

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import r2_score







class StackingEstimator(BaseEstimator, TransformerMixin):

    

    def __init__(self, estimator):

        self.estimator = estimator



    def fit(self, X, y=None, **fit_params):

        self.estimator.fit(X, y, **fit_params)

        return self

    def transform(self, X):

        X = check_array(X)

        X_transformed = np.copy(X)

        # add class probabilities as a synthetic feature

        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):

            X_transformed = np.hstack((self.estimator.predict_proba(X), X))



        # add class prodiction as a synthetic feature

        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))



        return X_transformed





train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(train[c].values) + list(test[c].values))

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))



stacked_pipeline = make_pipeline(

    StackingEstimator(estimator=LassoLarsCV(normalize=True)),

    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),

    LassoLarsCV()

    

)

kolom=train.columns

train['y_pred']=train['y']

kolom=[w for w in kolom if w not in ['y']]

for lp in range(0,4):

    stacked_pipeline.fit(train[kolom],train['y_pred'])

    train['y_pred'] = stacked_pipeline.predict(train[kolom])

test['y_pred'] = stacked_pipeline.predict(test[kolom])

print(test['y_pred'].T)

train['y_diff']=train['y']-train['y_pred']

print(train['y_diff'].T)

n_comp = 12



# tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)

tsvd_results_train = tsvd.fit_transform(train[kolom])

tsvd_results_test = tsvd.transform(test[kolom])



# PCA

pca = PCA(n_components=n_comp, random_state=420)

pca2_results_train = pca.fit_transform(train[kolom])

pca2_results_test = pca.transform(test[kolom])



# ICA

ica = FastICA(n_components=n_comp, random_state=420)

ica2_results_train = ica.fit_transform(train[kolom])

ica2_results_test = ica.transform(test[kolom])



# GRP

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)

grp_results_train = grp.fit_transform(train[kolom])

grp_results_test = grp.transform(test[kolom])



# SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

srp_results_train = srp.fit_transform(train[kolom])

srp_results_test = srp.transform(test[kolom])



#save columns list before adding the decomposition components



usable_columns = list(set(train.columns) - set(['y']))



# Append decomposition components to datasets

for i in range(1, n_comp + 1):

    train['pca_' + str(i)] = pca2_results_train[:, i - 1]

    test['pca_' + str(i)] = pca2_results_test[:, i - 1]



    train['ica_' + str(i)] = ica2_results_train[:, i - 1]

    test['ica_' + str(i)] = ica2_results_test[:, i - 1]



    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]

    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]



    train['grp_' + str(i)] = grp_results_train[:, i - 1]

    test['grp_' + str(i)] = grp_results_test[:, i - 1]



    train['srp_' + str(i)] = srp_results_train[:, i - 1]

    test['srp_' + str(i)] = srp_results_test[:, i - 1]



#usable_columns = list(set(train.columns) - set(['y']))



y_train = train['y_diff'].values

y_mean = np.median(y_train)

id_test = test['ID'].values

#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 

#finaltrainset = train[usable_columns].values

#finaltestset = test[usable_columns].values





'''Train the xgb model then predict the test data'''



xgb_params = {

    'n_trees': 1000, # 520 boost 2000 > lb 0.55  #520 boost 1250 LB 0.56

    'eta': 0.0045,

    'max_depth': 4,

    'subsample': 0.93,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean, # base prediction = mean(target)

    'silent': 1

}

# NOTE: Make sure that the class is labeled 'class' in the data file

kolom=train.columns

train['y_pred']=train['y']

kolom=[w for w in kolom if w not in ['y','y_pred','y_diff']]



dtrain = xgb.DMatrix(train[kolom], y_train)

dtest = xgb.DMatrix(test[kolom])



num_boost_rounds = 750

# train model

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)



fig, ax = plt.subplots(figsize=(10,50))

xgb.plot_importance(model, height=0.9, ax=ax)

plt.show()

    

print(r2_score(dtrain.get_label(),model.predict(dtrain)))



sub = pd.DataFrame()

sub['ID'] = id_test

sub['y'] = y_pred + test['y_pred']

sub.to_csv('stacked-models.csv', index=False)

plt.figure(figsize=(12,8))

sns.distplot(sub.y.values, bins=50, kde=False)

plt.xlabel('Predicted AVG Time on Test platform', fontsize=12)

plt.show()
