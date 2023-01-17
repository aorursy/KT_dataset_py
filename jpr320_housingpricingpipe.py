# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.preprocessing import RobustScaler as RS



from scipy import stats



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import RobustScaler as RS

## Data Modelling 



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, RidgeCV,LassoCV,ElasticNetCV

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, VotingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb



#Model Validation



from sklearn.model_selection import KFold



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')



train.columns.tolist()
train = pd.read_csv('../input/train.csv')



test = pd.read_csv('../input/test.csv')

y_train = np.log1p(train.SalePrice.values)

ID = test['Id'].copy()



train.drop(columns = 'SalePrice',inplace = True)

ntrain = train.shape[0]



big_df = pd.concat((train,test)).reset_index(drop = True)
def clean(df):

    train = df.copy()

    

    train.columns = train.columns.str.upper()

    

    #tgt = train['SALEPRICE'] # store target because we will need it later

    ID = train['ID'] # save for later

    train.drop(columns = 'ID') # irrelevant column

    if 'SALEPRICE' in train.columns:

        train.drop(columns = 'SALEPRICE',inplace = True)

    

    special_fills = ['STREET','BSMTCOND','BSMTEXPOSURE','BSMTFINTYPE1','BSMTFINTYPE2',

                'FIREPLACEQU','GARAGETYPE','GARAGEFINISH','GARAGEQUAL','GARAGECOND','POOLQC',

                'FENCE','MISCFEATURE'] # these NAs in columns are filled without imputation

    ### Missing value imputation ###

    for col in special_fills:

        train[col] = train[col].fillna('NF') #NF means no feature in this case

    for col in list(set(train.select_dtypes(include = [np.number]).columns.tolist()) - set(special_fills)):

        if train[col].nunique()<100:

            train[col] = train[col].fillna(stats.mode(train[col].values)[0].tolist()[0])

        else:

            train[col] = train[col].fillna(np.mean(train[col])) #might consider a more robust measure here

    for col in list(set(train.select_dtypes(include = 'object').columns.tolist()) - set(special_fills)):

        print('column {} has {} unique values and {} % values missing'.format(col, train[col].nunique(), np.mean(train[col].isna()*100)))

        if np.mean(train[col].isna()) > .1:

            train.drop(columns = col,inplace = True)

            print('dropped {}'.format(col))

        else:

            train[col] = train[col].fillna(stats.mode(train[col])[0].tolist()[0])

    

    if sum(train.isna().any()) > 0:

        print('Something wrong, look at code')

        return None

    

    print('imputation complete')

    

    

    ##Feature engineering##

    

    for col in ['MSSUBCLASS','YRSOLD','MoSold','YEARREMODADD']:

        col = col.upper()

        train[col] = train[col].astype(str)

    

    scaler = RS() # for scaling robust to outliers

    

    for col in train.select_dtypes(include = [np.number]).columns.tolist():

        train[[col]] = scaler.fit_transform(train[[col]]) # scale

        

    train = pd.get_dummies(train) # one hot encode categorical variables

    

    #tgt = np.log1p(tgt)

    

    print('Cleaning complete. Data Frame now has {} columns and {} rows\nOld df had {} columns and {} rows'.format(train.shape[1],train.shape[0],df.shape[1],df.shape[0]))

    

    return train
big_df = clean(big_df)
train = big_df[:ntrain]

test = big_df[ntrain:]
test.shape
def rmse(model):

    rmse = np.sqrt(-cross_val_score(model,train.values,tgt,scoring = 'neg_mean_squared_error',cv = kf))

    return rmse

def train_models(feat,tgt,seed = 4233234795):

    #set up kfold 

    train = feat.copy()

    tgt = tgt.copy()

    kf = KFold(n_splits = 5,shuffle = True,random_state = seed).get_n_splits(train.values)

    

    #Ridge Regression

    ridge = RidgeCV(alphas = np.logspace(-5,5,10),cv = kf)

    

    #Lasso 

    lasso = LassoCV(eps = .001,n_alphas = 100,cv = kf,random_state = seed)

    

    #BayesianRidge 

    

    bridge = BayesianRidge(n_iter = 1000)

    

    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

    vregress = VotingRegressor(estimators = [('rr',ridge),('lr',lasso),('gb',GBoost),('br',bridge)])

    

    vregress.fit(train,tgt)

    

    return vregress

    

    

    

    

    
model = train_models(train,y_train)
sub = pd.DataFrame()

sub['Id'] = ID

sub['SalePrice'] = np.exp(model.predict(test))
sub.to_csv('submission.csv',index = False)