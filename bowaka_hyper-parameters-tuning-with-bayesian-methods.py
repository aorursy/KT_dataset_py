#import some necessary librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from hyperopt import hp

from hyperopt import tpe

from hyperopt import Trials

from hyperopt import fmin

from hyperopt import STATUS_OK



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from xgboost import XGBRegressor



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
df_train  = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
#We make the transformation on both train and test sets

df = pd.concat([df_train.drop('SalePrice', axis=1),df_test])

df = df.set_index('Id')



#When no pool, we put 'na'

df[["PoolQC"]] = df[["PoolQC"]].fillna('na')



#When no miscFeature, we put 'na'

df[["MiscFeature"]] = df[["MiscFeature"]].fillna('na')



#When noAlley, we put 'na'

df[["Alley"]] = df[["Alley"]].fillna('na')



#When noFence, we put 'na'

df[["Fence"]] = df[["Fence"]].fillna('na')



#When noFirePlace, we put 'na'

df[["FireplaceQu"]] = df[["FireplaceQu"]].fillna('na')



#Linear regressions for "Lot Frontage" based on 'LotArea'

lfront = df[["LotArea","LotFrontage"]].dropna()

model = LinearRegression().fit(lfront[["LotArea"]],lfront["LotFrontage"])

clfront = model.coef_

ilfront = model.intercept_

u = df["LotFrontage"]

v = ilfront + clfront * df.loc[u.isnull(),["LotArea"]]

df.loc[u.isnull(),["LotFrontage"]].index

v["LotFrontage"] = v["LotArea"]

df.update(v["LotFrontage"])



#Update for the nogarage types, with 'na' and 0, as explained in the data description

df[["GarageCond"]] = df[["GarageCond"]].fillna('na')

df[["GarageQual"]] = df[["GarageQual"]].fillna('na')

df[["GarageYrBlt"]] = df[["GarageYrBlt"]].fillna('na')

df[["GarageFinish"]] = df[["GarageFinish"]].fillna('na')

df[["GarageType"]] = df[["GarageType"]].fillna('na')

df[["GarageCars"]] = df[["GarageCars"]].fillna(0)

df[["GarageArea"]] = df[["GarageArea"]].fillna(0)



#Basement conditions - Again we can fill 'na' as explained in the text

df[["BsmtExposure"]] = df[["BsmtExposure"]].fillna('na')

df[["BsmtCond"]] = df[["BsmtCond"]].fillna('na')

df[["BsmtQual"]] = df[["BsmtQual"]].fillna('na')

df[["BsmtFinType2"]] = df[["BsmtFinType2"]].fillna('na')

df[["BsmtFinType1"]] = df[["BsmtFinType1"]].fillna('na')



#Massonery type

df[["MasVnrType"]] = df[["MasVnrType"]].fillna('na')

df[["MasVnrArea"]] = df[["MasVnrArea"]].fillna(0)







df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])



df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])



df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])



df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])

df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])







#For the other nan values, that are a small minority, I just put 0

df = df.fillna(value=0)



#We will also change some categorical feature with a notion of order into integer. This will reduce the number of features 

#in the one-hot-encoding and improve the speed of our XGBoost algorithm.

change_dict = {'LotShape':{'Reg':0,

                           'IR1':1,

                           'IR2':2,

                           'IR3':3},

              'LandSlope':{'Gtl':0,

                           'Mod':1,

                           'Sev':2},

              'ExterQual':{'Ex':5,

                           'Gd':4,

                           'TA':3,

                           'Fa':2,

                           'Po':1},

               'ExterCond':{'Ex':5,

                           'Gd':4,

                           'TA':3,

                           'Fa':2,

                           'Po':1},

               'BsmtQual':{'Ex':5,

                           'Gd':4,

                           'TA':3,

                           'Fa':2,

                           'Po':1,

                           'na':0},

               'BsmtCond':{'Ex':5,

                           'Gd':4,

                           'TA':3,

                           'Fa':2,

                           'Po':1,

                           'na':0},

               'BsmtExposure':{'Gd':5,

                           'Av':4,

                           'Mn':3,

                           'No':2,

                           'na':1,

                            0:0},

               'BsmtFinType1':{'GLQ':7,

                               'ALQ':6,

                               'BLQ':5,

                               'Rec':4,

                               'LwQ':3,

                               'Unf':2,

                               'na':1},

               'BsmtFinType2':{'GLQ':7,

                               'ALQ':6,

                               'BLQ':5,

                               'Rec':4,

                               'LwQ':3,

                               'Unf':2,

                               'na':1},

               'HeatingQC':{'Ex':5,

                           'Gd':4,

                           'TA':3,

                           'Fa':2,

                           'Po':1},

               'CentralAir':{'N':0,

                             'Y':1},

               'KitchenQual':{'Ex':5,

                           'Gd':4,

                           'TA':3,

                           'Fa':2,

                           'Po':1,

                           'na':0,

                             0:0},

               'Functional':{'Typ':7,

                             'Min1':6,

                             'Min2':5,

                             'Mod':4,

                             'Maj1':3,

                             'Maj2':2,

                             'Sev':1,

                             'Sal':0,

                              0:0},

               'FireplaceQu':{'Ex':5,

                           'Gd':4,

                           'TA':3,

                           'Fa':2,

                           'Po':1,

                           'na':0,

                           'no':0},

              'GarageFinish':{'Fin':4,

                              'RFn':3,

                              'Unf':2,

                              'na':1},

              'GarageQual':{'Ex':5,

                           'Gd':4,

                           'TA':3,

                           'Fa':2,

                           'Po':1,

                           'na':0,

                           'no':0},

              'GarageCond':{'Ex':5,

                           'Gd':4,

                           'TA':3,

                           'Fa':2,

                           'Po':1,

                           'na':0,

                           'no':0},

              'PoolQC':{'Ex':5,

                           'Gd':4,

                           'TA':3,

                           'Fa':2,

                           'Po':1,

                           'na':0,

                           'no':0},

               'Fence':{'GdPrv':4,

                        'MnPrv':3,

                        'GdWo':2,

                        'MnWw':1,

                        'na':0},

              }





for k in change_dict.keys():

    df[k] = df[k].apply(lambda x:change_dict[k][x])



#OverallQual and OverallCond is an a quality score, we can keep it as an integer

df['OverallQual'] = df['OverallQual'].apply(lambda x:int(x))

df['GarageYrBlt'] = df['GarageYrBlt'].apply(lambda x:int(x) if x!='na' else 1900)





#Changing OverallCond into a categorical variable

df['OverallCond'] = df['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

df['YrSold'] = df['YrSold'].astype(str)

df['MoSold'] = df['MoSold'].astype(str)







df = pd.get_dummies(df)

df.shape

from scipy import stats

from scipy.stats import skew



#All numerical features

numeric_feats = df.dtypes[(df.dtypes == np.int64) | (df.dtypes == np.float64)].index



# Check the skew of all numerical features

skewed_feats = df[numeric_feats].apply(lambda x: np.abs(skew(x))).sort_values(ascending=False)

skew_df = pd.DataFrame({'skew_val' :skewed_feats})

skew_df.head(10)



plt.figure(figsize=(15,10))

plt.title("Skewness of numerical variables")

plt.xticks(rotation=90)

plt.bar(skew_df.index,skew_df.skew_val.values)

plt.show()



#sk_level can be an hyperparameter of the problem, let put it to one

sk_level = 0.75



#We log-transform features with a skew level above 1

skew_feats = skew_df[skew_df.skew_val > sk_level].index



for feat in skew_feats:

    df[feat] = np.log1p(df[feat])
X = df[:len(df_train)].values

y = np.log(df_train['SalePrice'].values)
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error



def cv(model, X, y, cv = 5, seed=1989):

    """Our cross-validation function"""

    

    #List of losses

    losses = []

    

    #Define the kfold object, with a number of split cv, a random_seed to keep always same splits and with shuffle = True

    #meaning the sets are shuffled before splitting

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    pred = np.zeros(len(y))

    #kf returns a list of (train/test) indexes. We will loop over this list, train a model, and get each score

    for train_index, test_index in kf.split(X):

        

        #We define X_train, X_test, y_train and y_test

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        #We train our model. We evaluate it for this fold with X_test and y_test. 

        #Once the model didn't improve for 100 rounds, we stop the model and take the best iteration

        model.fit(X_train,y_train, verbose=False)

        

        pred[test_index] = model.predict(X_test)

        

    return np.sqrt(mean_squared_error(pred,y))

        
import time

#We generate our model. We can take a huge n_estimators as there will be early stopping

#We also use our custom loss function for the evaluation of the model



#To evaluate the time taken to train the model

t1 = time.time()



xgbr1 = XGBRegressor()



#We set a seed in the parameters to have consistent results

params = {'seed' : 1989,

          'n_estimators' :  2000,

          'feval' : 'rmse',

          'n_jobs': 4}

xgbr1.set_params(**params)



#We check the score using our crossval function

rmse_score = np.round(cv(xgbr1,X,y,5),4)



t2 = time.time()



print(f"Score of XGBoost vanilla : {rmse_score}")

print(f'Model trained in : {np.round(t2-t1)} s')
# Define the domain space to search for global minimum

space = {

    'boosting_type': hp.choice('boosting_type', 

                               [{'boosting_type': 'gbdt', 

                                    'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 

                                 {'boosting_type': 'goss'}]),

    'num_leaves': hp.quniform('num_leaves', 10, 400, 1),

    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),

    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 1000),

    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 1),

    'reg_alpha': hp.loguniform('reg_alpha', np.log(0.001), np.log(1.0)),

    'reg_lambda': hp.loguniform('reg_lambda', np.log(0.001), np.log(1.0)),

    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),

    'n_jobs':4,

    'n_estimators': 2000,

    'seed':1989

}
trials = Trials()
# Algorithm

tpe_algorithm = tpe.suggest
def objective(params, n_folds = 5):

    """Objective function for Gradient Boosting Hyperparameter Tuning"""

    

    #Our XGBRegressor

    model = XGBRegressor()

    

    #We set the parameters of the model

    model.set_params(**params)

    

    #We calculate the loss with our cv function

    loss = cv(model , X , y , n_folds)

    

    #We return a dictionnary for hyperopt framework

    return {'loss': loss, 'params': params, 'status': STATUS_OK}
# Optimize

t1=time.time()



optimized = fmin(fn = objective,

                 space = space,

                 algo = tpe.suggest,

                 max_evals = 100,

                 trials = trials)



t2 = time.time()



print(f'Model trained in : {np.round(t2-t1)} s')
#We get back the parameters from our optimization

params = trials.best_trial['result']['params']

print(params)
xgbr2 = XGBRegressor()



xgbr2.set_params(**params)



#We check the score using our crossval function

rmse_score = np.round(cv(xgbr2,X,y,5),4)



print(f"Score of XGBoost after optimization : {rmse_score}")
#I modify my dictionnary a bit here, with 3 keys, I will define the spaces from my xgboost, my lasso and my ponderation

space2 = {'xgb':{'boosting_type': hp.choice('boosting_type',

                                           [{'boosting_type': 'gbdt', 

                                            'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 

                                            {'boosting_type': 'goss'}]),

                'num_leaves': hp.quniform('num_leaves', 10, 400, 1),

                'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),

                'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 1000),

                'min_child_samples': hp.quniform('min_child_samples', 20, 500, 1),

                'reg_alpha': hp.loguniform('reg_alpha', np.log(0.001), np.log(1.0)),

                'reg_lambda': hp.loguniform('reg_lambda', np.log(0.001), np.log(1.0)),

                'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),

                'n_jobs':4,

                'n_estimators': 2000,

                'seed':1989},

         'lasso':{'alpha': hp.loguniform('alpha', np.log(0.001), np.log(1.0))},

         'pond' : hp.uniform('pond',0,1)

        }



# Algorithm

tpe_algorithm2 = tpe.suggest



# History

trials2 = Trials()
def objective2(space, n_folds = 5):

    """Objective function for Gradient Boosting Hyperparameter Tuning"""

    

    #Our XGBRegressor

    xgb = XGBRegressor()

    #Lasso

    lasso = Lasso()

    

    #We set the parameters of the xgb that are stored in space['xgb']

    xgb.set_params(**space['xgb'])

    

    #Same with lasso

    lasso.set_params(**space['lasso'])

    

    #We also take back our ponderation coefficient

    pond = space['pond']



    #We redefine manually the crossvalidation are we are using several models here

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1989)

    pred = np.zeros(len(y))

    

    #kf returns a list of (train/test) indexes. We will loop over this list, train a model, and get each score

    for train_index, test_index in kf.split(X):

        

        #We define X_train, X_test, y_train and y_test

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]



        #We train the two models. And we make the prediction using our ponderation. 

        xgb.fit(X_train,y_train, verbose=False)

        lasso.fit(X_train,y_train)



        #We ponderate the prediction of the two models

        pred[test_index] += pond*xgb.predict(X_test)

        pred[test_index] += (1-pond)*lasso.predict(X_test)

    

    #Once our prediction vector is calcutated for all the folds, we can evaluate the loss

    loss = np.sqrt(mean_squared_error(pred,y))



    

    #We return a dictionnary for hyperopt framework

    return {'loss': loss, 'params': space, 'status': STATUS_OK}

# Optimize

t1=time.time()



optimized = fmin(fn = objective2, space = space2, algo = tpe_algorithm2, 

max_evals = 500, trials = trials2)



t2 = time.time()



print(f'Model trained in : {np.round(t2-t1)} s')
#We get back the parameters from our optimization

params = trials2.best_trial['result']['params']
X_test = df[len(df_train):].values



#Our XGBRegressor

xgb = XGBRegressor()

#Lasso

lasso = Lasso()



#We set the parameters of the xgb that are stored in space['xgb']

xgb.set_params(**params['xgb'])



#Same with lasso

lasso.set_params(**params['lasso'])



#We also take back our ponderation coefficient

pond = params['pond']



pred = np.zeros(len(X_test))



#We train the two models. And we make the prediction using our ponderation. 

xgb.fit(X,y, verbose=False)

lasso.fit(X,y)



#We ponderate the prediction of the two models

pred += pond*xgb.predict(X_test)

pred += (1-pond)*lasso.predict(X_test)



#We remove the logarithm form of the prediction

pred = np.exp(pred)
my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': pred})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)