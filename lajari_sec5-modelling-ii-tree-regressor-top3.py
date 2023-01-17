import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_squared_error,make_scorer

from sklearn.preprocessing import LabelEncoder

pd.set_option('max_columns',100)
train = pd.read_csv('/kaggle/input/sec3-eda-fe-categorical/eng_filt_train.csv')

test = pd.read_csv('/kaggle/input/sec3-eda-fe-categorical/eng_filt_test.csv')

train.shape,test.shape
train.columns
train['MSSubClass'] = train['MSSubClass'].astype('category')

test['MSSubClass'] = test['MSSubClass'].astype('category')

traintest = pd.concat([train,test],axis=0,ignore_index=True)

cats = traintest.select_dtypes(include = ['object','category']).columns



le =LabelEncoder()

traintest[cats] = traintest[cats].apply(le.fit_transform)

traintest.drop(['LogPrice','SalePrice','Id'],axis=1,inplace=True)

trs = traintest[:1425]

tes = traintest[1425:]



traintest.head()
def rmse(y,y_):

    mse = mean_squared_error(y,y_)

    return np.sqrt(mse)



scorer = make_scorer(rmse,greater_is_better=False)

def default_eval(model):

    scores = cross_val_score(model,trs,train['SalePrice'],scoring=scorer,cv=5)

    mean = np.mean(-scores)

    std = np.std(-scores)

    print("Mean score:{:.2f}    Std. Dev.:{:.2f}".format(mean,std))
dt = DecisionTreeRegressor(random_state=1)

default_eval(dt)
rf = RandomForestRegressor(random_state=1)

default_eval(rf)
gb = GradientBoostingRegressor(random_state=1)

default_eval(gb)

xgb = XGBRegressor(random_state=1)

default_eval(dt)
def grid_analysis(hyperparameters):

    gb = GradientBoostingRegressor()

    grid = GridSearchCV(gb,param_grid=hyperparameters,scoring = scorer,cv=5,n_jobs = -1)

    grid.fit(trs,train['SalePrice'])

    print("Best Parameters:\n",grid.best_params_)

    print("\nBest score ",-grid.best_score_)
hyperparameters = {'learning_rate':[0.02,0.05,0.08,0.09],

                   'n_estimators':[500,700,800,900,1000],

                  'random_state':[1],

                   'min_samples_split':[10],

                   'min_samples_leaf' : [5],

                   'max_depth': [8],

                   'max_features' : ['sqrt'] ,

                   'subsample' : [0.8],

                  }

grid_analysis(hyperparameters)

hyperparameters = {'learning_rate':[0.02],

                   'n_estimators':[500],

                  'random_state':[1],

                   'min_samples_split':[3,4,5,10,15,20],

                   'max_depth': range(3,16),

                   'max_features' : ['sqrt'] ,

                   'subsample' : [0.8],

                  }

grid_analysis(hyperparameters)
hyperparameters = {'learning_rate':[0.02],

                   'n_estimators':[500],

                  'random_state':[1],

                   'min_samples_split':[5],

                   'max_depth':[4] ,

                   'min_samples_leaf': [1,2,3,4],

                   'max_features' : ['sqrt'] ,

                   'subsample' : [0.8]

                  }

grid_analysis(hyperparameters)
hyperparameters = {'learning_rate':[0.02],

                   'n_estimators':[500],

                  'random_state':[1],

                   'min_samples_split':[5],

                   'min_samples_leaf': [3],

                   'max_depth': [4],

                   'max_features' : range(5,27,2) ,

                   'subsample' : [0.8]

                  }

grid_analysis(hyperparameters)
hyperparameters = {'learning_rate':[0.02],

                   'n_estimators':[500],

                  'random_state':[1],

                   'min_samples_split':[5],

                   'min_samples_leaf': [3],

                   'max_depth': [4],

                   'max_features' : [19] ,

                   'subsample' : [0.6,0.7,0.75,0.8,0.85,0.9,0.95]

                  }

grid_analysis(hyperparameters)


best_params = {'learning_rate': 0.02 ,     

                   'n_estimators':500,   

                  'random_state':1,

                   'min_samples_split':5,

                   'min_samples_leaf': 3,

                   'max_depth': 4,

                   'max_features' : 19 ,

                   'subsample' : 0.8

                  }

gb = GradientBoostingRegressor(**best_params)



scores = cross_val_score(gb,trs,train['SalePrice'],scoring=scorer,cv=5)

np.mean(-scores),np.std(-scores)

from sklearn.model_selection import learning_curve





estimator = GradientBoostingRegressor(**best_params)



train_sizes, train_scores, test_scores = learning_curve(estimator,trs,train['SalePrice'],cv=5,

                                                        train_sizes = np.linspace(0.1, 1.0, 10), n_jobs=-1,scoring=scorer)

train_scores = -train_scores

test_scores = -test_scores



train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)



# Plot learning curve



plt.figure(figsize=(10,7))

plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1,

                     color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training error (RMSE)")

plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation error (RMSE)")

plt.axhline(y=10000, color='b', linestyle='--',label='Desired RMSE Threshold')

plt.title('Learning Curve for Best Model')

plt.xlabel('Training Sizes')

plt.ylabel('RMSE score')

plt.legend(loc="best")



model = GradientBoostingRegressor(**best_params).fit(trs,train['SalePrice'])





pred = model.predict(tes)



submission = pd.DataFrame({'Id':test['Id'].astype(int),'SalePrice':pred})

submission.to_csv('submission.csv',index=False)