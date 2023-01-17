import pandas as pd 

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier

from sklearn.feature_selection import RFECV

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBRegressor

from sklearn.metrics import make_scorer,mean_squared_error,mean_absolute_error,r2_score,explained_variance_score

train = pd.read_csv('/kaggle/input/sec3-eda-fe-categorical/eng_filt_train.csv')

test = pd.read_csv('/kaggle/input/sec3-eda-fe-categorical/eng_filt_test.csv')

train['MSSubClass'] = train['MSSubClass'].astype('category')

test['MSSubClass'] = test['MSSubClass'].astype('category')

traintest = pd.concat([train,test],axis=0,ignore_index = True)
train.shape,test.shape,traintest.shape
train.columns
def prepare_data_for_linear_model(whole,tr,te):

    """ 

    Preparing data for non-tree based modelling approach and returns train, test

    """

    

    # Removing unnecessary columns

    whole = whole.drop(['LogPrice','SalePrice','Id'],axis = 1)

    tr = tr.drop(['LogPrice','SalePrice','Id'],axis = 1)

    te = te.drop(['Id'],axis = 1)

    

    # Select cats column and one hot encode remaining columns keep as it is

    cats = train.select_dtypes(include=['object','category']).columns

    

    whole = pd.get_dummies(whole,columns = cats)

    tr = whole[:1425]

    te = whole[1425:]

    

    return tr,te

    

    

trs,tes = prepare_data_for_linear_model(traintest,train,test)

features = trs.columns



scaler = StandardScaler(with_mean=False)





tr = scaler.fit_transform(trs)

te = scaler.transform(tes)
def rmse(y_test,pred):

    y = np.exp(y_test)-1

    y_ = np.exp(pred)-1

    mse = mean_squared_error(y,y_)

    return np.sqrt(mse)



def mae(y_test,pred):

    y = np.exp(y_test)-1

    y_ = np.exp(pred)-1

    mae = mean_absolute_error(y,y_)

    return mae



def r2score(y_test,pred):

    y = np.exp(y_test)-1

    y_ = np.exp(pred)-1

    r2 = r2_score(y,y_)

    return r2



def exp_var(y_test,pred):

    y = np.exp(y_test)-1

    y_ = np.exp(pred)-1

    evs = explained_variance_score(y,y_)

    return evs



    

scorer = {'mae':make_scorer(mae,greater_is_better=False),'rmse': make_scorer(rmse,greater_is_better=False), 

          'r2score':make_scorer(r2score),'expvar':make_scorer(exp_var)}



def parameter_select(model,hyperparameters,tr,features):

    """

    Grid search for best parameter based on lowest RMSE score for given model. Display 2 plots 1.RMSE plot with lower and upper bound

    2. R2 score and explained variance score plot

    """

    grid = GridSearchCV(model,hyperparameters,scoring = scorer,cv=3,n_jobs = -1,refit='rmse')

    grid.fit(tr,train['LogPrice'])

    print('Best RMSE Score:', -grid.best_score_)

    print('Best Params',grid.best_params_)

    print()

    result = grid.cv_results_

    

    fig = plt.figure(figsize=(20,13))



    X_axis = np.array(result['param_alpha'].data, dtype=float)

    plt.subplot(2,2,1)

    plt.semilogx(X_axis,-result['mean_test_rmse'], linestyle='--', marker='x',label='RMSE')

    plt.fill_between(X_axis, -result['mean_test_mae'],np.sqrt(len(tr)/3)*(-result['mean_test_mae']),alpha=0.1)

    plt.semilogx(grid.best_params_['alpha'],-grid.best_score_, 'Xr')

    plt.xlabel('alpha')

    plt.ylabel('score')



    plt.subplot(2,2,2)

    plt.semilogx(X_axis, result['mean_test_r2score'],  linestyle='--', marker='x',label='r2score')

    plt.semilogx(X_axis, result['mean_test_expvar'],  linestyle='--', marker='x',label='expvar')

    plt.legend(loc = 'best')

    plt.xlabel('alpha')

    plt.ylabel('score')

    

    plt.subplot(2,1,2)

    plt.bar(x= features,height=grid.best_estimator_.coef_)

    plt.xticks(rotation=90)

    

    return grid.best_params_
##------Linear Regression------##

# lr = LinearRegression(fit_intercept=False,copy_X=True,n_jobs=-1)

# rmse,r2 = test_model(lr,tr)

# print("Linear Regression:\nRMSE:{:.3f}\nScore:{:.3f}".format(rmse,r2))





##-----Lasso Regression-------##

lasso = Lasso()

hyperparameters = {'alpha':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000],

                   'random_state' : [1],

                   'fit_intercept':[True],

                   'max_iter':[2000]}

print( 'Lasso Regression Analysis')

best_params = parameter_select(lasso,hyperparameters,tr,features)
##-----Ridge Regression-------##

ridge = Ridge()

hyperparameters = {'alpha':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000],

                   'random_state' : [1],

                   'fit_intercept':[True],

                   'max_iter':[2000]}

print('Ridge Regression Analysis')

best_params = parameter_select(ridge,hyperparameters,tr,features)



##---ElasticNet Regression----##

eln = ElasticNet()

hyperparameters = {'alpha': [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000],

                   'l1_ratio':[0.1,0.3,0.5,0.7,0.9,1],

                   'fit_intercept':[True],

                   'max_iter':[2000]}



print('Elastic Net regression analysis')

best_params = parameter_select(eln,hyperparameters,tr,features)

selector = RFECV(Ridge(),step = 1, cv=5,scoring = 'neg_mean_squared_error',n_jobs = -1).fit(tr,train['LogPrice'])

selected_features = trs.columns[selector.support_]



selected_features
ridge = ElasticNet()

hyperparameters = {'alpha':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000],

                   'l1_ratio':[0.1,0.3,0.5,0.7,0.9,1],

                   'random_state' : [1],

                   'fit_intercept':[True],

                   'max_iter':[2000],

                   'tol': [1e-4]}

sc = StandardScaler()

finaltrain =sc.fit_transform(trs[selected_features])

best_params = parameter_select(ridge,hyperparameters,finaltrain,selected_features)



model = ElasticNet(**best_params).fit(finaltrain,train['LogPrice'])



final_test = sc.transform(tes[selected_features])

pred = model.predict(final_test)

pred = np.exp(pred)-1

submission = pd.DataFrame({'Id':test['Id'].astype(int),'SalePrice':pred})

submission.to_csv('submission.csv',index=False)