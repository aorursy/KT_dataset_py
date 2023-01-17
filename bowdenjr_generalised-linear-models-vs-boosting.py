import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
data = pd.read_csv('https://stats.idre.ucla.edu/stat/data/poisson_sim.csv',index_col='id')
display(data.shape)

display(data.head())
data.prog = data.prog.astype('object')

display(data.dtypes)
plt.figure(figsize=(14,8))

plt.title('Number of awards based on type of academic programme',fontsize=20,weight='bold')

sns.countplot(data.num_awards,hue=data.prog,palette="rocket_r",lw=1.5,edgecolor='#444444')

plt.legend(loc='upper right',labels=['General','Academic','Vocational'])
prog = pd.get_dummies(data['prog'],prefix='prog')
prog
data = pd.merge(data.drop('prog',axis=1),prog,on='id')
data
X = data[['prog_1','prog_2','prog_3','math']]

y = data['num_awards']
from sklearn.linear_model import PoissonRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=81,test_size = .33)



Poisson_GLM = PoissonRegressor(alpha=33) # Optimisation is performed below



Poisson_GLM.fit(X_train,y_train)



y_pred = Poisson_GLM.predict(X_test)



MA_error = mean_absolute_error(y_test,y_pred)

RMS_error = mean_squared_error(y_test,y_pred,squared = False)

print('MAE = {:.4f}'.format(MA_error))

print('RMSE = {:.4f}'.format(RMS_error))
output = []

last_error =999



for i in range(1,101):

    iPoisson_GLM = PoissonRegressor(alpha=i,tol=1e-6)

    iPoisson_GLM.fit(X_train,y_train)    

    y_pred = iPoisson_GLM.predict(X_test)

    iRMS_error = mean_squared_error(y_test,y_pred,squared = False)

    if iRMS_error < last_error: best_alpha = i

    last_error = iRMS_error                                               

    output.append(iRMS_error)

                                               

print("RMSE is minimised at \u03B1 =",best_alpha)
plt.plot(output)

plt.title('RMSE for optimising \u03B1 in Poisson GLM')

plt.xlabel('\u03B1')

plt.ylabel('RMSE')

plt.show()
import statsmodels.api as sm

import statsmodels.formula.api as smf



Poisson_GLM2 = sm.GLM(y_train,X_train,family=sm.families.Poisson())

results = Poisson_GLM2.fit()

print(results.summary())
y_pred2 = Poisson_GLM2.predict(params=results.params,exog=X_test)



MA_error2 = mean_absolute_error(y_test,y_pred2)

RMS_error2 = mean_squared_error(y_test,y_pred2,squared = False)

print('MAE = {:.4f}'.format(MA_error2))

print('RMSE = {:.4f}'.format(RMS_error2))
residuals = results.resid_deviance

sns.distplot(residuals)

plt.title("Poission GLM residuals")
# Statsmodel formula wants the target and predictor in the same data

train = pd.merge(X_train,y_train,on='id')



formula = 'num_awards ~ math + math^2 + prog_1 + prog_2 + prog_3'

Poisson_GLM3 = smf.glm(formula=formula, data=train, family=sm.families.Poisson())

results2 = Poisson_GLM3.fit()

print(results2.summary())

y_pred3 = Poisson_GLM3.predict(params=results.params,exog=X_test)



MA_error3 = mean_absolute_error(y_test,y_pred3)

RMS_error3 = mean_squared_error(y_test,y_pred3,squared = False)

print('MAE = {:.4f}'.format(MA_error3))

print('RMSE = {:.4f}'.format(RMS_error3))
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



xgb = XGBRegressor()



parameters = {'n_estimators': [50,100,200,500],

              'max_depth': [1,2,3,5],

              'learning_rate': [0.001,0.01,0.1],

             }



grid = GridSearchCV(estimator = xgb, param_grid = parameters,n_jobs =-1,scoring=make_scorer(mean_squared_error,squared=False,greater_is_better=False))



grid.fit(X_train,y_train)



y_pred_grid = grid.predict(X_test)



MA_error4 = mean_absolute_error(y_test,y_pred_grid)

RMS_error4 = mean_squared_error(y_test,y_pred_grid,squared = False)



print('MAE = {:.4f}'.format(MA_error4))

print('RMSE = {:.4f}'.format(RMS_error4))

print('Best Parameters = ',grid.best_params_)
from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



Cat = CatBoostRegressor(verbose=0)



parameters = {'n_estimators': [50,100,200,300],

              'max_depth': [1,2,3,5],

              'learning_rate': [0.001,0.01,0.1],

             }



grid2 = GridSearchCV(estimator = Cat, param_grid = parameters,n_jobs =-1,scoring=make_scorer(mean_squared_error,squared=False,greater_is_better=False))



grid2.fit(X_train,y_train)



y_pred_grid = grid2.predict(X_test)



MA_error5 = mean_absolute_error(y_test,y_pred_grid)

RMS_error5 = mean_squared_error(y_test,y_pred_grid,squared = False)



print('MAE = {:.4f}'.format(MA_error5))

print('RMSE = {:.4f}'.format(RMS_error5))

print('Best Parameters = ',grid2.best_params_)
from sklearn.linear_model import LinearRegression



LR = LinearRegression()



LR.fit(X_train,y_train)



y_pred_6 = LR.predict(X_test)



MA_error6 = mean_absolute_error(y_test,y_pred_6)

RMS_error6 = mean_squared_error(y_test,y_pred_6,squared = False)



print('MAE = {:.4f}'.format(MA_error6))

print('RMSE = {:.4f}'.format(RMS_error6))
from sklearn.linear_model import Ridge



RR = Ridge()



RR.fit(X_train,y_train)



y_pred_ridge = RR.predict(X_test)



MA_error_ridge = mean_absolute_error(y_test,y_pred_ridge)

RMS_error_ridge = mean_squared_error(y_test,y_pred_ridge,squared = False)



print('MAE = {:.4f}'.format(MA_error_ridge))

print('RMSE = {:.4f}'.format(RMS_error_ridge))
output = []

last_error =999



for i in range(1,101):

    iRidge = Ridge(alpha=i,tol=1e-6)

    iRidge.fit(X_train,y_train)    

    y_pred = iRidge.predict(X_test)

    iRMS_error = mean_squared_error(y_test,y_pred,squared = False)

    if iRMS_error < last_error: best_alpha = i

    last_error = iRMS_error                                               

    output.append(iRMS_error)

                                               

print("RMSE is minimised at \u03B1 =",best_alpha)
plt.plot(output)

plt.title('RMSE for optimising \u03B1 in Ridge Regression')

plt.xlabel('\u03B1')

plt.ylabel('RMSE')

plt.show()
from sklearn.ensemble import StackingRegressor



estimators = [('XGBoost',xgb),

              ('CatBoost',Cat),

              ('Poisson GLM',Poisson_GLM),

              ('Linear Regression',LR),

              ('Ridge Regression',RR)]



stack = StackingRegressor(estimators=estimators)

stack.fit(X_train,y_train)

y_pred_stack = stack.predict(X_test)



MA_error_stack = mean_absolute_error(y_test,y_pred_stack)

RMS_error_stack = mean_squared_error(y_test,y_pred_stack,squared = False)



print('MAE = {:.4f}'.format(MA_error_stack))

print('RMSE = {:.4f}'.format(RMS_error_stack))
overall_results = {'Model':['Optimised SKLearn Poisson GLM','StatsModel Poission GLM','XGBoost','CatBoost','Linear Regression','Ridge Regression','Stacked Model'],

                    'MA Error':[MA_error,MA_error2,MA_error4,MA_error5,MA_error6,MA_error_ridge,MA_error_stack],

                    'RMS Error': [RMS_error,RMS_error2,RMS_error4,RMS_error5,RMS_error6,RMS_error_ridge,RMS_error_stack]

                  }



overall_results = pd.DataFrame(overall_results).round(3).sort_values(by='RMS Error',ascending=True)

overall_results