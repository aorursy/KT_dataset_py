import numpy as np

import pandas as pd

import sklearn.preprocessing as skp

from subprocess import check_output

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv',index_col=0)

test = pd.read_csv('../input/test.csv',index_col=0)
y_train = np.log(train.pop('SalePrice'))#log transform the target

all_df = pd.concat((train,test),axis=0) #concat the training and test data for faster preprocessing



all_df['MSSubClass']=all_df['MSSubClass'].astype(str) #Datatype fixes



all_dummy = pd.get_dummies(all_df)



mean_cols=all_dummy.mean() #calculate means of the categorical columns



all_dummy = all_dummy.fillna(mean_cols)

#Split the datasets back into training and test set

dummy_train = all_dummy.loc[train.index]

dummy_test = all_dummy.loc[test.index]

#prepare data for sklearn

X_train = dummy_train.values

X_test = dummy_test.values
from sklearn.model_selection import cross_val_score



from sklearn.linear_model import LassoCV,Lasso,ElasticNet, ElasticNetCV,RidgeCV,Ridge

from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge
elnet_cv = ElasticNetCV()

ridge_cv = RidgeCV(alphas=np.arange(1,100,10))

rf = RandomForestRegressor(n_estimators = 500)

ada = AdaBoostRegressor(n_estimators = 500)

gbr = GradientBoostingRegressor(n_estimators = 500)

lasso_cv = LassoCV()
elnet_cv.fit(X_train,y_train)

print(elnet_cv.alpha_)

ridge_cv.fit(X_train,y_train)

print(ridge_cv.alpha_)

lasso_cv.fit(X_train,y_train)

print(lasso_cv.alpha_)

rf.fit(X_train,y_train)

ada.fit(X_train,y_train)

gbr.fit(X_train,y_train)
elnet = ElasticNet(elnet_cv.alpha_)

ridge = Ridge(ridge_cv.alpha_)

lasso = Lasso(lasso_cv.alpha_)

elnet.fit(X_train,y_train)

ridge.fit(X_train,y_train)

lasso.fit(X_train,y_train)
y_1 = np.exp(elnet.predict(X_test))

y_2 = np.exp(ridge.predict(X_test))

y_3 = np.exp(rf.predict(X_test))

y_4 = np.exp(ada.predict(X_test))

y_5 = np.exp(gbr.predict(X_test))

y_6 = np.exp(lasso.predict(X_test))
score1 = np.mean(np.sqrt(-cross_val_score(elnet,X_train,y_train,cv=5,scoring='neg_mean_squared_error')))

score2 = np.mean(np.sqrt(-cross_val_score(ridge,X_train,y_train,cv=5,scoring='neg_mean_squared_error')))

score3 = np.mean(np.sqrt(-cross_val_score(rf,X_train,y_train,cv=5,scoring='neg_mean_squared_error')))

score4 = np.mean(np.sqrt(-cross_val_score(ada,X_train,y_train,cv=5,scoring='neg_mean_squared_error')))

score5 = np.mean(np.sqrt(-cross_val_score(gbr,X_train,y_train,cv=5,scoring='neg_mean_squared_error')))

score6 = np.mean(np.sqrt(-cross_val_score(lasso,X_train,y_train,cv=5,scoring='neg_mean_squared_error')))

scores = [score1,score2,score3,score4,score5,score6]

print(scores)

print(np.mean(scores))
y_final = (y_2+y_3+y_5)/3
submission_df = pd.DataFrame(data={'Id':test.index,'SalePrice':y_final})
submission_df.to_csv('submission_final_ridge_rf_gbr.csv',index=False)