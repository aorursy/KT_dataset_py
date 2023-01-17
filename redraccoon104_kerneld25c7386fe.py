# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

# Any results you write to the current directory are saved as output.



dataset = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")



y = dataset['Chance of Admit ']

x = dataset.drop(['Chance of Admit ', 'Serial No.'], axis = 1)



x_train, x_dev, y_train, y_dev = train_test_split(x,y, test_size=0.2)



#Linear Regression

lm = linear_model.LinearRegression()



lm.fit(x_train, y_train)



result_val = cross_val_score(lm,x_train, y_train, cv=5).mean()



score = lm.score(x_dev, y_dev)





result_val, score
#Logistic Regression

from sklearn.linear_model import LogisticRegression



#Converting probabilty to Binary Class based on 0.8 probability threshold

y_train_binary = [1 if each > 0.8 else 0 for each in y_train]

y_dev_binary  = [1 if each > 0.8 else 0 for each in y_dev]



logr = LogisticRegression()



logr.fit(x_train,y_train_binary)



result_val_log = cross_val_score(logr,x_train,y_train_binary, cv=5).mean()



score2 = logr.score(x_dev, y_dev_binary)
result_val_log, score2
#Random Forest

from sklearn.ensemble import RandomForestRegressor

ranFr = RandomForestRegressor()

ranFr.fit(x_train,y_train)



from sklearn.model_selection import GridSearchCV

n_estimators = [10,50,100,200,300]

max_depth = [1,2,3,4,5,6,7]

max_leaf_nodes = [2,3,4,5,6,7]

param_grid = {'n_estimators':n_estimators, 'max_depth':max_depth, 'max_leaf_nodes':max_leaf_nodes}

grid_search = GridSearchCV(ranFr, param_grid, cv=5)

grid_search.fit(x_train, y_train)

print(grid_search.best_params_)



ranFr_best = grid_search.best_estimator_

ranFr_best.fit(x_train,y_train)



result_val_rfR = cross_val_score(ranFr_best,x_train, y_train, cv=5).mean()



score3 = ranFr_best.score(x_dev, y_dev)

result_val_rfR, score3
#XGBoost

from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(x_train,y_train)



xgb_max_depth = [1,2,3,4,5,6,7]

min_child_weight = [1,3,5,7]

eta = [0.01,0.05,0.1,0.2]



xgb_param_grid = {'max_depth':xgb_max_depth, 'min_child_weight':min_child_weight, 'eta':eta}

grid_search2 = GridSearchCV(xgb, xgb_param_grid, cv=5)

grid_search2.fit(x_train, y_train)

print(grid_search2.best_params_)



xgb_best = grid_search2.best_estimator_

xgb_best.fit(x_train,y_train)



result_val_xgb = cross_val_score(xgb_best,x_train, y_train, cv=5).mean()



score4 = xgb_best.score(x_dev, y_dev)





result_val_xgb, score4
#Predicting probabilities

div_pred = [328,101,3.5,4,4,9.22,0]

vam_pred = [318,108,5,3,4,8.35,1]





#{'GRE Score':328,'TOEFL Score':101,'University Rating':3.5,'SOP':4,'LOR ':4,'CGPA':9.22,'Research':0}





pred_xgb2 = pd.DataFrame({'GRE Score':[328,318],'TOEFL Score':[101,108],'University Rating':[3.5,5],'SOP':[4,3],'LOR ':[4,4],'CGPA':[9.22,8.35],'Research':[0,1]})





#Linear Reg

print(lm.predict([div_pred]),lm.predict([vam_pred]))



#Random Forest

print(ranFr_best.predict([div_pred]),lm.predict([vam_pred]))



#Logistic Reg

#Logistic Regressions predicted that Vamsi would get into UCLA although other models, gave him a lower probability.

#This is likely due to our threshold set, when we looked at the training data, people with over .8 probability had an average of 4.2 Univeristy Rating versus the overall dataset of 3.1, as well as most that attended top schools also participated in undergraduate research.

#Thus Vamsi's higher University rating, and Undergraduate Research was given more weight.

print(logr.predict([div_pred]),logr.predict([vam_pred]))

print(logr.predict_proba([div_pred]), logr.predict_proba([vam_pred]))



#XGBoost

#pred_xgb2

print(xgb_best.predict(pred_xgb2))