import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

%matplotlib inline
data = pd.read_csv('../input/train.csv')
data_test=pd.read_csv('../input/test.csv')

#data_test=data_test_original.drop(columns= ['id'], axis=1)
data_test.head()
data.head()
data.describe()
data.info()
data.isnull().values.any()
# Compute the correlation matrix

corr = data.corr(method="kendall")



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)



plt.show()
#data = data.drop(columns = ['year'], axis = 1)

#data= data.drop(columns = ['4770'], axis=1)
#data=data.drop(columns= ['id'], axis=1)
data.head()
#from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()

X=data.drop(['AveragePrice'],axis=1)

#X=scaler.fit_transform(X)

#data_new=pd.DataFrame(X, columns=data.columns)

y = data['AveragePrice'].tolist()
#data_test = data_test.drop(columns = ['year'], axis = 1)

#data_test= data_test.drop(columns = ['4770'], axis=1)
#X_test=scaler.fit_transform(data_test)

X_test=data_test


#from sklearn.tree import DecisionTreeRegressor

#regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=100, min_samples_split=2, 

#                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 

#                                  random_state=10, max_leaf_nodes=None, min_impurity_decrease=0.0, 

#                                  min_impurity_split=None, presort=False)
#from sklearn.neural_network import MLPRegressor

#regressor = MLPRegressor(hidden_layer_sizes=(100,200,300,400,300,200,100))
#regressor.fit(X=X,y=y)

#y_pred = regressor.predict(X_test)
#added latest

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import explained_variance_score



def performance_metrics(y_true,y_pred):

    rmse = mean_absolute_error(y_true,y_pred)

    r2 = r2_score(y_true,y_pred)

    explained_var_score = explained_variance_score(y_true,y_pred)

    

    return rmse,r2,explained_var_score

# #This is working the best so far

# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import make_scorer



# from sklearn.ensemble import GradientBoostingRegressor

# clf = GradientBoostingRegressor(criterion='friedman_mse',learning_rate=0.1)        #Initialize the classifier object



# parameters = {'random_state':[6,12,75,108,92,7,81,64]}    #Dictionary of parameters



# scorer = make_scorer(r2_score)         #Initialize the scorer using make_scorer



# grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



# grid_fit = grid_obj.fit(X,y)        #Fit the gridsearch object with X_train,y_train



# best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



# y_pred = best_clf.predict(X_test)
from sklearn import preprocessing

import xgboost as xgb

from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



xgb1 = XGBRegressor(random_state=99, booster='gbtree',n_estimators=800)

parameters = {'objective':['reg:linear'],

              #'learning_rate': [0.05,0.1,0.5], #so called `eta` value

              'max_depth': [9],

              'silent': [1],

              #'subsample': [0.7],

              #'colsample_bytree': [0.7],

              #,'booster': ['gbtree', 'gblinear','dart']

             }



#xgb_grid = GridSearchCV(xgb1, parameters,cv = 2,n_jobs = 5, verbose=False)

#xgb_grid_fit=xgb_grid.fit(X, y)

scorer = make_scorer(r2_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(xgb1,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X,y)        

best_clf=grid_fit.best_estimator_

y_pred=best_clf.predict(X_test)
best_clf
#from sklearn.model_selection import GridSearchCV

#from sklearn.metrics import make_scorer





#from sklearn.tree import DecisionTreeRegressor

#clf = DecisionTreeRegressor(splitter='best', max_depth=None)

#parameters={'criterion':['mse','friedman_mse','mae'], 'random_state': [10,49,100]}

#scorer = make_scorer(r2_score)         #Initialize the scorer using make_scorer

#grid_obj = GridSearchCV(clf,parameters,scoring=scorer)   

#grid_fit = grid_obj.fit(X,y)        #Fit the gridsearch object with X_train,y_train



#best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object





#y_pred = best_clf.predict(X_test)
list_id=data_test['id'].tolist()
y_pred=np.array(y_pred).tolist()
d = {'id':list_id,'AveragePrice':y_pred}
df = pd.DataFrame(d)
df.head()
df.to_csv('finalfinal.csv', index=False)