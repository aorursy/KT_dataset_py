import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

%matplotlib inline
train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')
train.head() 
train.describe() #gives mean, stdev etc
train.info()
test.info()
train.dtypes
test.dtypes
train.isnull().values.any()
test.isnull().values.any()
train.duplicated().sum()
import seaborn as sns

fig, axs = plt.subplots(figsize=(10, 8))

corr = train.corr()

sns.heatmap(corr, center=0)
corr
X_train=train.drop('AveragePrice',axis=1)



y_train=train['AveragePrice']



X_train.head()
y_train.describe()
X_test = test
# Now for Linear regression



from sklearn import linear_model



y_train = np.array(y_train)

X_train = np.array(X_train)



X_test = np.array(X_test)
X_train
y_train
X_test
regr = linear_model.LinearRegression()

regr.fit(X_train, y_train) # where the training happens
y_pred = regr.predict(X_test)
#Now using Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators=1000,max_depth=100,n_jobs=1).fit(X_train,y_train)



y_pred = regressor.predict(X_test)
import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn import model_selection, preprocessing

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder



#model = xgb.XGBRegressor()

#model.fit(X_train,y_train)
y_pred = model.predict(X_test)
ID = X_test[:,0] #this is the ID: all entries of 0th column



a = np.array(ID)

b = np.array(y_pred)

p = [a,b]

pd.DataFrame(p).transpose().to_csv("Kag2_3.csv", index = 0)
from sklearn.metrics import explained_variance_score,mean_squared_error,r2_score





def performance_metrics(y_true,y_pred):

    rmse = mean_squared_error(y_true,y_pred)

    r2 = r2_score(y_true,y_pred)

    explained_var_score = explained_variance_score(y_true,y_pred)

    

    return rmse,r2,explained_var_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_pred = train_test_split(X_train,y_train,test_size = 0.33,random_state=42)





#TODO

clf = RandomForestRegressor()        #Initialize the classifier object



parameters = {'max_depth':range(10,1000,10),'n_estimators': range(10,100,10)}    #Dictionary of parameters



scorer = make_scorer(r2_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_test)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_test)        #Same, but use the best estimator



acc_unop = r2_score(y_pred, unoptimized_predictions)       #Calculate accuracy for unoptimized model

acc_op = r2_score(y_pred, optimized_predictions)         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))
y_pred = best_clf.predict(test)
ID =test["id"] 



a = np.array(ID)

b = np.array(y_pred)

p = [a,b]

pd.DataFrame(p).transpose().to_csv("Kag.csv", index = 0)
y_pred_op = best_clf.predict(X_train)



rmse_op,r2_score_op,explained_var_score_op = performance_metrics(y_train,y_pred_op)



op_params = best_clf.get_params()



print("Root mean squared error:{} \nR2-score:{} \nExplained variance score:{}".format(rmse_op,r2_score_op,explained_var_score_op))

print("\n\nOptimal parameter values:{}".format(op_params))