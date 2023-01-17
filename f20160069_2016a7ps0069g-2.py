import numpy as np

import pandas as pd



data = pd.read_csv('../input/train.csv')

tdata=pd.read_csv('../input/test.csv')
data.head()
data.info()
import matplotlib.pyplot as plt

import seaborn as sns

#TODO

%matplotlib inline

corr = data.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True,annot=True, linewidths=.5, cbar_kws={"shrink": .5})





plt.show()


data.describe()
tdata.describe()

newvar=data['4046']*data['Total Volume']

newvar1=data['Small Bags']*data['Total Bags']

data['newvar']=newvar

data['newvar1']=newvar1

newdata = pd.get_dummies(columns=["type"],data=data)

newdata1=pd.get_dummies(columns=["year"],data=newdata)

#newdata=data

#newdata1.info()
newtvar=tdata['4046']*tdata['Total Volume']

newtvar1=tdata['Small Bags']*tdata['Total Bags']

tdata['newtvar']=newtvar

tdata['newtvar1']=newtvar1

tnewdata = pd.get_dummies(columns=["type"],data=tdata)

tnewdata1=pd.get_dummies(columns=["year"],data=tnewdata)
corr = newdata1.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True,annot=True, linewidths=.5, cbar_kws={"shrink": .5})





plt.show()
from sklearn.preprocessing import MinMaxScaler

m = MinMaxScaler()

m.fit_transform(newdata1)

X = newdata1.drop(['AveragePrice'],axis=1)

y = newdata1['AveragePrice'].tolist()

m.fit_transform(tnewdata1)

xt = tnewdata1

#yt = tnewdata1['AveragePrice'].tolist()
from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import explained_variance_score



def performance_metrics(y_true,y_pred):

    rmse = mean_squared_error(y_true,y_pred)

    r2 = r2_score(y_true,y_pred)

    explained_var_score = explained_variance_score(y_true,y_pred)

    

    return rmse,r2,explained_var_score

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from math import sqrt

lr = LinearRegression()



X_train =X

#X_train = X_train.reshape(-1,1)



y_train =y



X_test = xt

#X_test = X_test.reshape(-1,1)



y_test = y





#lr.fit(X_train,y_train)

#pred = lr.predict(X_test)

#print(sqrt(mean_squared_error(y_test,pred)))

#print( performance_metrics(y_test,pred))
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer





#TODO

clf = GradientBoostingRegressor(n_estimators = 500,max_depth=10)        #Initialize the classifier object



parameters = {'loss':['ls', 'lad', 'huber', 'quantile'],'learning_rate':[0.01,0.05,0.1,0.5,1]}    #Dictionary of parameters



scorer = make_scorer(r2_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer,cv=4,n_jobs=5)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



#unoptimized_predictions = (clf.fit(tr_X, tr_Y)).predict(te_x)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_test)        #Same, but use the best estimator

#print( performance_metrics(y_train,optimized_predictions))

#acc_unop = r2_score(te_Y, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

#acc_op = r2_score(te_Y, optimized_predictions)*100         #Calculate accuracy for optimized model

print('12')

#print("Accuracy score on unoptimized model:{}".format(acc_unop))

#print("Accuracy score on optimized model:{}".format(acc_op))

import csv

#print( performance_metrics(y_train,optimized_predictions))

with open('out.csv', 'w') as myfile:

    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,delimiter='\n')

    wr.writerow(optimized_predictions)