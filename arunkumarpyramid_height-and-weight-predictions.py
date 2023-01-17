# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the Dataset

import pandas as pd

dataset = pd.read_csv("../input/weight-height/weight-height.csv")
# Displaying the head and tail of the dataset

dataset.head()
dataset.tail()
# Displaying the shape and datatype for each attribute

print(dataset.shape)

dataset.dtypes
# Displaying the describe of each attribute

dataset.describe()
# Histogram Visualisation For Height Attribute with distribution plot



import seaborn as sb

sb.distplot(dataset['Height'])
sb.distplot(dataset['Weight'])
# Checking the correlation between input and output attributes.

corr_value=dataset.corr()

sb.heatmap(corr_value,square=True)
# Displaying the Null or empty values 

dataset.info()
# Displaying the Null or empty values sum

dataset.isna().sum()
# encoding gender column

dataset['Gender'].unique()
dataset['Gender']=dataset['Gender'].map({'Male':0,'Female':1})

dataset['Gender'].unique()
# Displaying first 5 rows

dataset.head()
# Checking the outliers with each input attribute to output attribute.



plt.plot(dataset['Gender'],dataset['Weight'])

plt.title("Checking Outliers")

plt.xlabel("Gender")

plt.ylabel('Weight')

plt.show()
# Checking the outliers with Height input attribute and Weight output attribute.



plt.plot(dataset['Height'],dataset['Weight'])

plt.title("Checking Outliers")

plt.xlabel("Height")

plt.ylabel('Weight')

plt.show()
y=dataset['Weight'].values

x=dataset.drop(['Weight'],axis=1)
# Splitting dataset into train and test split.



train_size=0.80

test_size=0.20

seed=5

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=train_size,test_size=test_size,random_state=seed)
# Spot Checking and Comparing Algorithms Without Feature Scale

models=[]

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

models.append(('linear_reg',LinearRegression()))

models.append(('knn',KNeighborsRegressor()))

models.append(('SVR',SVR()))

models.append(("decision_tree",DecisionTreeRegressor()))



# Evaluating Each model

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

names=[]

predictions=[]

error='neg_mean_squared_error'

for name,model in models:

    fold=KFold(n_splits=10,random_state=0)

    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)

    predictions.append(result)

    names.append(name)

    msg="%s : %f (%f)"%(name,result.mean(),result.std())

    print(msg)

    



# Visualizing the Model accuracy

fig=plt.figure()

fig.suptitle("Comparing Algorithms")

plt.boxplot(predictions)

plt.show()
# Create Pipeline with Standardization Scale and models

# Standardize the dataset

from sklearn.pipeline import Pipeline

from sklearn. preprocessing import MinMaxScaler

pipelines=[]

pipelines.append(('scaler_lg',Pipeline([('scaler',MinMaxScaler()),('lg',LinearRegression())])))

pipelines.append(('scale_KNN',Pipeline([('scaler',MinMaxScaler()),('KNN',KNeighborsRegressor())])))

pipelines.append(('scale_SVR',Pipeline([('scaler',MinMaxScaler()),('SVR',SVR())])))

pipelines.append(('scale_decision',Pipeline([('scaler',MinMaxScaler()),('decision',DecisionTreeRegressor())])))



# Evaluate Pipelines

predictions=[]

names=[]

for name, model in pipelines:

    fold=KFold(n_splits=10,random_state=5)

    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)

    predictions.append(result)

    names.append(name)

    msg='%s : %f (%f)'%(name,result.mean(),result.std())

    print(msg)

    

#Visualize the compared algorithms

fig=plt.figure()

fig.suptitle("Algorithms Comparisions")

plt.boxplot(predictions)

plt.show()
# SVR Tuning

import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=MinMaxScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

kernel=['linear','poly','rbf','sigmoid']

c=[0.2,0.4,0.6,0.8,1.0]

param_grid=dict(C=c,kernel=kernel)

model=SVR()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Linear Regression Algorithm tuning





import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=MinMaxScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

param_grid=dict()

model=LinearRegression()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Ensemble and Boosting algorithm to improve performance





# Boosting methods

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

# Ensemble Bagging methods

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor

ensembles=[]

ensembles.append(('scaledAB',Pipeline([('scale',MinMaxScaler()),('AB',AdaBoostRegressor())])))

ensembles.append(('scaledGBR',Pipeline([('scale',MinMaxScaler()),('GBR',GradientBoostingRegressor())])))

ensembles.append(('scaledRF',Pipeline([('scale',MinMaxScaler()),('rf',RandomForestRegressor(n_estimators=10))])))

ensembles.append(('scaledETR',Pipeline([('scale',MinMaxScaler()),('ETR',ExtraTreesRegressor(n_estimators=10))])))

ensembles.append(('scaledRFR',Pipeline([('scale',MinMaxScaler()),('RFR',RandomForestRegressor(n_estimators=10))])))

# Evaluate each Ensemble Techinique

results=[]

names=[]

for name,model in ensembles:

    fold=KFold(n_splits=10,random_state=5)

    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)

    results.append(result)

    names.append(name)

    msg="%s : %f (%f)"%(name,result.mean(),result.std())

    print(msg)

    

# Visualizing the compared Ensemble Algorithms

fig=plt.figure()

fig.suptitle('Ensemble Compared Algorithms')

plt.boxplot(results)

plt.show()
# GradientBoostingRegressor Tuning



import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=MinMaxScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

learning_rate=[0.1,0.2,0.3,0.4,0.5]

n_estimators=[5,10,15,20,25,30,40,50,100,200]

param_grid=dict(n_estimators=n_estimators,learning_rate=learning_rate)

model=GradientBoostingRegressor()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# AdaBoostRegressor Tuning



import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=MinMaxScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

learning_rate=[0.1,0.2,0.3,0.4,0.5]

n_estimators=[5,10,15,20,25,30,40,50,100,200]

param_grid=dict(n_estimators=n_estimators,learning_rate=learning_rate)

model=AdaBoostRegressor()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Finalize Model

# we will finalize the gradient boosting regression algorithm and evaluate the model for house price predictions.



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

scaler=MinMaxScaler().fit(x_train)

scaler_x=scaler.transform(x_train)

model=GradientBoostingRegressor(random_state=5,n_estimators=50,learning_rate=0.1)

model.fit(scaler_x,y_train)



#Transform the validation test set data

scaledx_test=scaler.transform(x_test)

y_pred=model.predict(scaledx_test)
# Accuracy of algorithm

from math import sqrt

mse=mean_squared_error(y_test,y_pred)

rmse=np.sqrt(mse)

print("rmse",rmse)

r2=r2_score(y_test,y_pred)

print("mse",mse)

print("r2_score",r2)