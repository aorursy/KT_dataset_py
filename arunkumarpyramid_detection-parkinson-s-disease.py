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
# importing dataset

import pandas as pd

dataset = pd.read_csv("../input/detection-of-parkinson-disease/parkinsons.csv")
# Displaying the head and tail of the dataset



dataset.head()
dataset.tail()
# Displaying the shape and datatype for each attribute

print(dataset.shape)

dataset.dtypes
# Dispalying the descriptive statistics describe each attribute



dataset.describe()
# Heatmap visulisation for each attribute coefficient correlation.

import seaborn as sb

corr_map=dataset.corr()

sb.heatmap(corr_map,square=True)
# Now visualise the heat map with correlation coefficient values for pair of attributes.

import matplotlib.pyplot as plt

import numpy as np



# K value means how many features required to see in heat map

k=10



# finding the columns which related to output attribute and we are arranging from top coefficient correlation value to downwards.

cols=corr_map.nlargest(k,'status')['status'].index



# correlation coefficient values

coff_values=np.corrcoef(dataset[cols].values.T)

sb.set(font_scale=1.25)

sb.heatmap(coff_values,cbar=True,annot=True,square=True,fmt='.2f',

           annot_kws={'size': 10},yticklabels=cols.values,xticklabels=cols.values)

plt.show()
# correlation coefficient values in each attributes.

correlation_values=dataset.corr()['status']

correlation_values.abs().sort_values(ascending=False)
# Checking null values

dataset.info()
# Checking null value sum

dataset.isna().sum()
# split the dataset into input and output attribute.



y=dataset['status']

cols=['MDVP:RAP','Jitter:DDP','DFA','NHR','MDVP:Fhi(Hz)','name','status']

x=dataset.drop(cols,axis=1)
# Splitting the dataset into trianing and test set



train_size=0.80

test_size=0.20

seed=5



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=train_size,test_size=test_size,random_state=seed)
# Spotcheck and compare algorithms with out applying feature scale.......



n_neighbors=5

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



# keeping all models in one list

models=[]

models.append(('LogisticRegression',LogisticRegression()))

models.append(('knn',KNeighborsClassifier(n_neighbors=n_neighbors)))

models.append(('SVC',SVC()))

models.append(("decision_tree",DecisionTreeClassifier()))

models.append(('Naive Bayes',GaussianNB()))



# Evaluating Each model

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

names=[]

predictions=[]

error='accuracy'

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
# Spot Checking and Comparing Algorithms With StandardScaler Scaler

from sklearn.pipeline import Pipeline

from sklearn. preprocessing import StandardScaler

pipelines=[]

pipelines.append(('scaled Logisitic Regression',Pipeline([('scaler',StandardScaler()),('LogisticRegression',LogisticRegression())])))

pipelines.append(('scaled KNN',Pipeline([('scaler',StandardScaler()),('KNN',KNeighborsClassifier(n_neighbors=n_neighbors))])))

pipelines.append(('scaled SVC',Pipeline([('scaler',StandardScaler()),('SVC',SVC())])))

pipelines.append(('scaled DecisionTree',Pipeline([('scaler',StandardScaler()),('decision',DecisionTreeClassifier())])))

pipelines.append(('scaled naive bayes',Pipeline([('scaler',StandardScaler()),('scaled Naive Bayes',GaussianNB())])))



# Evaluating Each model

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

names=[]

predictions=[]

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
# Decision Tree Tunning Algorithms

import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

param_grid=dict()

model=DecisionTreeClassifier()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Logistic Regression Tuning Algorithm

import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

c=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

param_grid=dict(C=c)

model=LogisticRegression()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Ensemble and Boosting algorithm to improve performance



#Ensemble

# Boosting methods

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

# Bagging methods

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

ensembles=[]

ensembles.append(('scaledAB',Pipeline([('scale',StandardScaler()),('AB',AdaBoostClassifier())])))

ensembles.append(('scaledGBC',Pipeline([('scale',StandardScaler()),('GBc',GradientBoostingClassifier())])))

ensembles.append(('scaledRFC',Pipeline([('scale',StandardScaler()),('rf',RandomForestClassifier(n_estimators=10))])))

ensembles.append(('scaledETC',Pipeline([('scale',StandardScaler()),('ETC',ExtraTreesClassifier(n_estimators=10))])))



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
# GradientBoosting ClassifierTuning

import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

n_estimators=[10,20,30,40,50,100,150,200,250,300]

learning_rate=[0.001,0.01,0.1,0.3,0.5,0.7,1.0]

param_grid=dict(n_estimators=n_estimators,learning_rate=learning_rate)

model=GradientBoostingClassifier()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Extra Trees Classifier Classifier Tuning

import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

n_estimators=[10,20,30,40,50,100,150,200]

param_grid=dict(n_estimators=n_estimators)

model=ExtraTreesClassifier()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Finalize Model

# we finalized the Extra Trees Classification Algoriothm and evaluate the model for Detection parkinsons disease



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

scaler=StandardScaler().fit(x_train)

scaler_x=scaler.transform(x_train)

model=ExtraTreesClassifier(n_estimators=30)

model.fit(scaler_x,y_train)



#Transform the validation test set data

scaledx_test=scaler.transform(x_test)

y_pred=model.predict(scaledx_test)

y_predtrain=model.predict(scaler_x)
accuracy_mean=accuracy_score(y_train,y_predtrain)

accuracy_matric=confusion_matrix(y_train,y_predtrain)

print("train set",accuracy_mean)

print("train set matrix",accuracy_matric)



accuracy_mean=accuracy_score(y_test,y_pred)

accuracy_matric=confusion_matrix(y_test,y_pred)

print("test set",accuracy_mean)

print("test set matrix",accuracy_matric)