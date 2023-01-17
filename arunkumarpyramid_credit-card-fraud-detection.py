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
# Importing the dataset

import pandas as pd

dataset = pd.read_csv("../input/creditcardfraud/creditcard.csv")
# Displaying the head and tail of the dataset

dataset.head()
dataset.tail()
# Column key for each attributes

dataset.columns
# Displaying the shape and data type for each attribute.



print(dataset.shape)

dataset.dtypes
# Displaying the describe statistics in each column



dataset.describe()
# Displaying the only non empty count

dataset.info()
# Displaying the Empty cell and total cell count in each attribute.



dataset.isna().sum()
# Checking percentage of each class rows.

non_fraud=round(dataset['Class'].value_counts()[0])/len(dataset)*100.0

fraud=round(dataset['Class'].value_counts()[1])/len(dataset)*100.0



print("Non-Fraud Transaction data percentage %f"%(non_fraud))

print("Fraud Trabsaction data percentage %f"%(fraud))
# Dispalying our class row count in bar plot manner.



colors=['red','blue']

import seaborn as sb

import matplotlib.pyplot as plt

sb.countplot('Class',data=dataset,palette=colors)

plt.title("Class Distribution 0-Not Fraud and 1- Fraud")
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.



# Lets shuffle the data before creating the subsamples



dataset = dataset.sample(frac=1)



# amount of fraud classes 492 rows.

fraud_dataset = dataset.loc[dataset['Class'] == 1]

non_fraud_dataset = dataset.loc[dataset['Class'] == 0][:492]



normal_distributed_dataset = pd.concat([fraud_dataset, non_fraud_dataset])



# Shuffle dataframe rows

new_dataset = normal_distributed_dataset.sample(frac=1, random_state=42)



new_dataset.head()
print('Distribution of the Classes in the subsample dataset')

print(new_dataset['Class'].value_counts()/len(new_dataset))







sb.countplot('Class', data=new_dataset, palette=colors)

plt.title('Equally Distributed Classes')

plt.show()
y=new_dataset['Class']

x=new_dataset.drop(['Class'],axis=1)
x=x.values

y=y.values

x[:1,:]
# Splitting the dataset into training and test set

train_size=0.80

test_size=0.20

seed=5

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=train_size,

                                               test_size=test_size,random_state=seed)
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
# tuning to Logistic Regression

import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

c=[0.01,0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

param_grid=dict(C=c)

model=LogisticRegression()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# tuning to Decision Tree Classification Algorithm

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
# Random forest Classifier Tuning

import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

n_estimators=[10,20,30,40,50,100,150,200]

param_grid=dict(n_estimators=n_estimators)

model=RandomForestClassifier()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Gradient Boosting Classifier Tuning

import numpy as np

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)

rescaledx=scaler.transform(x_train)

learning_rate=[0.01,0.05,0.1,0.2,0.3,0.4]

n_estimators=[10,20,30,40,50,100,150,200]

param_grid=dict(n_estimators=n_estimators,learning_rate=learning_rate)

model=GradientBoostingClassifier()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)

grid_result=grid.fit(rescaledx,y_train)



print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Finalize Model

# we finalized the Gradient Boosting Algorithm and evaluate the model for Hotel Booking Demand Dataset



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

scaler=StandardScaler().fit(x_train)

scaler_x=scaler.transform(x_train)

model=GradientBoostingClassifier(learning_rate=0.2,n_estimators=50)

model.fit(scaler_x,y_train)



#Transform the validation test set data

scaledx_test=scaler.transform(x_test)

y_pred=model.predict(scaledx_test)

y_trainpred=model.predict(scaler_x)
accuracy_mean=accuracy_score(y_train,y_trainpred)

accuracy_matric=confusion_matrix(y_train,y_trainpred)

print("train set %f"%accuracy_mean)

print("train set ",accuracy_matric)







accuracy_mean=accuracy_score(y_test,y_pred)

accuracy_matric=confusion_matrix(y_test,y_pred)

print("test set %f"%accuracy_mean)

print("test set ",accuracy_matric)