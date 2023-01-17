# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/itjen-data-science-competition/train.csv")

dataset.head()
dataset.info()
dataset.describe()
#FillAge = dataset['Age'].mean()

#let's drop rows with NA Age value

AgeNotNull = dataset[['Age','Pclass','Sex']].dropna()



Class1FemaleAgeMedian = np.median(((AgeNotNull.query('Pclass==1 and Sex=="female"')))['Age'])

Class2FemaleAgeMedian = np.median(((AgeNotNull.query('Pclass==2 and Sex=="female"')))['Age'])

Class3FemaleAgeMedian = np.median(((AgeNotNull.query('Pclass==3 and Sex=="female"')))['Age'])

Class1MaleAgeMedian = np.median(((AgeNotNull.query('Pclass==1 and Sex=="male"')))['Age'])

Class2MaleAgeMedian = np.median(((AgeNotNull.query('Pclass==2 and Sex=="male"')))['Age'])

Class3MaleAgeMedian = np.median(((AgeNotNull.query('Pclass==3 and Sex=="male"')))['Age'])



## Get the subset data and fillna with median

dataset1 = dataset.query('Pclass == 1 and Sex == "female"')

dataset1['Age'] = dataset1['Age'].fillna(Class1FemaleAgeMedian)



dataset2 = dataset.query('Pclass == 2 and Sex == "female"')

dataset2['Age'] = dataset2['Age'].fillna(Class2FemaleAgeMedian)



dataset3 = dataset.query('Pclass == 3 and Sex == "female"')

dataset3['Age'] = dataset3['Age'].fillna(Class3FemaleAgeMedian)



dataset4 = dataset.query('Pclass == 1 and Sex == "male"')

dataset4['Age'] = dataset4['Age'].fillna(Class1MaleAgeMedian)



dataset5 = dataset.query('Pclass == 2 and Sex == "male"')

dataset5['Age'] = dataset5['Age'].fillna(Class2MaleAgeMedian)



dataset6 = dataset.query('Pclass == 3 and Sex == "male"')

dataset6['Age'] = dataset6['Age'].fillna(Class3MaleAgeMedian)



## Merge all subsetted datasets and sort by PassengerID

dataset = pd.concat([dataset1,dataset2,dataset3,dataset4,dataset5,dataset6])



dataset = dataset.sort_values('PassengerId')



#dataset['Age'].fillna(FillAge,inplace=True)

dataset.info()
bins = [0,12,18, 30, 40, 50, 60, 70, 120]

labels = ['0-12','12-18','18-29', '30-39', '40-49', '50-59', '60-69', '70+']

dataset['agegroup'] = pd.cut(dataset.Age, bins, labels = labels,include_lowest = True)

dataset.head()
MostEmbarked = dataset['Embarked'].value_counts().idxmax()

dataset['Embarked'].fillna(MostEmbarked,inplace=True)

dataset.info()
MeanFare = dataset['Fare'].mean()

MeanFare
dataset['LogFare']=dataset['Fare']

dataset['LogFare']=dataset.LogFare.mask(dataset.LogFare == 0,MeanFare)

dataset['LogFare'] = np.log(dataset.LogFare)

dataset.info()
df = dataset[['Sex','agegroup','Parch','LogFare','SibSp','Embarked','Survived']]

import category_encoders as ce

ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)

df_OHE = ohe.fit_transform(df)

df_OHE.head()
df_OHE.info()
X_train = df_OHE.drop(['Survived'],axis=1)

y_train=df_OHE[['Survived']]
import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score



#create new a knn model

knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors

params_knn = {'n_neighbors':np.arange(1,25),

          'leaf_size':[1,2,3,5],

          'weights':['uniform', 'distance'],

          'algorithm':['auto', 'ball_tree','kd_tree','brute'],

          'n_jobs':[-1]}

#use gridsearch to test all values for n_neighbors

knn_gs = GridSearchCV(knn, params_knn, cv=5)

#fit model to training data

knn_gs.fit(X_train, y_train)
#save best model

knn_best = knn_gs.best_estimator_

#check best n_neigbors value

print(knn_gs.best_params_)
from sklearn.ensemble import RandomForestClassifier

#create a new random forest classifier

rf = RandomForestClassifier()

#create a dictionary of all values we want to test for n_estimators

params_rf = {'criterion':['gini','entropy'],

          'n_estimators':[10,15,20,25,30],

          'min_samples_leaf':[1,2,3],

          'min_samples_split':[3,4,5,6,7], 

          'random_state':[123],

          'n_jobs':[-1]}

#use gridsearch to test all values for n_estimators

rf_gs = GridSearchCV(rf, params_rf, cv=5)

#fit model to training data

rf_gs.fit(X_train, y_train)
#save best model

rf_best = rf_gs.best_estimator_

#check best n_estimators value

print(rf_gs.best_params_)
from sklearn.linear_model import LogisticRegression

#create a new logistic regression model

log_reg = LogisticRegression()

#fit the model to the training data

log_reg.fit(X_train, y_train)
from sklearn import svm

#making the instance

svm_model=svm.SVC()

#Hyper Parameters Set

params_svm = {'C': [6,7,8,9,10,11,12], 

          'kernel': ['linear','rbf']}

#Making models with hyper parameters sets

svm_gs = GridSearchCV(svm_model, param_grid=params_svm, n_jobs=-1,cv=5)

#Learning

svm_gs.fit(X_train,y_train)
#save best model

svm_best = svm_gs.best_estimator_

#check best n_estimators value

print(svm_gs.best_params_)
from sklearn.tree import DecisionTreeClassifier

#making the instance

dtree= DecisionTreeClassifier(random_state=1234)

#Hyper Parameters Set

params_dtree = {'max_features': ['auto', 'sqrt', 'log2'],

          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 

          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],

          'random_state':[123]}

#Making models with hyper parameters sets

dtree_gs = GridSearchCV(dtree, param_grid=params_dtree, n_jobs=-1,cv=5)

#Learning

dtree_gs.fit(X_train,y_train)
#save best model

dtree_best = dtree_gs.best_estimator_

#check best n_estimators value

print(dtree_gs.best_params_)
test_dataset = pd.read_csv("../input/itjen-data-science-competition/test.csv")

test_dataset.head()
#test_dataset['Age'].fillna(FillAge,inplace=True)



## Get the subset data and fillna with median

test_dataset1 = test_dataset.query('Pclass == 1 and Sex == "female"')

test_dataset1['Age'] = test_dataset1['Age'].fillna(Class1FemaleAgeMedian)



test_dataset2 = test_dataset.query('Pclass == 2 and Sex == "female"')

test_dataset2['Age'] = test_dataset2['Age'].fillna(Class2FemaleAgeMedian)



test_dataset3 = test_dataset.query('Pclass == 3 and Sex == "female"')

test_dataset3['Age'] = test_dataset3['Age'].fillna(Class3FemaleAgeMedian)



test_dataset4 = test_dataset.query('Pclass == 1 and Sex == "male"')

test_dataset4['Age'] = test_dataset4['Age'].fillna(Class1MaleAgeMedian)



test_dataset5 = test_dataset.query('Pclass == 2 and Sex == "male"')

test_dataset5['Age'] = test_dataset5['Age'].fillna(Class2MaleAgeMedian)



test_dataset6 = test_dataset.query('Pclass == 3 and Sex == "male"')

test_dataset6['Age'] = test_dataset6['Age'].fillna(Class3MaleAgeMedian)



## Merge all subsetted datasets and sort by PassengerID

test_dataset = pd.concat([test_dataset1,test_dataset2,test_dataset3,test_dataset4,test_dataset5,test_dataset6])



test_dataset = test_dataset.sort_values('PassengerId')



test_dataset.info()
test_dataset['agegroup'] = pd.cut(test_dataset.Age, bins, labels = labels,include_lowest = True)

test_dataset.head()
test_dataset['Embarked'].fillna(MostEmbarked,inplace=True)

test_dataset.info()
test_dataset['Fare'].fillna(MeanFare,inplace=True)

test_dataset.info()
test_dataset['LogFare']=test_dataset['Fare']

test_dataset['LogFare']=test_dataset.LogFare.mask(test_dataset.LogFare == 0,MeanFare)

test_dataset['LogFare'] = np.log(test_dataset.LogFare)

test_dataset.info()
X_test = test_dataset[['Sex','agegroup','Parch','LogFare','SibSp','Embarked']]

X_test_OHE = ohe.fit_transform(X_test)

X_test_OHE.head()
from sklearn.ensemble import VotingClassifier

#create a dictionary of our models

estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg),('svm',svm_best),('dtree',dtree_best)]

#create our voting classifier, inputting our models

ensemble = VotingClassifier(estimators, voting='hard')
#fit model to training data

ensemble.fit(X_train, y_train)
prediction=pd.DataFrame(ensemble.predict(X_test_OHE))

prediction
from sklearn.metrics import accuracy_score

actual_data = pd.read_csv("../input/testwithsurvived/test-with-survived.csv", sep=";")

true_data = pd.DataFrame(actual_data[['Survived']])

score = accuracy_score(true_data, prediction)

score
finalresult = pd.concat([test_dataset['PassengerId'],prediction],axis=1)

finalresult.columns = ['PassengerId','Survived']

finalresult.head()
finalresult.to_csv("final-result-ensemble-tuned-5-algorithms.csv", sep=',', encoding='utf-8',index=False)