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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

data_train.sample(3)

data_test.sample(3)
#Let's look at data description

data_train.describe() 


#let's check missing values in each columns
data_train.isna().sum()
# Age 177 , and Cabin 687 Values are missing 
# Female should have higher change of survival - lets see if thats right-
sns.barplot(x="Sex",y="Survived",data=data_train).set_title("Female percentage of survival {0}".format(data_train['Survived'][data_train.Sex=='female'].value_counts(normalize=True)[1]*100))
#over 74 of female passengers survived. Clearly a higher chance of survival.
#Let Draw a barplot between 'Embarked' and 'Survived' and compare it with 'Sex'
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)

#Looking at the Bar Plot , we can say male survival rate is less in each Embarked.
#Lets Draw a Bar plot between Pclass & Survived and Hue it on 'Sex'
sns.barplot(x='Pclass',y="Survived",hue="Sex",data=data_train)
# Here also Male survival rate is less in each Pclass.
#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=data_train)
#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=data_train)
data_train.head()
# Here i will impute missing values in Age and Cabin features 
# Ill use Knn for the imputation , coz data set is quite small and it will give me better imputation values.


#digitize age
#import Knn from fancyimpute
#Compare both values

from fancyimpute import KNN
#data_train['Age']=np.digitize(data_train['Age'],bins=[0,5,12,18,25,35,60,120])
features=['Age','Pclass','SibSp','Parch','Fare']
data_train_numeric=data_train[features].as_matrix()
data_test_numeric=data_test[features].as_matrix()
data_train_imputed=pd.DataFrame(KNN(6).complete(data_train_numeric),index=data_train.index)
data_test_imputed=pd.DataFrame(KNN(6).complete(data_test_numeric),index=data_test.index)
data_train['Age_imputed']=data_train_imputed.iloc[:,0]
data_test['Age_imputed']=data_train_imputed.iloc[:,0]


#Lets compare Age imputed values with Age column 

data_train[data_train['Age'].isnull()].head()
#Let's drop 'Age' column 
data_train=data_train.drop('Age',axis=1)
data_test=data_test.drop('Age',axis=1)


data_train['Age']=data_train['Age_imputed']
data_test['Age']=data_test['Age_imputed']
data_train=data_train.drop('Age_imputed',axis=1)
data_test=data_test.drop('Age_imputed',axis=1)
data_train.head()

#Funtion definiton for Age categorization 
def simplify_ages(df):
    
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df



def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['Title'] = df.Name.apply(lambda x: x.split(' ')[1])

    
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)
data_train.head()
#lets plot graph for Age,Cabin& fare 

#male- Bay and all female seems to have survival ratio
plt.figure(figsize=(15,8))
sns.barplot(x="Age", y="Survived", data=data_train)
sns.barplot(x="Fare", y="Survived", hue="Sex", data=data_train);
#Survival rate of males are increasing with Fare.
from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = [ 'Sex', 'Lname', 'Title','Age','Fare']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(data_train, data_test)
data_train.head()

from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived','Cabin', 'PassengerId'], axis=1)

y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
#import GridSearchCv 
from sklearn.model_selection import GridSearchCV
#Logestic regression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
param_grid={'C':[0.001,0.01,0.1,1,10,100]}
clf=LogisticRegression()
grid=GridSearchCV(clf,param_grid,cv=10,scoring='accuracy').fit(X_train,y_train)
y_pred=grid.predict(X_test)
acc_lreg=round(accuracy_score(y_pred,y_test)*100,2)
print("Accuracy Score for Logestic Regression {0}".format(acc_lreg))

#Support Vector Machine
from sklearn.svm import SVC
clf=SVC()
Cs=[0.001,0.01,0.1,1,10,]
gammas=[0.001,0.01,0.1,1]
param_grid={'C':Cs,'gamma':gammas}
grid=GridSearchCV(clf,param_grid,cv=10,scoring='accuracy').fit(X_train,y_train)
y_pred=grid.predict(X_test)
acc_svc=round(accuracy_score(y_pred,y_test)*100,2)
print("Accuracy Score for Support Vector Machine{0}".format(acc_svc))

#Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
param_grid={'max_depth':[2,4,6,8,10],'max_features':[2,3,4,5,6,7]}
grid=GridSearchCV(clf,param_grid,cv=10,scoring='accuracy').fit(X_train,y_train)
y_pred=grid.predict(X_test)
acc_dtree=round(accuracy_score(y_pred,y_test)*100,2)
print("Accuracy score for decision tree{0}".format(acc_dtree))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
param_grid={'max_depth':[2,4,6,8,10],'max_features':[2,3,4,5,6,7]}
grid=GridSearchCV(clf,param_grid,cv=10,scoring='accuracy').fit(X_train,y_train)
y_pred=grid.predict(X_test)
acc_rforest=round(accuracy_score(y_pred,y_test)*100,2)
print("RandomForestAccuracy Score{0} ".format(acc_rforest))
#KNN or KNearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
param_grid={'n_neighbors':[2,4,6,8,10],'weights':['uniform','distance']}
grid=GridSearchCV(clf,param_grid,cv=10,scoring='accuracy').fit(X_train,y_train)
y_pred=grid.predict(X_test)
acc_knn=round(accuracy_score(y_pred,y_test)*100,2)
print("KnearrestNeighbors Accuracy Score{0} ".format(acc_knn))

#Gradient Boost decision Tree
from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
param_grid={'max_depth':[2,4,6,8,10],'max_features':[2,3,4,5,6,7]}
grid=GridSearchCV(clf,param_grid,cv=10,scoring='accuracy').fit(X_train,y_train)
y_pred=grid.predict(X_test)
acc_gbdtree=round(accuracy_score(y_pred,y_test)*100,2)
print("GradientboostingClassifier accuracy Score {0}".format(acc_gbdtree))

model=pd.DataFrame({'model':['GradientboostingClassifier','KnearrestNeighbors','RandomForest','DecisionTreeClassifier','SupportVectormachine','LogisticRegression'] ,'Acc_Score':[acc_gbdtree,acc_knn,acc_rforest,acc_dtree,acc_svc,acc_lreg]})
model.sort_values('Acc_Score',ascending=False)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9,12], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)


predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
from sklearn.cross_validation import KFold

def run_kfold(clf):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf)

ids = data_test['PassengerId']
predictions = clf.predict(data_test.drop(['PassengerId','Cabin'], axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)
output.head()

