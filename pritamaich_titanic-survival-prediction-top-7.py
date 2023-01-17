import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
training_raw = pd.read_csv('../input/titanic/train.csv')
training_raw
test_raw = pd.read_csv('../input/titanic/test.csv')
test_raw
training = training_raw.copy()
test = test_raw.copy()
training.info()
training.describe()
test.info()
test.describe()
sns.countplot(training['Survived'])
plt.show()
sns.countplot(training['Sex'])
plt.show()
sns.countplot(x = 'Sex',data = training, hue = 'Survived')
plt.show()
plt.figure(figsize=(10,5))
sns.distplot(training['Age'], kde = False, bins = 30)
plt.show()
training['Age_groups'] = pd.cut(x = training['Age'], bins = [0,18,60,100], labels = ['Child','Adult','Elderly'])
plt.figure(figsize = (10,5))
sns.countplot(x = 'Age_groups', data = training, hue = 'Survived', palette = 'rainbow')
plt.show()
plt.figure(figsize=(12,6))
sns.violinplot(y = 'Fare', x = 'Survived' , data = training)
plt.show()
plt.figure(figsize=(10,6))
sns.countplot(x = 'Pclass', data = training, hue = 'Survived', palette = 'Blues')
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(training.corr(), cmap = 'Spectral', annot = True)
plt.show()
plt.figure(figsize=(8,8))
correlation = pd.DataFrame()
correlation['correlation'] = training.corrwith(training['Survived']).sort_values(ascending = False)
sns.heatmap(correlation, cmap = 'Blues', annot = True)
plt.show()
plt.figure(figsize=(12,5))
training.corrwith(training['Survived']).sort_values().drop('Survived').plot(kind='bar', color = 'red')
plt.title('Correlation with Survived Feature (bar plot) ', fontsize= 15)
plt.show()
training.isnull().sum()

test.isnull().sum()
training = training.drop('Cabin', axis = 1)
test = test.drop('Cabin', axis = 1)
median = training['Age'].median()
training['Age'] = training['Age'].fillna(median)
training  = training.drop('Age_groups', axis = 1)
training.isnull().sum()
#Applying the same in test data

median = test['Age'].median()
test['Age'] = test['Age'].fillna(median)
test.isnull().sum()
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
training['Embarked'].value_counts()
training['Embarked'] = training['Embarked'].fillna('S')
training.select_dtypes(include = [np.object]) #Viewing the categorical feautures
training['Name']
"""
First splitting the names with ',' and taking the secong part. (Eg. : ' Mr. Owen Harris')
Then splitting the reamaining text with '.' and taking the first part(Eg. : ' Mr' )
Finally removing any space from begining or end with .strip() method

"""
title = [x.split(',')[1].split('.')[0].strip() for x in training['Name']]  
training['title'] = title
training['title'].unique()
training['title'].value_counts()
training['title'] = training['title'].map({'Mr':'Mr','Miss': 'Miss', 'Mrs': 'Mrs','Ms': 'Miss', 'Dr': 'Other', 'Rev': 'Other',
                                           'Col': 'Other', 'Mlle': 'Other', 'Major': 'Other', 'the Countess' : 'Other', 
                                           'Mme' : 'Other', 'Lady' : 'Mrs', 'Jonkheer': 'Other', 'Sir': 'Mr', 
                                           'Capt': 'Other', 'Don':'Other', 'Master': 'Master' })
training['title'].value_counts()
dummies = pd.get_dummies(training['title'])
training = pd.concat([dummies, training], axis = 1)

#We won't need the 'Name' and 'title' columns anymore

training = training.drop(['Name', 'title'], axis = 1)
"""
First splitting the names with ',' and taking the secong part. (Eg. : ' Mr. Owen Harris')
Then splitting the reamaining text with '.' and taking the first part(Eg. : ' Mr' )
Finally removing any space from begining or end with .strip() method

"""
title = [x.split(',')[1].split('.')[0].strip() for x in test['Name']]  
test['title'] = title
test['title'].unique()
test['title'].value_counts()
test['title'] = test['title'].map({'Mr':'Mr','Miss': 'Miss', 'Mrs': 'Mrs','Master': 'Master', 'Ms': 'Miss',
                                   'Dr': 'Other', 'Rev': 'Other', 'Col': 'Other', 'Dona':'Other' })
test['title'].value_counts()
dummies = pd.get_dummies(test['title'])
test = pd.concat([dummies, test], axis = 1)

#We won't need the 'Name' and 'title' columns anymore

test = test.drop(['Name', 'title'], axis = 1)

training.select_dtypes(include = np.object)
training['Ticket'].head(20)
training = training.drop('Ticket', axis = 1)
test = test.drop('Ticket', axis = 1)
training['Embarked'].value_counts()
dummies = pd.get_dummies(training['Embarked'])
training = pd.concat([dummies,training], axis = 1)
#Same for the test data

dummies = pd.get_dummies(test['Embarked'])
test = pd.concat([dummies,test], axis = 1)
#Dropping the original embarked column

training = training.drop('Embarked', axis = 1)
test = test.drop('Embarked', axis = 1 )
dummies = pd.get_dummies(training['Sex'])
training = pd.concat([dummies,training], axis = 1)
#Same for the test data

dummies = pd.get_dummies(test['Sex'])
test = pd.concat([dummies,test], axis = 1)
#Dropping the original Sex column

training = training.drop('Sex', axis = 1)
test = test.drop('Sex', axis = 1 )
training
training['Pclass'] = training['Pclass'].map({1:'1st_class', 2:'2nd_class', 3:'3rd_class'})

training['Pclass'].value_counts()
dummies = pd.get_dummies(training['Pclass'])
training = pd.concat([dummies,training], axis = 1)
# Same for test data

test['Pclass'] = test['Pclass'].map({1:'1st_class', 2:'2nd_class', 3:'3rd_class'})

dummies = pd.get_dummies(test['Pclass'])
test = pd.concat([dummies,test], axis = 1)
#Dropping the original Pclass column

training = training.drop('Pclass', axis = 1)
test = test.drop('Pclass', axis = 1 )
training['Parch'].value_counts()
training['SibSp'].value_counts()
#Seperating the two column values

Parch = training['Parch']

Sib = training['SibSp']
#Creating a new variable where we store the addition of the above two variables, i.e one full family 
Family = Parch + Sib
training['Family'] = Family
#Same for test set

Parch = test['Parch']

Sib = test['SibSp']

test['Family'] = Parch + Sib
#Dropping original Parch and SibSp columns from both

training = training.drop(['Parch', 'SibSp'], axis = 1)
test = test.drop(['Parch', 'SibSp'], axis = 1 )
test_id = test.copy()

test_id.to_csv('Test_with_Id.csv')
training = training.drop('PassengerId', axis = 1)
test = test.drop('PassengerId', axis = 1)
#Saving these datasets for final modeling

training.to_csv('Training_final.csv')
test.to_csv('Test_final.csv')
training_data = pd.read_csv('Training_final.csv')
testing_data = pd.read_csv('Test_final.csv')
#Assigning inputs and targets

x_train = training_data.drop('Survived', axis = 1)
y_train = training_data['Survived']

x_test = testing_data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
#Defining a method or function that will print the cross validation score for each model

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def model_report(cl):
    
    cl.fit(x_train, y_train)

    print('Cross Val Score: ',(cross_val_score(cl,x_train,y_train, cv=5).mean()*100).round(2))#using a 5-Fold cross validation

from sklearn.model_selection import GridSearchCV

#Defining a function that will calculate the best parameters and accuracy of the model based on those parameters
#Using GridSearchCV

def grid_search(classifier,parameters):
    
    grid = GridSearchCV(estimator = classifier,
                        param_grid = parameters,
                        scoring = 'accuracy',
                        cv = 5,
                        n_jobs = -1
                        )
    
    grid.fit(x_train,y_train)

    print('Best parameters: ', grid.best_params_) #Displaying the best parameters of the model

    print("Accuracy: ", ((grid.best_score_)*100).round(2))#Accuracy of the model based on those parameters
from sklearn.ensemble import RandomForestClassifier

param_rf = {
    'n_estimators': [10,50,100,500,1000],
    'min_samples_leaf': [1,10,20,50]
    }
rf = RandomForestClassifier(random_state = 0)
grid_search(rf,param_rf)
# Let's train our model using those parameters

rf = RandomForestClassifier(min_samples_leaf = 10, n_estimators = 100)
rf.fit(x_train, y_train)

model_report(rf)
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = list(range(5,10))#This is basically the value of k
                   
param_knn = {
    'n_neighbors' : n_neighbors,
    'p' : [1,2]
    
    }

knn = KNeighborsClassifier(algorithm ='auto', n_jobs = -1)
grid_search(knn,param_knn)
# Let's train our model using those parameters

knn = KNeighborsClassifier(n_neighbors = 6, p = 2)

knn.fit(x_train, y_train)

model_report(knn)
from sklearn.svm import SVC

param_svc = {
    'C': [0.1, 1, 10, 100],  
    'gamma': [0.0001, 0.001, 0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], 
    'kernel': ['linear','rbf']
    }
svc = SVC()

grid_search(svc,param_svc)
#Let's train our model using these parameters

svc = SVC( C = 100, gamma = 0.001, kernel = 'rbf')

svc.fit(x_train, y_train)

model_report(svc)
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state = 42, learning_rate = 0.01)

model_report(gb)
from xgboost import XGBClassifier

param_xg = {
    'learning_rate' : [0.1,0.2,0.01,0.02],
    'booster' : ['gbtree', 'gblinear'],
    'min_child_weight' : [3,4,5,6,7],
    'max_depth' : [3,4,5,6,7,8],
    'gamma' : [0.1,0.2,0.01,0.02],
    'subsample' : [0.5,0.6,0.7]
}

xg = XGBClassifier()

grid_search(xg,param_xg)


xg = XGBClassifier(learning_rate = 0.1, booster = 'gbtree', 
                   min_child_weight = 3, gamma = 0.2, max_depth = 8, subsample = 0.7)

model_report(xg)
y_pred = xg.predict(x_test)
submission = pd.DataFrame({'PassengerId': test_id['PassengerId'],'Survived':y_pred})

submission.to_csv('xgboost_submission.csv', index=False)
