# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
import re # text processing
import warnings
warnings.filterwarnings('ignore') # supress warnings
sns.set_style('whitegrid')

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Store passenger ID for easy access
PassengerId = test_data['PassengerId']
type(train_data)
train_data.head()
train_data.shape
train_data.info()
#checking for total null values in each column
train_data.isnull().sum()
f,ax=plt.subplots(1,2,figsize=(18,8))
train_data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survivors in Titanic')
sns.countplot('Survived', data=train_data, hue="Survived", ax=ax[1])
ax[1].set_title('Survivors in Titanic')
plt.show()
train_data.groupby(['Sex','Survived'])['Survived'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(color=['#CD7F32','#FFDF00'],ax=ax[0])
ax[0].set_title('Survived')
sns.countplot('Sex',hue='Survived',data=train_data,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
train_data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number Of Passengers Who Boarded Titatnic By Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=train_data,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()
f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=train_data,hue='Embarked', ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=train_data,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=train_data,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=train_data,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
full_data = [train_data, test_data]

# Derived Feature Has_Cabin that tells whether a passenger had a cabin on the Titanic or not
train_data['Has_Cabin'] = train_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_data['Has_Cabin'] = test_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Derived Feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
# Derived feature IsAlone from another Derived Feature: FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#Imputing Missing Values
# Replacing NULL values with most occuring value (mode) in the Embarked column
dataset['Embarked'].isnull().sum()
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
# Replacing NULL values with the median value in the Fare column
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train_data['Fare'].median()) 
    
# Replacing NULL values for Age column
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    # Generate random numbers between (mean — std) and (mean + std)
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Fill NaN values in Age column with random values generated
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)    
    
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Derived feature Title containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    
# Group all non-common titles into one single grouping "Other"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    # Fixing misspelled titles
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
       
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train_data = train_data.drop(drop_elements, axis=1)
test_data  = test_data.drop(drop_elements, axis=1)
# generate Pearson Correlation Heatmap
colormap = plt.cm.viridis
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]

# Create training and test sets from training data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
# Logistic Regression Hyper parameter Tuning
param_grid = {'C':[0.001,0.01,0.1,1,10,100,1000],
             'penalty': ['l1','l2']}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X, y)
print("best C", logreg_cv.best_params_)
print("best score", logreg_cv.best_score_)
# Logistic Regression Model Fitting and Performance Metrics
logreg = LogisticRegression(C=.1, penalty='l2')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
logreg_acc_score = round(logreg.score(X_train, y_train) * 100, 2)
print("***Logistic Regression***")
print("Accuracy Score:", logreg_acc_score)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
y_pred_prob = logreg.predict_proba(X_test)[:,1]
print("ROC_AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))
# KNN Hyper parameter Tuning
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)
print("neighbors", knn_cv.best_params_)
print("best score", knn_cv.best_score_)
# KNN  Model Fitting and Performance Metrics
knn = KNeighborsClassifier(n_neighbors = 25)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn_acc_score = round(knn.score(X_train, y_train) * 100, 2)
print("***K Nearest Neighbors***")
print("Accuracy Score:", knn_acc_score)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
y_pred_prob =knn.predict_proba(X_test)[:,1]
print("ROC_AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))
# Decision Tree Hyper parameter Tuning
param_grid = {'max_depth': np.arange(1, 20)}
decision_tree = DecisionTreeClassifier()
decision_tree_cv = GridSearchCV(decision_tree, param_grid, cv=5)
decision_tree_cv.fit(X, y)
print("best params", decision_tree_cv.best_params_)
print("best score", decision_tree_cv.best_score_)
# Decision Tree Model Fitting and Performance Metrics
decision_tree = DecisionTreeClassifier(max_depth=3)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
decision_tree_acc_score = round(decision_tree.score(X_train, y_train) * 100, 2)
print("***Decision Tree***")
print("Accuracy Score:", decision_tree_acc_score)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
y_pred_prob = decision_tree.predict_proba(X_test)[:,1]
print("ROC_AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))

# Random Forest Hyper parameter Tuning
param_grid = {'n_estimators': np.arange(1, 100)}
random_forest = RandomForestClassifier()
random_forest_cv = GridSearchCV(random_forest, param_grid, cv=5)
random_forest_cv.fit(X, y)
print("best params", random_forest_cv.best_params_)
print("best score", random_forest_cv.best_score_)
# Random Forest  Model Fitting and Performance Metrics
random_forest = RandomForestClassifier(n_estimators=73, max_depth=6, max_features="sqrt")
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
random_forest_acc_score = round(random_forest.score(X_train, y_train) * 100, 2)
print("***Random Forest***")
print("Accuracy Score:", random_forest_acc_score)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
y_pred_prob = random_forest.predict_proba(X_test)[:,1]
print("ROC_AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))

