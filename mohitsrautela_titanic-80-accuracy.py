from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")
import warnings
warnings.filterwarnings('ignore')
# Importing the Required Libraries
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
pd.set_option('display.max_columns',100)
pd.set_option('display.max_row',100)
pd.options.display.max_columns=100
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
train.head()
train.info()
test.head()
test.info()
print("Train Dimensions:",train.shape)
print("Test Dimensions:",test.shape)
train.describe(percentiles=[.25,.5,.75,.90,.95,.99])
test.describe(percentiles=[.25,.5,.75,.90,.95,.99])
# Train Data Set Missing Value
msno.bar(train)

# Test Data Set Missing Value

msno.bar(test)
missing=pd.concat([train.isnull().sum()/train.shape[0]*100,test.isnull().sum()/train.shape[0]*100],axis=1,keys=['Train_Datset_Percentage','Test_Dataset_Percentage'])
print(missing)    
# Class vs Survived
print(train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean())
# Gender vs Survived
print(train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean())
# Sibsp vs Survived
print(train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean())
# Parch vs Survived
print(train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean())
# We can change the Missing Value with Median or we can check the Average age by the Passenger Class and then impute the Missing Value
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
plt.show()
#train['Age']=train['Age'].fillna(train['Age'].median())
#test['Age']=test['Age'].fillna(test['Age'].median())

def age_impute(cols):
    Age=cols[0]
    global Pclass
    Pclass=cols[1]
   
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        if Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age
# Applying the Age_Impute Function
train['Age']=train[['Age','Pclass']].apply(age_impute,axis=1)
test['Age']=train[['Age','Pclass']].apply(age_impute,axis=1)
# Checking for any Missing Value in the Age Column
display(train.Age.isnull().sum())
display(test.Age.isnull().sum())
#Imputing the Missing Value with Mode as it is a Categorical Variable
train['Embarked']=train.Embarked.fillna(train.Embarked.mode()[0])

# Imputing the Missing Fare Value with median
test['Fare']=test.Fare.fillna(test.Fare.median())


plt.figure(figsize=(8,5))
sns.countplot('Survived',hue='Sex',data=train,palette='dark')
plt.show()
plt.figure(figsize=(20,6))
sns.violinplot(x='Sex',y='Age', hue='Survived',data=train, split=True,palette={0:'r',1:'g'})
plt.show()
plt.figure(figsize=(9,6))
sns.barplot(y="Survived", x="Sex", hue="Pclass", data=train, palette="gist_rainbow_r")
plt.show()
plt.figure(figsize=(8,5))
sns.violinplot(x="Survived", y = "Age",data = train,palette='cool',size=6)
plt.show()
plt.figure(figsize=(8,5))
sns.barplot('Embarked','Survived',data=train,palette='ocean')
plt.show()
fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=train, split=True, palette={0: "r", 1: "g"})
plt.show()
sns.pairplot(train)
plt.show()

# introducing other features based on the family size
train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
test['FamilySize'] = test['Parch'] + test['SibSp'] + 1
    

train['Singleton'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train['SmallFamily'] = train['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
train['LargeFamily']=train['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    

test['Singleton'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)
test['SmallFamily'] = test['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
test['LargeFamily']=test['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    
    
plt.figure(figsize=(9,6))
sns.boxplot(y="Age",x="FamilySize",hue="Survived", data=train,palette='spring_r')
plt.show()
train_test_data=[train,test]
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
 
# Here In dataset all the values is mapped in train and test data set
train_test_data=[train,test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()

test['Title'].value_counts()
# Mapping the Tit
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Royalty",
    "Lady" : "Royalty"
}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(Title_Dictionary)
# encoding in dummy variable
titles_dummies = pd.get_dummies(train['Title'], prefix='Title')
train = pd.concat([train, titles_dummies], axis=1)
    
titles_dummies = pd.get_dummies(test['Title'], prefix='Title')
test = pd.concat([test, titles_dummies], axis=1)
map_item={'male':0,'female':1}
train['Sex']=train['Sex'].map(map_item)
test['Sex']=test['Sex'].map(map_item)
# Removing the Unnecssary Columns
train=train.drop(['Name','Ticket','Cabin','Title','SibSp','Parch','PassengerId'],axis=1)
test=test.drop(['Name','Ticket','Cabin','Title','SibSp','Parch'],axis=1)
PasseengerID_test=test.pop('PassengerId')

train.head()
train.columns
train.info()
scaler = StandardScaler()

train[['Pclass','Age','Fare','FamilySize']] = scaler.fit_transform(train[['Pclass','Age','Fare','FamilySize']])
train.head()
test[['Pclass','Age','Fare','FamilySize']] = scaler.transform(test[['Pclass','Age','Fare','FamilySize']])
Y_train=train.pop('Survived')
train.shape, test.shape,Y_train.shape
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
# Selecting the Significant Features using RFE
from sklearn.feature_selection import RFE
rfe = RFE(LR, 11)             # running RFE with 11 variables as output
rfe = rfe.fit(train, Y_train)
list(zip(train.columns,rfe.support_,rfe.ranking_))
col = train.columns[rfe.support_]
col
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
# Fitting the model on the Train dataset.
LR.fit(train,Y_train)
# Making Predictions on the Test Data set
y_pred = LR.predict(test)
# Calculating the Accuracy of the model.

print("Accuracy:",round(LR.score(train, Y_train)*100,2))

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = SVC()

scoring = 'accuracy'
score = cross_val_score(clf, train[col], Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100, 2)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
scoring = 'accuracy'
score = cross_val_score(clf, train[col], Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Random Forest Score
round(np.mean(score)*100, 2)
# Using SVM to predict the test set as we are getting Better result from there
# Hypertuning the Parameter

from sklearn.model_selection import GridSearchCV 
  
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(train[col], Y_train)
# print best parameter after tuning 
print(grid.best_params_) 
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 
clf = SVC(C= 10, gamma= 0.01, kernel= 'rbf')

clf.fit(train[col], Y_train)

# Making Predictions on the Test Data set
y_pred = clf.predict(test[col])
submission = pd.DataFrame({
        "PassengerId": PasseengerID_test,
        "Survived": y_pred
    })

filename = 'Titanic Predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
from IPython.display import Image
Image(url= "https://images.unsplash.com/photo-1543118141-8598f6bfbc0a?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60")

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=11, units=11, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=9, kernel_initializer="uniform"))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation="relu", units=3, kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()

train.shape
Y_train.values
features=train[col].values

history=classifier.fit(features, Y_train.values, batch_size = 10, epochs = 100,
    validation_split=0.1,verbose = 1,shuffle=True)
Y_pred = classifier.predict(test[col])
Y_pred=Y_pred.round()
submission = pd.DataFrame({
        "PassengerId": PasseengerID_test,
        "Survived": y_pred
    })

filename = 'Titanic Predictions_1_Deep_Learning.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
