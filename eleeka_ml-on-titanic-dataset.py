import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,StratifiedKFold,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

# Suppress Warnings:-
import warnings
warnings.filterwarnings('ignore')
# Getting train and test dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine = [train,test]  # we will use this dataset while creating new features or during imputing missing values
print(train.shape)
print(test.shape)
train.head()
# There are 5 integer variable, 2 float and 5 string variales
print(train.dtypes.value_counts())
print('***'*20)
train.info() 

# Find out features with missing values
train.isnull().sum()
# On the given train dataset we will see how many survived
train['Survived'].value_counts()
# Let's plot the above data
sns.countplot('Survived',data = train)
train['Sex'].value_counts()
sns.countplot('Sex', data = train)
# Explore Sex Vs Survived
train[['Sex','Survived']].groupby('Sex').mean()
# Visualize the survival probability on Sex 
sns.catplot(x = 'Sex', y = 'Survived', data = train, kind = 'bar')
plt.ylabel('Survival Probability')
sns.countplot(x= 'Sex', hue = 'Survived', data = train)
# Get the break up of 891 across 3 classes
train['Pclass'].value_counts()
# plot the data
sns.countplot('Pclass',data = train)
# survival probability 
train[['Pclass','Survived']].groupby('Pclass').mean()
# plot the data
sns.catplot('Pclass', y = 'Survived', data = train, kind = 'bar')
plt.ylabel('Survival possibility')
sns.countplot(x='Pclass', hue = 'Survived',data = train)
sns.catplot(x='Pclass', y = 'Survived',hue = 'Sex',data = train, kind = 'point')
# Get the count of embarked feature
train['Embarked'].value_counts()
sns.countplot('Embarked', data = train)
# Explore Embarked vs Survived
train[['Embarked','Survived']].groupby('Embarked').mean()
sns.catplot('Embarked','Survived',data = train, hue = 'Sex', kind = 'point')
# Age distribution
print("Oldest Passenger's age:", train['Age'].max())
print("Youngest Passenger's age:", train['Age'].min())
print ("Average Age:",round(train['Age'].mean(),2))
# Visualize Age distribution
train['Age'].plot(kind = 'hist',)
plt.xlabel('Age')
# Explore 'Age' vs 'Survived'
g = sns.FacetGrid(train, col = 'Survived')
g.map(plt.hist, 'Age')
sns.kdeplot(train[train['Survived']==0]['Age'], shade = True, color = 'red')
sns.kdeplot(train[train['Survived']==1]['Age'],shade = True, color = 'blue')
plt.legend(['Not Survived','Survived'])
plt.title('Age Vs Survived')
plt.xlabel('Age')
# continue exploring the distribution of age with other features.
g = sns.FacetGrid(train,col = 'Survived', row = 'Sex')
g.map(plt.hist, 'Age')
g = sns.FacetGrid(train,col = 'Survived', row = 'Pclass')
g.map(plt.hist, 'Age')
g = sns.FacetGrid(train,col = 'Survived', row = 'Embarked')
g.map(plt.hist, 'Age')
train[['Name','Age']][train['Age']== 80]
train['Fare'].plot(kind = 'hist')
train['Fare'].max()
train[train['Fare']>500]
g = sns.FacetGrid(train,col = 'Survived')
g.map(plt.hist,'Fare')
train['Parch'].value_counts()
sns.countplot(x = 'Parch', data = train)
# Explore Parch Vs Survived
train[['Parch','Survived']].groupby('Parch').mean().sort_values(by = 'Survived',ascending = False)
sns.catplot('Parch','Survived',data = train, kind = 'point')
plt.ylabel('Survival Probability')
train['SibSp'].value_counts()
# Plot the data
sns.countplot('SibSp',data = train)
train[['SibSp','Survived']].groupby('SibSp').mean().sort_values(by = 'Survived',ascending = False)
# plot
sns.catplot('SibSp','Survived',data = train, kind = 'point')
sns.heatmap(train.corr(),annot = True)
# Let's view our training dataset
train.head()
# we need to create new feature for both train and test data:
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.')
# we will replace some of the titles to the most common titles
for dataset in combine:
    dataset['Title'].replace(['Capt','Col','Don','Jonkheer','Major','Rev','Sir','Countess','Dr'], 'Others',inplace=True)
    dataset['Title'].replace(['Lady','Miss','Mlle','Mme','Mrs','Ms'],'Miss/Mrs', inplace = True)
# Let's review the titles created.
train['Title'].unique()
pd.crosstab(train['Title'],train['Sex'])
# Calculate the average age for the designated title
train[['Title','Age']].groupby('Title').mean()
# Create a new feature 'Family' which will give us the total number of family members onboard
for dataset in combine:
    dataset['Family']= dataset['SibSp'] + dataset['Parch']+1
train['Family'].value_counts()
# Creating another variable which will tell us if the person was travelling alone or not.
for dataset in combine:
    dataset['Alone'] = dataset['Family'].map(lambda x: 1 if x == 1 else 0)
# Creating a function to fill in the missing age based on the average age as per title,
# which we have calculated earlier
def impute_age(cols):
    age = cols[0]
    title = cols[1]
    if pd.isnull(age):
        if title == 'Mr':
            return 32
        elif title == 'Master':
            return 5
        elif title == 'Miss/Mrs':
            return 28
        else:
            return 45
    else:
        return age
        
# Calling the above function to fill up missing age for both train and test set
for dataset in combine:
    dataset['Age']= dataset[['Age','Title']].apply(impute_age, axis = 1)
# Validating if the missing age has been updated.
print(train.isnull().sum())
print('*'*40)
print(test.isnull().sum())
#  Embarked
# fill up the missing data with S port, as the most of the passenger boarded from S port. 
train['Embarked'].fillna('S', inplace = True)
# Fare
test[test['Fare'].isnull()]
test['Fare'].fillna(test['Fare'].mode()[0], inplace = True)
test.isnull().sum()
# Explore title vs Survived
train[['Title', 'Survived']].groupby('Title').mean().sort_values(by = 'Survived',ascending = False)
sns.catplot('Title',y = 'Survived',data = train, kind = 'bar')
train[['Family','Survived']].groupby('Family').mean().sort_values(by = 'Survived', ascending = False)
sns.catplot(x= 'Family',y='Survived', data = train,kind = 'bar')
# Lets look at the dataset with added features
train.head()
train.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1, inplace = True)

PassengerId = test['PassengerId']
test.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1, inplace = True)
print(train.head())
print(test.head())
cols = ['Pclass','Sex','Embarked','Title']
train = pd.get_dummies(train,columns=cols,drop_first= True)
test = pd.get_dummies(test,columns=cols,drop_first= True)
# Age and Fare features are continous variable so, let's create bins for these two variables.
# For train set
train['Age_bin'] = pd.cut(train['Age'].astype(int),5)
train['Fare_bin'] = pd.qcut(train['Fare'],4)

# For test set:
test['Age_bin'] = pd.cut(test['Age'].astype(int),5)
test['Fare_bin'] = pd.qcut(test['Fare'],4)
   
train['Age_bin'].value_counts()
train['Fare_bin'].value_counts()
# Create 'Age_band' feature based on the bins
def age_band(dataset):
    dataset['Age_band']=0
    dataset.loc[dataset['Age']<=16,'Age_band']=0
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32),'Age_band']=1
    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48),'Age_band']=2
    dataset.loc[(dataset['Age']>48) & (dataset['Age']<64),'Age_band'] = 3
    dataset.loc[dataset['Age']>64,'Age_band']=4
# Calling the function on both train and test set
age_band(train)
age_band(test)
# Create 'Fare_cat' feature based on the bins
def fare_cat(dataset):
    dataset['Fare_cat']=0
    dataset.loc[dataset['Fare']<= 7.91, 'Fare_cat']=0
    dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare']<=14.45),'Fare_cat']=1
    dataset.loc[(dataset['Fare']>14.45) & (dataset['Fare']<=31),'Fare_cat']=2
    dataset.loc[(dataset['Fare']>31) & (dataset['Fare']<=513),'Fare_cat'] = 3
   
# Calling the function on both train and test set
fare_cat(train)
fare_cat(test)
train.head()
train.columns
# Dropping features
drop_features = ['Age','Fare','Age_bin','Fare_bin']
train.drop(columns = drop_features,axis = 1, inplace = True)
test.drop(columns = drop_features,axis = 1, inplace = True)
train.head()
# Create our feature matrix X and response vector ý
X = train.drop('Survived',axis = 1)
y = train['Survived']
kfold = StratifiedKFold(n_splits= 10)
classifiers = [LogisticRegression(),KNeighborsClassifier()
               ,DecisionTreeClassifier(),RandomForestClassifier(n_estimators=100)
               ,GradientBoostingClassifier(),SVC()]
clf_scores=[]
for classifier in classifiers:
    scores = cross_val_score(classifier,X,y,cv = kfold,scoring = 'accuracy')
    clf_scores.append(scores)

cv_mean =[]
cv_std = []
for score in clf_scores:
    cv_mean.append(score.mean())
    cv_std.append(score.std())
model_names = ['Logistic Regression','KNN','Decision Tree','Random Forest','Gradient Boosting','SVC']
Model_evaluation_scores = pd.DataFrame({'Algorithms': model_names,'Mean Score':cv_mean,'Std':cv_std})
Model_evaluation_scores
# Visualize the above scores:-
sns.barplot(x = 'Mean Score',y= 'Algorithms',data = Model_evaluation_scores,palette= 'Set1',)
plt.title('Cross validation scores')
rfc = RandomForestClassifier()
n_estimators = range(100,500,100)
param_grid = {'n_estimators':n_estimators}
grid = GridSearchCV(rfc,param_grid,scoring = 'accuracy',cv =10,verbose=1)
grid.fit(X,y)
print(grid.best_score_)
print(grid.best_params_)
gbc = GradientBoostingClassifier()
param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
grid = GridSearchCV(gbc,param_grid,cv = 10,scoring = 'accuracy',verbose = 1)
grid.fit(X,y)
print(grid.best_score_)
print(grid.best_params_)

svc = SVC(probability= True)
param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}
grid = GridSearchCV(svc,param_grid,cv=10,scoring = 'accuracy',verbose = 1)
grid.fit(X,y)
print(grid.best_score_)
print(grid.best_params_)
# SVC classification report
svc = SVC(kernel='rbf',C= 50,gamma=0.01)
pred_response = cross_val_predict(svc,X,y,cv=10)
print(confusion_matrix(y,pred_response))
# for some reason the 'Title_Master' still shows up under test set, which should have been dropped during dummy variable creation.
test = test.drop('Title_Master',axis = 1)
test.head()
# Let's predict the response for our test data set
svc = SVC(kernel='rbf',C= 50,gamma=0.01)
svc.fit(X,y)
predict_survival = svc.predict(test)
pred = pd.Series(predict_survival,name = 'Survived')
result = pd.concat([PassengerId,pred],axis = 1)
# submission.head()
result.to_csv('Titanic_survival_prediction.csv',index = False)