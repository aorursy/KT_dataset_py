import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/train.csv')
train_data.head()
train_data.columns
train_data.describe()
train_data.info()
train_data.count()
train_data.isnull().sum().sort_values(ascending=False)
train_data[['Survived','Embarked']].groupby(['Embarked'],as_index=False).count() #to check which port has highest intake
train_data['Embarked'] = train_data['Embarked'].fillna('S'); # fillna returns value 
train_data[['Survived','Embarked']].groupby(['Embarked'],as_index=False).count() #after filling the blank 2 with S
train_data['Family'] = train_data['SibSp']+train_data['Parch'] +1
train_data['Isalone'] = (train_data['Family'] == 1)*1
f,ax = plt.subplots(2,4,figsize=(20,16))
sns.countplot('Sex',data =train_data,ax = ax[0][0])
sns.countplot('Pclass',data = train_data,ax = ax[0][1])
sns.countplot('Family',data=train_data,ax = ax[0][2])
sns.countplot('Embarked',data=train_data,ax = ax[0][3] )
sns.countplot("Sex",hue = 'Survived',data = train_data, ax = ax[1][0])
sns.countplot("Pclass",hue = 'Survived',data = train_data, ax = ax[1][1])
sns.countplot("Family",hue = 'Survived',data = train_data, ax = ax[1][2])
sns.countplot("Embarked",hue = 'Survived',data = train_data, ax = ax[1][3])
plt.show()
train_data[['Family','Survived']].groupby(['Family'],as_index = False).mean().sort_values(by = 'Survived',ascending = False)
train_data['Title'] = train_data['Name'].str.extract('([A-Za-z]+)\.',expand = False) 
train_data['Title'].unique()
train_data['Title'].value_counts() # value_cout works only in series not in Dataframe
print(pd.crosstab(train_data['Title'], train_data['Sex'])) #crosstab breaks 2nd argument into its unique values and give results
train_data['Title'] = train_data['Title'].replace(['Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess',
       'Jonkheer'],'Few')
fig,(axis1,axis2) = plt.subplots(1,2,figsize=(20,8))
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train_data,ax = axis1)
sns.distplot(train_data[train_data['Survived']==0]['Age'].dropna(),ax=axis2,kde=False,color='r',bins=15) 
sns.distplot(train_data[train_data['Survived']==1]['Age'].dropna(),ax=axis2,kde=False,color='g',bins=15)
plt.show()
import random
age_mean = train_data['Age'].mean()
age_sd = train_data['Age'].std()
age_random_list = np.random.randint(age_mean-age_sd, age_mean+age_sd, train_data['Age'].isnull().sum() )
train_data['Age'][np.isnan(train_data['Age'])] = age_random_list
train_data['Age'] = train_data['Age'].astype(int)
train_data.loc[train_data['Age'] <16 ,'Age'] = 0
train_data.loc[(train_data['Age']>=16) & (train_data['Age']<32),'Age'] = 1
train_data.loc[(train_data['Age']>=32) & (train_data['Age']<45),'Age'] = 2
train_data.loc[(train_data['Age']>=45) & (train_data['Age']<55),'Age'] = 3
train_data.loc[(train_data['Age']>55),'Age'] = 4
fig, ax = plt.subplots(figsize=(20,6))
sns.distplot(train_data[train_data['Survived']==0]['Fare'],kde=False,color='r', ax = ax )
sns.distplot(train_data[train_data['Survived']==1]['Fare'],kde=False,color='b',ax = ax)
plt.show()
train_data['CategoricalFare'] = pd.qcut(train_data['Fare'], 3)
train_data['CategoricalFare'].value_counts()
train_data.loc[(train_data['Fare'])<= 8.662, 'Fare'] = 0
train_data.loc[(train_data['Fare']>8.662) & (train_data['Fare']<= 26), 'Fare'] = 1
train_data.loc[(train_data['Fare']>26), 'Fare']  =2
train_data['Fare'] = train_data['Fare'].astype(int)
gender = {'male':0, 'female' : 1}
title_dict = {'Mr' : 0, 'Mrs':1,'Miss':2,'Master':3, 'Few':4 }
emb = {'S':0,'C':1,'Q':2}
train_data['Title']= train_data['Title'].map(title_dict).astype(int)
train_data['Embarked'] = train_data['Embarked'].map(emb).astype(int)
train_data['Sex'] = train_data['Sex'].map(gender).astype(int)
train_data = train_data.drop(['PassengerId','Name','Cabin','CategoricalFare','Ticket'],axis = 1)
train_data.sample(10)
X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]
X_train.shape, Y_train.shape

#X_test = test.copy()
import sklearn         # Collection of machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,Y_train)
acc_log = round(logistic_regression.score(X_train, Y_train) * 100, 2)
acc_log
svc=SVC()
svc.fit(X_train, Y_train)
#Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
#Y_pred = gaussian.predict(test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
#Y_pred = perceptron.predict(test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
#Y_pred = linear_svc.predict(test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
#Y_pred = sgd.predict(test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
#Y_pred = decision_tree.predict(test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
#random_forest_predictions = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 
                           weights='uniform')
knn.fit(X_train, Y_train)
#knn_predictions = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
