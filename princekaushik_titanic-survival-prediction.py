import pandas as pd

import keras

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import eli5

from eli5.sklearn import PermutationImportance

import xgboost as xgb

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC,NuSVC

from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier,VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier

from catboost import CatBoostClassifier

from sklearn.decomposition import PCA

from sklearn.decomposition import FastICA

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers import Dense, Dropout

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV

from sklearn import metrics as met

from sklearn.metrics import classification_report

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

pID = test['PassengerId']

print(train.shape)

print(test.shape)
print(train.columns)

print(test.columns)
train.head()
train.info()
train.describe().T
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue = 'Sex',data = train)
sns.countplot(x='Survived',hue = 'Pclass',data = train)
sns.countplot(x='Survived',hue = 'Embarked',data = train)
sns.distplot(train.Age.dropna())
sns.countplot(x='SibSp',data=train)
train.Fare.hist()
print(train.isna().sum().sort_values(ascending=False)[:4])

print(test.isna().sum().sort_values(ascending=False)[:4])
for data in [train,test]:

    data['Age'].fillna(data['Age'].median(),inplace=True)

    

    data['Embarked'].fillna(method='ffill',inplace=True)

    

    data['Fare'].fillna(method='ffill',inplace=True)

    

drop = ['PassengerId','Cabin','Ticket','Name']

train.drop(drop,axis=1,inplace=True)

test.drop(drop,axis=1,inplace=True)
# feature Engineerig



for data in [train,test]:

    data['FamilySize'] = data['SibSp']+ data['Parch'] + 1

    

    data['IsAlone'] = 1

    data['IsAlone'].loc[data['FamilySize'] >1 ] = 0
train_y = train.Survived

train_x = train.drop('Survived',axis=1)
train_x.columns
dummy  = ['Pclass','Sex','Embarked']

train_x[dummy]=train_x[dummy].astype('category')

train_X  = pd.get_dummies(train_x[dummy],drop_first=True,prefix =dummy)

train_X = pd.concat([train_x,train_X],axis=1)

train_X.drop(dummy,axis=1,inplace=True)

train_X.shape
test[dummy]=test[dummy].astype('category')

test_X  = pd.get_dummies(test[dummy],drop_first=True,prefix =dummy)

test_X = pd.concat([test,test_X],axis=1)

test_X.drop(dummy,axis=1,inplace=True)

test_X.shape
ss= MinMaxScaler()

train_X = ss.fit_transform(train_X)

test = ss.fit_transform(test_X)
classifiers = {'Gradient Boosting Classifier': GradientBoostingClassifier(),'Ada Boost Classifier':AdaBoostClassifier(),'RadiusNN':RadiusNeighborsClassifier(radius=40.0),'Linear Discriminant Analyis': LinearDiscriminantAnalysis(),'GaussianNB':GaussianNB(),'BerNB':BernoulliNB(),'KNN':KNeighborsClassifier(),'Random Forest Classifier': RandomForestClassifier(min_samples_leaf=10,min_samples_split=20,max_depth=4),'Decision Tree Classifier' : DecisionTreeClassifier(),'Logistic Regression': LogisticRegression(),'XGBoost': xgb.XGBClassifier()}
#Splitting Data

train_X,test_X,train_y,test_y = train_test_split(train_X,train_y,test_size=0.2,random_state = 1)
base_accuracy = 0 

for Name,classify in classifiers.items():

    classify.fit(train_X,train_y)

    predicting_y = classify.predict(test_X)

    print('Accuracy Score of ' + str(Name) + ":"+ str(met.accuracy_score(test_y,predicting_y)))

    

    if met.accuracy_score(test_y,predicting_y) > base_accuracy:

        prediction_test = classify.predict(test)

        base_accuracy = met.accuracy_score(test_y,predicting_y)

    else:

        continue

# Generate Submission File

predicted_test_values = pd.DataFrame({'PassengerId': pID,'Survived' :prediction_test })

predicted_test_values.to_csv('PredictedTestScore.csv',index = False)