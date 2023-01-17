# Disabling warnings

import warnings

warnings.simplefilter("ignore")
# Import Main libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Import Visualization lib.

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
import os

print(os.listdir('../input'))
# set our Dataframe

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/gender_submission.csv')
# Show first 5 rows of train data

train_df.head()
# data size

print("Train Data Size: ", train_df.shape)

print("Test Data Size:  ", test_df.shape)
# Show if any NAN data

train_df.isnull().sum()
test_df.isnull().sum()
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(np.nan, "mean")



train_df['Age'] = imputer.fit_transform(np.array(train_df['Age']).reshape(891, 1)) # 1st

train_df.Embarked.fillna(method='ffill', inplace=True) # 2nd

train_df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True) # 3rd



test_df['Age'] = imputer.fit_transform(np.array(test_df['Age']).reshape(418, 1))

test_df.Embarked.fillna(method='ffill', inplace=True)

test_df.Fare.fillna(method='ffill', inplace=True)

test_df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
trn_mid = train_df.Age.median() # set median value



# fill NAN data

train_df.Age.fillna(trn_mid, inplace=True)

train_df.Embarked.fillna(method='ffill', inplace=True)

train_df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)



tst_mid = test_df.Age.median() # set median value



# fill NAN data

test_df.Age.fillna(tst_mid, inplace=True)

test_df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
sns.countplot(x='Survived', hue='Sex', data=train_df)
sns.countplot(x='Embarked', hue='Survived', data=train_df)
sns.countplot(x='SibSp', hue='Survived', data=train_df)
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.figure(figsize=(10,5))

sns.distplot(train_df['Age'], bins=24, color='b')
train_df.info()
objects_cols = train_df.select_dtypes("object").columns

objects_cols
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train_df[objects_cols] = train_df[objects_cols].apply(le.fit_transform)

test_df[objects_cols] = test_df[objects_cols].apply(le.fit_transform)

train_df[objects_cols].head()
train_df.head()
plt.figure(figsize=(12, 8))

plt.title('Titanic Correlation of Features', y=1.05, size=15)

sns.heatmap(train_df.corr(), linewidths=0.1, vmax=1.0, 

            square=True, linecolor='white', annot=True)
from sklearn.model_selection import train_test_split, cross_val_score



from sklearn.preprocessing import StandardScaler



from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.metrics import accuracy_score
# Machine Learning 

X = train_df.drop(['Survived'], 1).values

y = train_df['Survived'].values
scale = StandardScaler()

scale.fit(X)



X = scale.transform(X)
# Split data to 80% training data and 20% of test to check the accuracy of our model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
class Model:

    def __init__(self, model):

        self.model = model

        self.X, self.y = X, y

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        

        self.train()

    

    def model_name(self):

        model_name = type(self.model).__name__

        return model_name

        

    def cross_validation(self, cv=5):

        print(f"Evaluate {self.model_name()} score by cross-validation...")

        CVS = cross_val_score(self.model, self.X, self.y, scoring='accuracy', cv=cv)

        print(CVS)

        print("="*60, "\nMean accuracy of cross-validation: ", CVS.mean())

    

    def train(self):

        print(f"Training {self.model_name()} Model...")

        self.model.fit(X_train, y_train)

        print("Model Trained.")

        

    def prediction(self, test_x=None, test=False):

        if test == False:

            y_pred = self.model.predict(self.X_test)

        else:

            y_pred = self.model.predict(test_x)

            

        return y_pred

    

    def accuracy(self):

        y_pred = self.prediction()

        y_test = self.y_test

        

        acc = accuracy_score(y_pred, y_test)

        print(f"{self.model_name()} Model Accuracy: ", acc)
xgb = XGBClassifier()

xgb = Model(xgb)
xgb.cross_validation()
xgb.accuracy()
gnb = GaussianNB()

gnb = Model(gnb)
gnb.cross_validation()
gnb.accuracy()
svc = SVC()

svc = Model(svc)
svc.cross_validation()
svc.accuracy()
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc = Model(rfc)

rfc.cross_validation()

rfc.accuracy()
test_df.head()
# Predict our file test

test_X = test_df.values

test_X = scale.transform(test_X)
xgb_pred = xgb.prediction(test_x=test_X, test=True)

gnb_pred = gnb.prediction(test_x=test_X, test=True)

svc_pred = svc.prediction(test_x=test_X, test=True)

rfc_pred = rfc.prediction(test_x=test_X, test=True)
sub.head()

sub.to_csv('submission.csv', index=False)

sub.head()
sub['Survived'] = xgb_pred # Best Submission (Top 5% LB)

sub.to_csv('xgb_submission.csv', index=False)

sub.head(10)
sub['Survived'] = gnb_pred

sub.to_csv('gnb_submission.csv', index=False)

sub.head(10)
sub['Survived'] = svc_pred

sub.to_csv('svc_submission.csv', index=False)

sub.head(10)
sub['Survived'] = rfc_pred

sub.to_csv('rfc_submission.csv', index=False)

sub.head(10)