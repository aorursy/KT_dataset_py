# Common imports

import pandas as pd

import numpy as np

import os



# To plot Pretty figures

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import cufflinks as cf

cf.go_offline()



#Modelling and others

from sklearn.linear_model import LogisticRegression

from xgboost.sklearn import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import  KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn import cluster, datasets, mixture

from sklearn.metrics import accuracy_score,recall_score, f1_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.svm import SVC,LinearSVC

from sklearn.naive_bayes import GaussianNB

from collections import Counter

from sklearn.model_selection import train_test_split



%matplotlib inline

mpl.rc('axes', labelsize= 15)

mpl.rc('xtick', labelsize= 12)

mpl.rc('ytick', labelsize= 12)



mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

# avoid warnings

import warnings

warnings.filterwarnings(action="ignore")
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train=train.set_index('PassengerId')

test=test.set_index('PassengerId')
train .head()
# See the name of the columns

train.columns
train.info()
print("Numeric columns: \n", train.select_dtypes(include='number').columns)



print("Categorical columns: \n", train.select_dtypes(include='object').columns)
train.describe()
train.hist(bins=30, figsize=(16,12))

#save_fig("attribute_histogram_plots")

plt.show()
#check the percentage of the passenger survived and not survived on the training dataset

Passenger_notsurvived =  (train['Survived'].value_counts()[0] / len(train['Survived']) ) * 100 

Passenge_Survived = 100 - Passenger_notsurvived

print("Passenge Survived : {:.2f}% , Passenge not Survived : {:.2f}%".format(Passenge_Survived,Passenger_notsurvived))
# use seaborn

sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='RdBu_r')
g = sns.factorplot(x="Survived", y = "Age",data = train, kind="box")
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
#more checks

sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
def comparevariables(train,var1, var2):

    print(train[[var1, var2]][train[var2].isnull()==False].groupby([var1], as_index=False).mean().sort_values(by=var2, ascending=False))
comparevariables(train,'Age','Survived')
comparevariables(train,'Pclass','Survived')
comparevariables(train,'SibSp','Survived')
#Let's plot a few more

sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
train['Fare'].iplot(kind='hist',bins=30,color='green')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
# Combine train and test features

df = pd.concat([train, test], axis=0, sort=False)

df.shape
df.head()
#first check the missing data

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent*100], axis=1, keys=['Total', 'Percent'])

missing_data.head()
# Visualize missing values

missing_data=missing_data.head(5).drop('Survived')

f, ax = plt.subplots(figsize=(10, 8))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.title('Percent missing data by feature', fontsize=15)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)
df['Age'].describe()
df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
df['Title'].value_counts()
df['Age'] = df.groupby([ 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))
median_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

df['Fare'] = df['Fare'].fillna(median_fare)
em_mode = df[df['Pclass']==1]['Embarked'].mode()[0]

df['Embarked']=df['Embarked'].fillna(em_mode)
# Thanks to https://www.kaggle.com/mauricef/titanic for these amazing features

df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))

df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived

df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())

df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - df.Survived.fillna(0), axis=0)

df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)

df['Alone'] = (df.WomanOrBoyCount == 0)
df.columns
df=df[['Sex', 'WomanOrBoySurvived', 'Alone','Age','Fare']]
df.isna().sum()
#Check the info one more time

df.info()

#fill the non null value with 0 and replcae sex columns

df['WomanOrBoySurvived']=df['WomanOrBoySurvived'].fillna(0)

df['Sex']=df['Sex'].replace({'male': 0, 'female': 1})
#Check for the last time 

df.isna().sum()
df.head()
# Split features and labels

y = train['Survived'].reset_index(drop=True)

train = df[:len(train)]

test = df[len(train):]

train.shape,test.shape
Scores = pd.DataFrame({'Model': [],'Accuracy Score': [], 'Recall':[], 'F1score':[]})
#Split the train and test data

X_train, X_test, y_train, y_test = train_test_split(train, y,test_size=0.20, random_state=42)
xgboost = XGBClassifier(learning_rate=0.01, n_estimators=4060,gamma=0.0482,

                                     max_depth=4, min_child_weight=0,

                                     subsample=0.7,colsample_bytree=0.7,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006,random_state=42)



xgboost.fit(X_train,y_train)

y_pred = xgboost.predict(X_test)



score = pd.DataFrame({"Model":['XGBClassifier'],

                    "Accuracy Score": [accuracy_score(y_test, y_pred)],

                   "Recall": [recall_score(y_test, y_pred)],

                   "F1score": [f1_score(y_test, y_pred)]})

Scores = Scores.append(score)
RFmodel = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 3000]}, cv=10).fit(X_train,y_train)

RFmodel.fit(X_train,y_train)

y_pred = RFmodel.predict(X_test)



score = pd.DataFrame({"Model":['RFmodel'],

                    "Accuracy Score": [accuracy_score(y_test, y_pred)],

                   "Recall": [recall_score(y_test, y_pred)],

                   "F1score": [f1_score(y_test, y_pred)]})

Scores = Scores.append(score)
DTmodel = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid={'max_depth' : np.arange(2, 9, dtype=int),

              'min_samples_leaf' :  np.arange(1, 3, dtype=int)}, cv=10).fit(X_train,y_train)

DTmodel.fit(X_train,y_train)

y_pred = DTmodel.predict(X_test)



score = pd.DataFrame({"Model":['DTmodel'],

                    "Accuracy Score": [accuracy_score(y_test, y_pred)],

                   "Recall": [recall_score(y_test, y_pred)],

                   "F1score": [f1_score(y_test, y_pred)]})

Scores = Scores.append(score)
KNmodel = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': [2, 10]}, cv=10).fit(X_train,y_train)



KNmodel.fit(X_train,y_train)

y_pred = KNmodel.predict(X_test)



score = pd.DataFrame({"Model":['KNmodel'],

                    "Accuracy Score": [accuracy_score(y_test, y_pred)],

                   "Recall": [recall_score(y_test, y_pred)],

                   "F1score": [f1_score(y_test, y_pred)]})

Scores = Scores.append(score)
logmodel= LogisticRegression()

logmodel.fit(X_train,y_train)



y_pred = logmodel.predict(X_test)



score = pd.DataFrame({"Model":['logmodel'],

                    "Accuracy Score": [accuracy_score(y_test, y_pred)],

                   "Recall": [recall_score(y_test, y_pred)],

                   "F1score": [f1_score(y_test, y_pred)]})

Scores = Scores.append(score)
SVCmodel= SVC(probability=True)

SVCmodel.fit(X_train,y_train)



y_pred = SVCmodel.predict(X_test)



score = pd.DataFrame({"Model":['SVCmodel'],

                    "Accuracy Score": [accuracy_score(y_test, y_pred)],

                   "Recall": [recall_score(y_test, y_pred)],

                   "F1score": [f1_score(y_test, y_pred)]})

Scores = Scores.append(score)
vot_classifier = VotingClassifier(estimators=[('xg', xgboost),('log', logmodel), ('rf', RFmodel), ('dt', DTmodel), ('svc', SVCmodel)], voting='soft', n_jobs=4)



vot_classifier=vot_classifier.fit(X_train, y_train)



y_pred = vot_classifier.predict(X_test)



score = pd.DataFrame({"Model":['vot_classifier'],

                    "Accuracy Score": [accuracy_score(y_test, y_pred)],

                   "Recall": [recall_score(y_test, y_pred)],

                   "F1score": [f1_score(y_test, y_pred)]})

Scores = Scores.append(score)

vot_classifier=vot_classifier.fit(train, y)
Scores
df1=df[['Sex', 'WomanOrBoySurvived', 'Alone']]
# Split features and labels

y=y

train = df1[:len(train)]

test = df1[len(train):]

train.shape,test.shape
DTmodel.fit(train,y)
submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission.iloc[:,1] = DTmodel.predict(test).astype(int)
submission.head()
submission.to_csv("submissionTitanic.csv", index=False)