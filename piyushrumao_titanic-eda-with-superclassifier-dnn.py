# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set(style='white', context='notebook', palette='deep')

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

# example of a super learner using the mlens library

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from mlens.ensemble import SuperLearner



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

print(df_train.shape)

df_train.head()
profile  =  ProfileReport(df_train, title='Titanic Training Data')

profile
g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 0) & (df_train["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 1) & (df_train["Age"].notnull())], ax =g, color="Green", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
df_train[["Sex","Survived"]].groupby('Sex').mean()
g = sns.barplot(x="Sex",y="Survived",data=df_train)

g = g.set_ylabel("Survival Probability")
g = sns.barplot(x="Pclass",y="Survived",data=df_train)

g = g.set_ylabel("Survival Probability")
df_train[["Sex","Pclass","Survived"]].groupby(['Pclass','Sex']).mean()
# Explore Pclass vs Survived by Sex

g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=df_train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")

df_train[["Pclass","Fare","Survived"]].groupby(['Pclass','Survived']).mean()
df_train[["Sex","Pclass","Survived","Fare"]].groupby(['Sex','Survived','Pclass']).mean()

# Explore Embarked vs Survived 

g = sns.barplot(x="Embarked", y="Survived",  data=df_train)

g = g.set_ylabel("survival probability")
### male to female proportion getting in at different locations



df_try = df_train[df_train['Embarked']=='C']



print(df_try.Sex.value_counts())

print(df_try.Pclass.value_counts())

df_try_1 = df_train[df_train['Embarked']=='Q']



print(df_try_1.Sex.value_counts())

print(df_try_1.Pclass.value_counts())
df_try = df_train[df_train['Embarked']=='S']



print(df_try.Sex.value_counts())

print(df_try.Pclass.value_counts())
df_train["Family_size"] = df_train["SibSp"] + df_train["Parch"] + 1

df_train["Family_size"].unique()
# Explore FAmily size vs Survived 

g = sns.barplot(x="Family_size", y="Survived",  data=df_train)

g = g.set_ylabel("survival probability")
### Check Null values

print(df_train.shape)

df_train.isnull().sum()
## Since Age is a continious numerical feature, a groupby will not make much sense.



df_train['age_by_decade'] = pd.cut(x=df_train['Age'], bins=[0,10, 20, 30, 40, 50, 60, 80], labels=['babies','Teenagers','20s', '30s','40s','50s','Seniors'])

print(df_train.shape)

df_train.head()

### Check Null values

df_train.isnull().sum()
## So we will impute all the null values in Embarkment with the mode value

df_train['Embarked'].fillna(method ='ffill', inplace = True)
age_encoder = preprocessing.LabelEncoder()

sex_encoder = preprocessing.LabelEncoder()

embark_encoder = preprocessing.LabelEncoder()



df_train['age_by_decade'] = df_train['age_by_decade'].astype(str)

df_train['Embarked'] = df_train['Embarked'].astype(str)

df_train['Age'] = age_encoder.fit_transform(df_train['age_by_decade'])

df_train['Sex'] = sex_encoder.fit_transform(df_train['Sex'])

df_train['Embarked'] = embark_encoder.fit_transform(df_train['Embarked'])

df_train.head()
#dropping the 'Age','Name','Ticket','Cabin' features 

df_train = df_train.drop(['PassengerId','age_by_decade','Name','Ticket','Cabin','Family_size'], axis = 1)



df_train.head()
X = df_train.drop(['Survived'], axis=1)

y = df_train["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.22)

print(X_train.shape,X_val.shape )

print(X.shape,y.shape )

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(max_iter = 1000)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100)

print(acc_logreg)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100)

print(acc_svc)


# create a list of base-models

def get_models():

    models = list()

    models.append(LogisticRegression(solver='liblinear'))

    models.append(DecisionTreeClassifier())

    models.append(SVC(gamma='scale', probability=True))

    models.append(GaussianNB())

    models.append(KNeighborsClassifier())

    models.append(AdaBoostClassifier())

    models.append(BaggingClassifier(n_estimators=10))

    models.append(RandomForestClassifier(n_estimators=10))

    models.append(ExtraTreesClassifier(n_estimators=10))

    return models



# create the super learner

def get_super_learner(X):

    ensemble = SuperLearner(scorer=accuracy_score, folds=10, sample_size=len(X))

    # add base models

    models = get_models()

    ensemble.add(models)

    # add the meta model

    ensemble.add_meta(LogisticRegression(solver='lbfgs'))

    return ensemble





print('Train', X_train.shape, y_train.shape, 'Test', X_val.shape, y_val.shape)

# create the super learner

ensemble = get_super_learner(X_train)

# fit the super learner

ensemble.fit(X_train, y_train)

# summarize base learners

print(ensemble.data)

# make predictions on hold out set

yhat = ensemble.predict(X_val)

print('Super Learner: %.3f' % (accuracy_score(y_val, yhat) * 100))
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(df_test.shape)

df_test.head()
## Since Age is a continious numerical feature, a groupby will not make much sense.



df_test['age_by_decade'] = pd.cut(x=df_test['Age'], bins=[0,10, 20, 30, 40, 50, 60, 80], labels=['babies','Teenagers','20s', '30s','40s','50s','Seniors'])

print(df_test.shape)

df_test.head()
df_test['age_by_decade'] = df_test['age_by_decade'].astype(str)

df_test['Embarked'] = df_test['Embarked'].astype(str)

df_test['Age'] = age_encoder.transform(df_test['age_by_decade'])

df_test['Sex'] = sex_encoder.transform(df_test['Sex'])

df_test['Embarked'] = embark_encoder.transform(df_test['Embarked'])

df_test.head()
#dropping the 'Age','Name','Ticket','Cabin' features

df_test = df_test.drop(['age_by_decade','Name','Ticket','Cabin'], axis = 1)



df_test.head()
df_test.isnull().sum()
## As there is a null value in fare  So we will impute all the null values in Fare with the mean value

df_test['Fare'].fillna(method ='ffill', inplace = True)
X_pred = df_test.drop(['PassengerId'], axis=1)

X_pred.head()
y_pred_true = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(y_pred_true.shape)

y_pred_true.head()


# make predictions on hold out set

y_pred = ensemble.predict(X_pred).astype(int)

y_pred
print('Super Learner: %.3f' % (accuracy_score(y_pred_true['Survived'], y_pred) * 100))
#set ids as PassengerId and predict survival 

ids = y_pred_true['PassengerId']

#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': y_pred })

output.to_csv('submission.csv', index=False)
# Checking my input data

X.head()
# Checking input features

y.head()
from keras.models import Sequential

from keras.layers import Dense



# define the keras model

model = Sequential()

model.add(Dense(12, input_dim=7, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# compile the keras model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset

model.fit(X, y, epochs=100, batch_size=10)
# evaluate the keras model

_, accuracy = model.evaluate(X, y)

print('Accuracy: %.2f' % (accuracy*100))
# make class predictions with the model

predictions_keras = model.predict_classes(X_pred)

predictions_keras
from pandas.core.common import flatten

keras_prediction = list(flatten(predictions_keras))

keras_prediction
print('Keras Deep Learner: %.3f' % (accuracy_score(y_pred_true['Survived'], keras_prediction) * 100))
#set ids as PassengerId and predict survival 

ids = y_pred_true['PassengerId']

#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': keras_prediction })

output.to_csv('submission.csv', index=False)