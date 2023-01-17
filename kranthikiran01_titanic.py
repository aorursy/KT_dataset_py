# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
data.head()
import missingno as msn
msn.matrix(data)
msn.heatmap(data)
msn.dendrogram(data)
data.info()
data.isnull().sum()
data.drop('Cabin',axis=1,inplace=True)
data.Embarked.fillna(data.Embarked.mode()[0],inplace=True)
data.Fare.fillna(data.Fare.median(),inplace=True)
def getSes(x):

    if x==1.0:

        return "Upper"

    if x==2.0:

        return "Middle"

    if x==3.0:

        return "Lower"

data['ses']=data.Pclass.apply(lambda x: getSes(x))
import string

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if type(big_string)==type("abc"):

            if str.find(big_string, substring) != -1:

                return substring

    return np.nan

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev','Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess','Don', 'Jonkheer']

data['Title']=data['Name'].map(lambda x: substrings_in_string(str(x), title_list))
def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

data['Title']=data.apply(replace_titles, axis=1)
data['Family_Size']=data['SibSp']+data['Parch']+1
data['isAlone']=1

data['isAlone'].loc[data['Family_Size']>1]=0
data['Age']=data.Age.fillna(data.Age.median())
from sklearn import preprocessing

data[['Age']]=preprocessing.MinMaxScaler().fit_transform(data[['Age']])
data['Fare_per_person']=data['Fare']/(data['Family_Size'])
# cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

# data['Deck']=data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
one_hot_encodings = pd.get_dummies(data[['ses','Sex','Age','Embarked','Survived','SibSp','Parch','Title','Family_Size','Fare_per_person','isAlone']])
one_hot_encodings.head()
msn.matrix(one_hot_encodings)
# from sklearn.preprocessing import Imputer

# age_imputer = Imputer()

# one_hot_encodings = age_imputer.fit_transform(one_hot_encodings)

# data[data.Age.notnull()]
# one_hot_encodings = pd.DataFrame(one_hot_encodings,columns=['Age','Survived','SibSp','Parch','Family_Size','ses_Lower','ses_Middle','ses_Upper','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S','Title_Master','Title_Miss','Title_Mr','Title_Mrs'])
y_column = ['Survived']

x_columns = ['isAlone','Family_Size','Fare_per_person','ses_Lower','ses_Middle','ses_Upper','Sex_female','Sex_male','Age','Embarked_C','Embarked_Q','Embarked_S','Title_Master','Title_Miss','Title_Mr','Title_Mrs']

#'Deck_A','Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G','Deck_T'

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(one_hot_encodings[x_columns], one_hot_encodings[y_column], test_size=0.33 , random_state=42,stratify=one_hot_encodings[y_column])
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score

from sklearn import metrics

# instantiate learning model (k = 3)

knn = KNeighborsClassifier(n_neighbors=3)



# fitting the model

knn.fit(X_train, y_train)



# predict the response

pred = knn.predict(X_test)



# evaluate accuracy

print(accuracy_score(y_test, pred))

print(r2_score(y_test,pred))

print(metrics.classification_report(y_test, pred))
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

model = LogisticRegression(solver='liblinear')

model.fit(X_train, y_train)

log_predicted = model.predict(X_test)

print(accuracy_score(y_test,log_predicted))

print(r2_score(y_test,log_predicted))

# summarize the fit of the model

print(metrics.classification_report(y_test, log_predicted))

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train, y_train)

nb_predicted = model.predict(X_test)

print(r2_score(y_test,nb_predicted))

print(metrics.classification_report(y_test, nb_predicted))
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=452)

model.fit(X_train, y_train)

gb_predicted = model.predict(X_test)

print(r2_score(y_test,gb_predicted))

print(metrics.classification_report(y_test, gb_predicted))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150,max_depth=6, oob_score=True, random_state=123456)

rf.fit(X_train, y_train)

rf_predicted = rf.predict(X_test)

accuracy = accuracy_score(y_test, rf_predicted)

print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')

print(f'Mean accuracy score: {accuracy:.3}')

print(r2_score(y_test,rf_predicted))

print(metrics.classification_report(y_test, rf_predicted))
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

dt_predicted = model.predict(X_test)

print(r2_score(y_test,dt_predicted))

print(metrics.classification_report(y_test, dt_predicted))
from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

svc_predicted = model.predict(X_test)

print(r2_score(y_test,svc_predicted))

print(metrics.classification_report(y_test, svc_predicted))

from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train,y_train)

xgb_predicted = model.predict(X_test)

print(r2_score(y_test,xgb_predicted))

print(metrics.classification_report(y_test,svc_predicted))
from sklearn.metrics import confusion_matrix

import seaborn as sns

cm = pd.DataFrame(confusion_matrix(y_test, nb_predicted))

sns.heatmap(cm, annot=True)

cm1 = pd.DataFrame(confusion_matrix(y_test, log_predicted))

sns.heatmap(cm1, annot=True)
cm2 = pd.DataFrame(confusion_matrix(y_test, dt_predicted))

sns.heatmap(cm2, annot=True)
cm3 = pd.DataFrame(confusion_matrix(y_test, svc_predicted))

sns.heatmap(cm3, annot=True)
cm4 = pd.DataFrame(confusion_matrix(y_test, gb_predicted))

sns.heatmap(cm4, annot=True)
cm5 = pd.DataFrame(confusion_matrix(y_test, xgb_predicted))

sns.heatmap(cm5, annot=True)
cm6 = pd.DataFrame(confusion_matrix(y_test, rf_predicted))

sns.heatmap(cm6, annot=True)
# y_column = ['Survived']

# x_columns = ['ses_Lower','ses_Middle','ses_Upper','Sex_female','Sex_male','Age','Embarked_C','Embarked_Q','Embarked_S']

# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=5)

# knn.fit(one_hot_encodings[x_columns], one_hot_encodings[y_column])

# test = pd.read_csv('../input/test.csv')

# test['ses']=test.Pclass.apply(lambda x: getSes(x))

# one_hot_test_encoding = pd.get_dummies(test[['ses','Sex','Age','Embarked']])



# one_hot_test_encoding = age_imputer.fit_transform(one_hot_test_encoding)

# one_hot_test_encoding = pd.DataFrame(one_hot_test_encoding,columns=x_columns)



# predictions = knn.predict(one_hot_test_encoding[x_columns])

# predictions
test = pd.read_csv('../input/test.csv')

test['ses']=test.Pclass.apply(lambda x: getSes(x))

test['Title']=test['Name'].map(lambda x: substrings_in_string(str(x), title_list))

test['Title']=test.apply(replace_titles, axis=1)

test['Family_Size']=test['SibSp']+test['Parch']

test['Fare_per_person']=test['Fare']/(test['Family_Size']+1)

test['isAlone']=1

test['isAlone'].loc[test['Family_Size']>1]=0

test.Age.fillna(test.Age.median(),inplace=True)

test[['Age']]=preprocessing.MinMaxScaler().fit_transform(test[['Age']])

#test['Deck']=test['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

one_hot_test_encodings = pd.get_dummies(test[['ses','Sex','Age','Embarked','Title','Family_Size','Fare_per_person','isAlone']])

# one_hot_test_encodings.head()

from sklearn.preprocessing import Imputer

imputer = Imputer()

one_hot_test_encodings = imputer.fit_transform(one_hot_test_encodings)

one_hot_test_encodings = pd.DataFrame(one_hot_test_encodings,columns=['Age','Family_Size','Fare_per_person','isAlone','ses_Lower','ses_Middle','ses_Upper','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S','Title_Master','Title_Miss','Title_Mr','Title_Mrs'])

#one_hot_test_encodings['Deck_T']=0
# rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)

# rf.fit(one_hot_encodings[x_columns], one_hot_encodings[y_column])

# # predict the response

# pred = knn.predict(one_hot_test_encodings[x_columns])
# knn = KNeighborsClassifier(n_neighbors=4)



# # fitting the model

# knn.fit(one_hot_encodings[x_columns], one_hot_encodings[y_column])



# # predict the response

# pred = knn.predict(one_hot_test_encodings[x_columns])
model = SVC()

model.fit(one_hot_encodings[x_columns], one_hot_encodings[y_column])

log_predicted = model.predict(one_hot_test_encodings[x_columns])
test['Survived']=log_predicted
result=test[['PassengerId','Survived']]
result.Survived.unique()
result.to_csv('submission.csv',index=False)