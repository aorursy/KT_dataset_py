import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

test.info()


train.head(5)
train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

Title_Dictionary = {

                        "Capt":       0,

                        "Col":        0,

                        "Major":      0,

                        "Jonkheer":   1,

                        "Don":        1,

                        "Sir" :       1,

                        "Dr":         0,

                        "Rev":        0,

                        "the Countess":1,

                        "Dona":       1,

                        "Mme":        2,

                        "Mlle":       3,

                        "Ms":         2,

                        "Mr" :        4,

                        "Mrs" :       2,

                        "Miss" :      3,

                        "Master" :    5,

                        "Lady" :      1



                        }

    

train['Title'] = train.Title.map(Title_Dictionary)

f,ax1 = plt.subplots()

corr = train.corr()

sns.heatmap(corr, vmax=1, square=True,ax=ax1)

plt.show()
fdf = train.drop(['PassengerId','Name','Ticket','Cabin','SibSp'],1)

fdf.head()
survived = fdf['Survived']

features = fdf.drop(['Survived'],axis=1)

features['Sex'] = features['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

features['Embarked'] = features['Embarked'].map( {'S': 0, 'C': 1, 'Q':2},na_action=None )
grouped = features.groupby(['Sex','Pclass','Title'])

gm = grouped.median()

print(gm)
gm['Age'][0]



features.Age = features.apply(lambda item : gm['Age'][item['Sex'],item['Pclass'],item['Title']] if np.isnan(item['Age']) else item['Age'], axis=1)

features.Age = features['Age'].apply(lambda x: np.log(x + 1))
sns.distplot(features['Age'])

plt.show()
features.Fare = features.apply(lambda item : gm['Fare'][item['Sex'],item['Pclass'],item['Title']] if np.isnan(item['Fare']) else item['Fare'], axis=1)

features.Fare = features['Fare'].apply(lambda x: np.log(x + 1))



sns.distplot(features['Fare'])

plt.show()
from math import ceil

features.Embarked = features.apply(lambda item : ceil(gm['Embarked'][item['Sex'],item['Pclass'],item['Title']]) if np.isnan(item['Embarked']) else item['Embarked'], axis=1)

#features.Embarked = features['Embarked'].apply(lambda x: np.log(x + 1))



sns.distplot(features['Embarked'])

plt.show()
pclass_dummies = pd.get_dummies(features['Pclass'],prefix="Pclass")

embarked_dummies =  pd.get_dummies(features['Embarked'],prefix="Embarked")

 # adding dummy variables

features.drop('Pclass',axis=1,inplace=True)

features.drop('Embarked',axis=1,inplace=True)

features = pd.concat([features,pclass_dummies],axis=1)

features = pd.concat([features,embarked_dummies],axis=1)
from sklearn import tree

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.svm import SVC

import random 

from sklearn.ensemble import RandomForestClassifier

from sklearn import neighbors

from time import time

from sklearn.metrics import accuracy_score

from sklearn.metrics import fbeta_score



X_train, X_test, y_train, y_test = train_test_split(features, survived, test_size = 0.2, random_state = 0)

print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))



def train_predict(learner, X_train, y_train, X_test, y_test): 

    

    results = {}

    start = time() # Get start time

    learner.fit(X_train,y_train)

    end = time() # Get end time

    results['train_time'] = end-start

        

    start = time() # Get start time

    predictions_test = learner.predict(X_test)

    predictions_train = learner.predict(X_train)

    end = time() # Get end time

    

    results['pred_time'] = end-start

    results['acc_train'] = accuracy_score(y_train,predictions_train)

    results['acc_test'] = accuracy_score(y_test,predictions_test)

    results['f_train'] = fbeta_score(y_train,predictions_train,beta=0.5)

    results['f_test'] = fbeta_score(y_test,predictions_test,beta=0.5)

    

    return results


clf_A = GaussianNB()

clf_B = tree.DecisionTreeClassifier()

clf_C = RandomForestClassifier(n_estimators=200,criterion='gini')



results = {}

for clf in [clf_A, clf_B, clf_C]:

    clf_name = clf.__class__.__name__

    results[clf_name] = {}

    results[clf_name] =  train_predict(clf, X_train, y_train, X_test, y_test)



train_time = {}

pred_time = {}

acc_train = {}

acc_test = {}

f_train = {}

f_test ={}



for k in results.keys():

    train_time[k] = results[k]["train_time"]

    pred_time[k] = results[k]["pred_time"]

    acc_train[k]  = results[k]["acc_train"]

    acc_test[k]   = results[k]["acc_test"]

    f_train[k]    = results[k]["f_train"]

    f_test[k]     = results[k]["f_test"]

    

    



f,axarray = plt.subplots(2,2)

axarray[0,0].set_title("Training time")

axarray[0,0].bar(range(len(train_time)), train_time.values(), align='center')

axarray[0,0].set_xticks(range(len(train_time)), train_time.keys())



axarray[0,1].set_title("Prediction time")

axarray[0,1].bar(range(len(pred_time)), pred_time.values(), align='center')

axarray[0,1].set_xticks(range(len(pred_time)), pred_time.keys())



axarray[1,0].set_title("Accuracy Test")

axarray[1,0].bar(range(len(acc_test)), acc_test.values(), align='center')

axarray[1,0].set_xticks(range(len(acc_test)), acc_test.keys())



axarray[1,1].set_title("F-Score Test")

axarray[1,1].bar(range(len(f_test)), f_test.values(), align='center')

axarray[1,1].set_xticks(range(len(f_test)), f_test.keys())

plt.show()

#clf_C.fit(features,survived)





imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

test['Title'] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

test['Title'] = test.Title.map(Title_Dictionary)



test_input = test.drop(['PassengerId','Name','Ticket','Cabin','SibSp'],1)



test_input['Sex'] = test_input['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test_input['Embarked'] = test_input['Embarked'].map( {'S': 0, 'C': 1, 'Q':2},na_action=None )

test_input['Embarked'] =  imp.fit_transform(test_input['Embarked'].values.reshape(-1,1))



test_input.Embarked = test_input.apply(lambda item : ceil(gm['Embarked'][item['Sex'],item['Pclass'],item['Title']]).astype(int) if np.isnan(item['Embarked'].astype(int)) else item['Embarked'], axis=1)



test_input.Age = test_input.apply(lambda item : gm['Age'][item['Sex'],item['Pclass'],item['Title']] if np.isnan(item['Age']) else item['Age'], axis=1)

test_input.Age = test_input['Age'].apply(lambda x: np.log(x + 1))



test_input.Fare = test_input.apply(lambda item : gm['Fare'][item['Sex'],item['Pclass'],item['Title']] if np.isnan(item['Fare']) else item['Fare'], axis=1)

test_input.Fare = test_input['Fare'].apply(lambda x: np.log(x + 1))



pclass_dummies = pd.get_dummies(test_input['Pclass'],prefix="Pclass")

embarked_dummies =  pd.get_dummies(test_input['Embarked'],prefix="Embarked")

 # adding dummy variables

test_input.drop('Pclass',axis=1,inplace=True)

test_input.drop('Embarked',axis=1,inplace=True)

test_input = pd.concat([test_input,pclass_dummies],axis=1)

test_input = pd.concat([test_input,embarked_dummies],axis=1)



pd.isnull(test_input).any()



#test_input['Fare'] = test_input['Fare'].apply(lambda x: np.log(x + 1))

#test_input['Fare'] =  imp.fit_transform(test_input['Fare'].values.reshape(-1,1))







#test_input.head()
print ("using {}".format(clf_C.__class__.__name__))

prediction = clf_C.predict(test_input)

predition =  prediction.astype(int)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction.astype(int)

    })

submission.to_csv('titanic.csv', index=False)