

import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
train_data.count()

#Age,Cabin,Embarked
test_data.count()

#Age,Fare,Cabin
train_data.describe()
test_data.describe()
#Feature Engineering



complete_df = [train_data, test_data] 

train_data.drop(['Ticket'],axis = 1, inplace = True)

train_data.drop(['Cabin'],axis = 1, inplace = True)

train_data.drop(['Name'],axis = 1, inplace = True)

test_data.drop(['Ticket'],axis = 1, inplace = True)

test_data.drop(['Cabin'],axis = 1, inplace = True)

test_data.drop(['Name'],axis = 1, inplace = True)



for datadf in complete_df:    

    #complete missing age with median

    datadf['Age'].fillna(datadf['Age'].median(), inplace = True)

    

    datadf['Embarked'].fillna(datadf['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    datadf['Fare'].fillna(datadf['Fare'].median(), inplace = True)
train_data['Sex'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

train_data['Embarked'].replace(to_replace=['C','Q', 'S'], value=[0,1,2],inplace=True)



test_data['Sex'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

test_data['Embarked'].replace(to_replace=['C','Q', 'S'], value=[0,1,2],inplace=True)
train_data.describe()
test_data.describe()
train_data.head()
test_data.head()
#Una vez listo el data cleaning empezamos a probar los algoritmos de ML



from sklearn import preprocessing

from sklearn.model_selection import train_test_split



X = train_data

y = train_data["Survived"].values

X.drop(["Survived"],axis = 1, inplace = True)



X_submit = test_data



X = preprocessing.StandardScaler().fit(X).transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.metrics import jaccard_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss



DT_model = DecisionTreeClassifier(criterion="entropy")

DT_model.fit(X_train,y_train)



DT_yhat = DT_model.predict(X_test)



print("DT accuracy: %.2f" % accuracy_score(y_test, DT_yhat))

#print("DT Jaccard index: %.2f" % jaccard_score(y_test, DT_yhat,pos_label='1'))

print("DT F1-score: %.2f" % f1_score(y_test, DT_yhat, average='weighted') )
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



RanFor_model = RandomForestClassifier(n_estimators=100,random_state=1).fit(X_train,y_train)

#RanFor_yhat

predictions = RanFor_model.predict(X_test)



#features = ["Pclass", "Sex", "SibSp", "Parch"] X = pd.get_dummies(train_data[features]) X_test = pd.get_dummies(test_data[features])

#model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1) model.fit(X, y) predictions = model.predict(X_test)





print("Random Forest accuracy: %.2f" % accuracy_score(y_test, predictions))

#print("Random Forest Jaccard index: %.2f" % jaccard_score(y_test, RanFor_yhat,pos_label='1'))

print("Random Forest F1-score: %.2f" % f1_score(y_test, predictions, average='weighted') )
from sklearn.linear_model import LogisticRegression



LR_model = LogisticRegression(C=0.01).fit(X_train,y_train)

LR_yhat = LR_model.predict(X_test)



print("LR accuracy: %.2f" % accuracy_score(y_test, LR_yhat))

#print("LR Jaccard index: %.2f" % jaccard_score(y_test, LR_yhat,pos_label='1'))

print("LR F1-score: %.2f" % f1_score(y_test, LR_yhat, average='weighted') )
predictions = RanFor_model.predict(X_submit)

print(predictions)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False) 

print("Your submission was successfully saved!")
