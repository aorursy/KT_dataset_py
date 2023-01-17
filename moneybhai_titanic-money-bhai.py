import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

data_train = pd.read_csv("../input/titanic/train.csv")
data_test= pd.read_csv("../input/titanic/test.csv")

#data_train.sample(10)

sns.barplot(x="Pclass", y="Survived", hue="Sex", data=data_train);


#grouping age feature of train data

data_train.Age = data_train.Age.fillna(-0.5) 
bins= (-1, 0, 10, 30, 70, 80)
group_names= ['unknown','child','youngadult','adult','seniorcitizen']

data_train.Age= pd.cut(data_train.Age, bins, labels=group_names)

#grouping age feature of test data

data_test.Age = data_test.Age.fillna(-0.5) 
bins= (-1, 0, 10, 30, 70, 80)
group_names= ['unknown','child','youngadult','adult','seniorcitizen']

data_test.Age= pd.cut(data_test.Age, bins, labels=group_names)

#dropping other irrelevant features

data_train = data_train.drop(['Ticket', 'Name', 'Embarked','Fare','SibSp','Parch','Cabin'], axis=1)
data_test = data_test.drop(['Ticket', 'Name', 'Embarked','Fare','SibSp','Parch','Cabin'], axis=1)

#encoding the train and test data 

features = [ 'Pclass','Age', 'Sex']


data_combined = pd.concat([data_train[features], data_test[features]] )


for feature in features:
    le = preprocessing.LabelEncoder()
    le = le.fit(data_combined[feature])
    data_train[feature] = le.transform(data_train[feature])
    data_test[feature] = le.transform(data_test[feature])
    
    

from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

#splitting up the training data  into train and test

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


#using random forest classifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 


clf = RandomForestClassifier()
#clf = svm.SVC(kernel='linear',C=0.1,gamma=0.1)
#clf = LogisticRegression()
#clf = DecisionTreeClassifier()
#clf = KNeighborsClassifier() 
#clf = GaussianNB()



clf.fit(X_train, y_train)

#predict accuracy with 20% train data (X_test)

acc_scorer = make_scorer(accuracy_score)
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

predictions_out = clf.predict(data_test.drop('PassengerId', axis=1))
titanic = pd.DataFrame({ 'PassengerId' : data_test['PassengerId'], 'Survived': predictions_out })


titanic.to_csv('predictions.csv' , index=False)


