import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data_train = pd.read_csv("../input/titanic/train.csv")
print("*"*10,"Data Information", 10*"*")

print("\n")

print(data_train.info())
print("***  First 5 rows ***")

data_train.head()
data_train.tail()
data_test = pd.read_csv("../input/titanic/test.csv")
data_test.info()
# it should be all variable as integer
data_train.Sex.value_counts()

data_train.Name.value_counts()
data_train.Age.value_counts()
data_train.SibSp.value_counts()
data_train.Parch.value_counts()
data_train.Ticket.value_counts()
data_train.Fare.value_counts()
data_train.Cabin.value_counts()
data_train.head()
data_train.drop(["Name","Age","Ticket","Fare","Cabin"],  inplace = True,axis = 1)

data_test.drop(["Name","Age","Ticket","Fare","Cabin"], inplace = True,axis = 1)
print(data_train.info() ,  data_test.info() )
data_train.Embarked.value_counts()
data = [data_train, data_test]

for dataset in data:

    dataset.Embarked = dataset.Embarked.fillna("S") #if value is Nan, then fill in the blank as a S: because S is bigger than all of them (Q and C)
print(data_train.info(),data_test.info())
data_train.head()
##identfy string or object, and convert them int or float
data_train["Sex"].value_counts()
genderMap = {"male" : 0,"female" : 1}

for dataset in data:

    dataset["Sex"] = dataset["Sex"].map(genderMap)
data_train.head()
data_train["Embarked"].value_counts()
embarkedMap = {"S" : 0,"C" : 1,"Q":2}

for dataset in data:

    dataset["Embarked"] = dataset["Embarked"].map(embarkedMap)
data_test.head()
data_train.head()
print(data_train.info(),data_test.info())
#yes now all of them is integer
#Separating feature of X and Y
X_train = data_train.drop(["Survived","PassengerId"], axis = 1)

Y_train = data_train["Survived"]
X_test = data_test.drop(["PassengerId"], axis = 1)

Y_test = data_train["Survived"]
#selection model and train
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0)

clf.fit(X_train, Y_train)





Y_pred = clf.predict(X_test)

print(Y_pred)
##evaulate model
acc_logistic = round(clf.score(X_train,Y_train)*100,2)

print(acc_logistic)
output=pd.DataFrame({'PassengerId': data_test.PassengerId,'Survived': Y_pred})

output.to_csv("my_submission.csv", index = False)

print("Your submission was succesfully saved")