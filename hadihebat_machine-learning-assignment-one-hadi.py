"""Import modules"""

import pandas as pd

from sklearn.naive_bayes import GaussianNB
"""Read Csv files"""

dfTrain = pd.read_csv("../input/train.csv")

dfTest = pd.read_csv("../input/test.csv")
dfTrain["CAge"]=pd.cut(dfTrain["Age"], bins = [0,10,18,40,max(dfTrain["Age"])] ,labels=["Child","MYoung","Young","Older"])

dfTest["CAge"]=pd.cut(dfTest["Age"], bins = [0,10,18,40,max(dfTest["Age"])] ,labels=["Child","MYoung","Young","Older"])
"""Make dummy variables for categorical data"""

dfTrain= pd.get_dummies(data = dfTrain, dummy_na=True, prefix= ["Pclass","Sex","Embarked","Age"] ,columns=["Pclass","Sex","Embarked","CAge"])

dfTest= pd.get_dummies(data = dfTest, dummy_na=True, prefix= ["Pclass","Sex","Embarked","Age"] ,columns=["Pclass","Sex","Embarked","CAge"])



"""Store the train outcomes for survived"""

Y_train=dfTrain["Survived"]
"""Store PassengerId"""

submission=pd.DataFrame()

submission["PassengerId"]=dfTest["PassengerId"]



"""Ignore useless data"""

dfTrain=dfTrain[dfTrain.columns.difference(["Age","Survived","PassengerId","Name","Ticket","Cabin"])]

dfTest=dfTest[dfTest.columns.difference(["Age","PassengerId","Name","Ticket","Cabin"])]



"""handling a Nan value"""

dfTest["Fare"].iloc[dfTest[dfTest["Fare"].isnull()].index] = dfTest[dfTest["Pclass_3.0"]==1]["Fare"].median()
"""Fit Model"""

clf = GaussianNB()

clf.fit(dfTrain,Y_train)
"""Make a Csv with Results"""

pred = pd.DataFrame(clf.predict(dfTest),columns=["Survived"])

submission=submission.join(pred,how="inner")

submission.to_csv("../working/submit.csv", index=False)

submission.head(10)