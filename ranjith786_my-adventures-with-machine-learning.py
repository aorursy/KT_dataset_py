import pandas as pd
import numpy as np
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.drop(["Name","Ticket"],axis=1,inplace=True)
test.drop(["Name","Ticket"],axis=1,inplace=True)
train.head()
train.describe()
#train.Age
train.info()
def getCabinCode(x):
    try:
        return x[0]
    except TypeError:
        return "N"
def categorial_to_ZOS(df):
    
    categorial_columns = list(df.dtypes[df.dtypes == 'object'].index)
    
    print (categorial_columns)
    for column in categorial_columns:
        dummies = pd.get_dummies(df[column],prefix=column)
        
        df = pd.concat([df,dummies],axis=1)
        
        df.drop([column],axis=1,inplace=True)
        
    print (list(df.columns))
    return df
train.Age.fillna(train.Age.mean(),inplace=True)
train.Embarked.fillna("S",inplace=True)
#train.Cabin.fillna("N",inplace=True)
train.Cabin = train.Cabin.apply(getCabinCode)
train.Fare.fillna(train.Fare.mean(),inplace=True)

train = categorial_to_ZOS(train)


test.Age.fillna(test.Age.mean(),inplace=True)
test.Embarked.fillna("S",inplace=True)
#train.Cabin.fillna("N",inplace=True)
test.Cabin = test.Cabin.apply(getCabinCode)
test.Fare.fillna(test.Fare.mean(),inplace=True)

test = categorial_to_ZOS(test)
train.info()
train.drop(["PassengerId","Cabin_T"],axis=1,inplace=True)
Survived = train.pop("Survived")
train.info()
#train.Cabin.value_counts()
#test.Cabin.value_counts()
test.info()
features = list(test.columns)
features.remove('PassengerId')
features

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
model = RandomForestClassifier(n_estimators=200,max_features=18)
model.fit(train,Survived)
model.score(train, Survived)
test["Survived"] = model.predict(test[features])
submission = pd.DataFrame(test,columns=["PassengerId","Survived"])
submission.to_csv("titanic.csv",columns=["PassengerId","Survived"],index=False)
submission.head()
submission.Survived.value_counts().plot(kind='bar',title="Survival Count")
