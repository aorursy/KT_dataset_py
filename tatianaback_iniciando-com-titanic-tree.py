

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.dtypes

import pandas_profiling as pdp
#df = pd.read_csv('/kaggle/input/titanic/test.csv')

#pdp.ProfileReport(df)

#profile_report = df.profile_report()

#profile_report.to_file("/content/drive/My Drive/AI TiagoScopel/example.html")

#profile_report
train_data.isnull().sum()
train_data.shape
train_data = train_data.drop(columns =[

    "Name",

    "Ticket"

])
mapping = train_data.Cabin.value_counts()



df = train_data.Cabin.map(mapping)
train_data.loc[train_data['Cabin'].isnull(),'Cabin_value'] = 0

train_data.loc[train_data['Cabin'].notnull(), 'Cabin_value'] = 1



print (train_data)
test_data.loc[test_data['Cabin'].isnull(),'Cabin_value'] = 0

test_data.loc[test_data['Cabin'].notnull(), 'Cabin_value'] = 1



print (test_data)
train_data = train_data.drop(columns =[

    "Cabin"

])

test_data = test_data.drop(columns =[

    "Cabin"

])
train_data =pd.get_dummies(train_data)

test_data =pd.get_dummies(test_data)
train_data.dtypes
train_data.head()
train_data.isnull().sum()
y = train_data["Survived"]



features = ["Pclass","Sex_male", "SibSp", "Parch", "Cabin_value", "Embarked_C", "Embarked_Q","Embarked_S"]

X = train_data[features]

X_test = test_data[features]



from sklearn import model_selection

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import(

LogisticRegression,

)



from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import(KNeighborsClassifier,)



#help(model_selection)
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import (

RandomForestClassifier,)

import xgboost
from sklearn.model_selection import KFold
for model in[

    DummyClassifier,

    LogisticRegression,

    DecisionTreeClassifier,

    GaussianNB,

    SVC,

    RandomForestClassifier,

    xgboost.XGBClassifier,

    ]:

    cls=model()

    kfold = model_selection.KFold(n_splits=10,random_state=42)

    s=model_selection.cross_val_score(

    cls, X,y,scoring="roc_auc",cv=kfold

    )

    print(

    f"{model.__name__:22} AUC: "

    f"{s.mean():3f} STD: {s.std():.2f}"

    )
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestClassifier(

n_estimators=100,random_state=42)



rf.fit(X,y)

predictions = rf.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission2.csv', index=False)

print("Your submission was successfully saved!")
rf.score(X, y)