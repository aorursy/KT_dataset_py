# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

data=train[pd.to_numeric(train['Age'], errors='coerce').notnull()]

X_train=data[["Pclass","Sex","Age","Parch","Fare","Embarked"]]

Y_train=data["Survived"]

X_train=pd.get_dummies(X_train,columns=["Pclass","Sex","Embarked"])





userID=test["PassengerId"]

pred=test[["Pclass","Sex","Age","Parch","Fare","Embarked"]]

X_test=pd.get_dummies(pred,columns=["Pclass","Sex","Embarked"])



X_test["Age"]=X_test["Age"].fillna((X_test["Age"]).mean())

X_test["Fare"]=X_test["Fare"].fillna((X_test["Age"]).mean())

X_test.isna().sum()

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



random_forest
X_train.columns==X_test.columns
p1=random_forest.predict(X_test)



result=pd.DataFrame({"PassengerId":userID,"Survived":p1})

# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(result)