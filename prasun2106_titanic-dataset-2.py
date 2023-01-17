# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
sns.heatmap(pd.isnull(train), cbar = True, cmap= "viridis")
sns.countplot(x = "Survived", data = train)
train.columns
sns.countplot("Pclass", data = train)
sns.countplot("Sex", data = train)
sns.countplot(x = "Survived", hue ="Parch", data = train, palette = "RdBu")
train.columns
sns.distplot(train["Age"].dropna(), kde = False)
train["Age"].plot.hist()
sns.countplot(train.SibSp)
train.info()
sns.countplot(train.Fare)
train["Fare"].hist()
plt.hist(train["Fare"], bins = 60)
import cufflinks as cf
cf.go_offline()
train["Fare"].iplot(kind ="hist", bins =30)
plt.boxplot(train["Pclass"])
sns.boxplot(x = "Pclass", y = "Age", data = train, palette = "Accent_r")
sns.boxplot(x = "Pclass", y = "Age", data = train)
train.columns
sns.boxplot(x = "SibSp", y = "Age", data = train)
sns.boxplot(x = "Parch", y = "Age", data = train)
combine = train.append(test)
combine.head()
train.shape
test.shape
combine.shape
combine.head()
combine.loc[1]
train_new = train.drop("Survived", axis = 1)
combine = train_new.append(test)
train_new.shape

combine.shape

test.shape
combine.count()
sns.boxplot(x = "Pclass", y= "Age", data = combine)
combine = combine.drop("Cabin", axis =1)
combine.info()
from sklearn.impute import SimpleImputer
imp= SimpleImputer(missing_values= np.NaN, strategy = "median")
combine["Fare"] = combine["Fare"].fillna(combine.Fare.median())
combine.info()
from sklearn.preprocessing import LabelEncoder
labelencoder_Embarked = LabelEncoder()
combine["Sex"] = labelencoder_Embarked.fit_transform(combine.iloc[:,3])
combine.head()
sns.countplot(combine["Embarked"])
combine["Embarked"] = combine["Embarked"].fillna("S")
combine["Embarked"] = labelencoder_Embarked.fit_transform(combine.iloc[:,9])
combine.head()
combine = combine.drop(["Ticket", "Name"], axis = 1)
combine.info()
combine.head()
pd.DataFrame(combine["Embarked"]).count()
missing = combine.loc[combine["Age"].isnull(), :]
missing.head()
missing

not_missing = combine.loc[ ~combine["Age"].isnull(), :]
not_missing.info()
missing.info()
X_train = not_missing.drop("Age", axis = 1)
X_test = missing.drop("Age", axis = 1)
X_train.head()

y_train = not_missing["Age"]
from sklearn.linear_model import LinearRegression
model_age = LinearRegression()
model_age.fit(X_train, y_train)
from sklearn import metrics
prediction = model_age.predict(X_test)
prediction

missing["Age"] = prediction
missing["Age"]
train.info()
from sklearn.model_selection import train_test_split

combine.info()
total_data = not_missing.append(missing)
total_data.info()
total_data
total_data = total_data.drop("PassengerId", axis =1)
train.count()
missing_train = train.loc[train["Age"].isnull(), :]
not_missing_train = train.loc[~train["Age"].isnull(), :]
model_Age = LinearRegression()
X_train  = not_missing_train.drop("Age", axis = 1)
y_train = not_missing_train["Age"]
sns.heatmap(train.isnull())
train = train.drop("Cabin", axis = 1)
sns.boxplot(train["Pclass"], train["Age"])
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train.Pclass.value_counts()
train["Age"] = train[["Age", "Pclass"]].apply(impute_age, axis = 1)
train.info()
train["Embarked"] = train["Embarked"].fillna("S")
train.Embarked.value_counts()
train.info()
train.head()
train = train.drop(["Name", "Ticket"], axis= 1 )
train.head()
from sklearn.preprocessing import LabelEncoder
labelencoder.fit(train["Embarked"])
train.head()
train["Embarked"] = labelencoder.fit_transform(train["Embarked"])
train.head()
train["Sex"] = labelencoder.fit_transform(train["Sex"])
train.head()

sns.boxplot(x = "Pclass", y = "Fare", data = train)
train_new = train.drop([258, 679, 737], axis = 0)
sns.boxplot(x = "Pclass", y = "Fare", data = train_new)
train = train_new
train.head()
train = train.drop("PassengerId", axis= 1)
train.head()
test.head()
test = test.drop(["Cabin", "Name", "PassengerId", "Ticket"], axis = 1)
test.info()
sns.boxplot(test["Pclass"], test["Age"])
test.info()
labelencoder_test_age = LabelEncoder()
test["Sex"]= labelencoder_test_age.fit_transform(test["Sex"])
test.head()
test["Embarked"]= labelencoder_test_age.fit_transform(test["Embarked"])
test.head()
def impute_age2(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 42

        elif Pclass == 2:

            return 28

        else:

            return 27

    else:

        return Age
test["Age"] = test[["Age", "Pclass"]].apply(impute_age2, axis = 1)
test.head()
test.head()
train.head()
test.info()
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test.info()
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
X_train = train.drop("Survived", axis = 1)
y_train = train["Survived"]
X_test = test
log_model.fit(X_train, y_train)
y_test = log_model.predict(X_test)
a = pd.read_csv("../input/test.csv")
b = a["PassengerId"]
b
submission = pd.DataFrame({'PassengerId' : a["PassengerId"], 'Survived': y_test})
pd.DataFrame(y_test).count()
submission.to_csv("submission3.csv", index= False)
submission = pd.DataFrame({'PassengerId' : a["PassengerId"], 'Survived': y_test})

submission.to_csv("submission3.csv", index= False)
submission.head()
temp_submission = submission

temp_submission.head()

temp_submission.info()
import base64

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
create_download_link(submission)
