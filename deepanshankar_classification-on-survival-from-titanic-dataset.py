# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        if(filename == 'train.csv'):

            train_data = pd.read_csv(os.path.join(dirname, filename))

            train_data["Source"] = "Train"

        elif(filename == 'test.csv'):

            test_data = pd.read_csv(os.path.join(dirname, filename))

            test_data["Source"] = "Test"

        else:

            submission_file = os.path.join(dirname, filename)



df = pd.concat([train_data,test_data],axis=0)

df.head()



# Any results you write to the current directory are saved as output.
df.isna().sum()
df['Age'].fillna(df['Age'].mean(),inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df.drop(["Name","Ticket"],1,inplace=True)
df['Cabin'].fillna("No Cabin",inplace=True)

df['Cabin_Status'] = df['Cabin'].apply(lambda x : 0 if (x == "No Cabin") else 1)
df.drop(["Cabin"],inplace=True,axis=1)
df["Pclass"] = df["Pclass"].astype("object")

pclass_dummies = pd.get_dummies(df[["Pclass"]])
df = pd.concat([df,pclass_dummies],1)

df.drop(["Pclass"],1,inplace=True)
df["Sex"] = df["Sex"].map({"male":1,"female":0})

em_dummies = pd.get_dummies(df[["Embarked"]])

df = pd.concat([df,em_dummies],1)



df.drop("Embarked",axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

df[['Age','Fare']] = scale.fit_transform(df[['Age','Fare']])
train = df[df["Source"] == "Train"]

test = df[df["Source"] == "Test"]
X = train.drop(["Source","PassengerId"],1)

y = train["Survived"]

X.drop("Survived",1,inplace=True)
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.30)
print(X_train.shape)

print(X_val.shape)

print(y_train.shape)

print(y_val.shape)

from sklearn.metrics import confusion_matrix,classification_report



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

model_lr = lr.fit(X_train,y_train)



lr_y = model_lr.predict(X_val)



print(confusion_matrix(y_val,lr_y))

print(classification_report(y_val,lr_y))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()



model_knn = knn.fit(X_train,y_train)



knn_y = model_knn.predict(X_val)



print(confusion_matrix(y_val,knn_y))

print(classification_report(y_val,knn_y))
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

model_gbc = gbc.fit(X_train,y_train)

y_gbc = model_gbc.predict(X_val)



print(confusion_matrix(y_val,y_gbc))

print(classification_report(y_val,y_gbc))
from xgboost import XGBClassifier

gbc = XGBClassifier()

model_gbc = gbc.fit(X_train,y_train)

y_gbc = model_gbc.predict(X_val)



print(confusion_matrix(y_val,y_gbc))

print(classification_report(y_val,y_gbc))



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000,max_depth=10,min_samples_split=2)

model_rf = rf.fit(X_train,y_train)



rf_y = model_rf.predict(X_val)



print(confusion_matrix(y_val,rf_y))

print(classification_report(y_val,rf_y))



test.head()
test.drop("Survived",1,inplace=True)
test.shape
pass_id = test["PassengerId"]

test.drop("PassengerId",1,inplace=True)
test.isna().sum()
test["Fare"].fillna(test["Fare"].mean(),inplace=True)
test.drop("Source",1,inplace=True)
y_test = model_rf.predict(test)
df_survived = pd.DataFrame(y_test,columns=["Survived"])

df_survived["Survived"] = df_survived["Survived"].astype("int")

df_survived
df_pass = pd.DataFrame(pass_id,columns=["PassengerId"])

df_submission = pd.concat([df_pass,df_survived],1)
df_submission.head()
df_submission.shape
df_submission.to_csv("Titanic_Submission.csv",index=False)