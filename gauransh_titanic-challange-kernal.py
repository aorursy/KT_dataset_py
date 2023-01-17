import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display #Display DataFrame

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
display(df_train.head())

display(df_test.head())
df_train.info()
df_train.describe()
cabin_nan = df_train["Cabin"].isna().sum()

print("Number of NaN in Cabin : "+str(cabin_nan))

print("Percentage : "+str(cabin_nan/len(df_train["Cabin"])*100)+"%")
# Training Data

df_train.drop(columns="Cabin",inplace=True)

df_train.info()
# Testing Data

df_test.drop(columns="Cabin",inplace=True)

df_test.info()
df_train.Age.replace(np.nan, df_train.Age.median(), inplace=True)

df_train.info()
df_test.Age.replace(np.nan, df_test.Age.median(), inplace=True)

df_test.Fare.replace(np.nan, df_test.Fare.mean(), inplace=True)

df_test.info()
df_train.Embarked.replace(np.nan, df_train.Embarked.value_counts().idxmax(), inplace=True)

df_train.info()
df_train.drop(columns=["Name", "Ticket"], inplace=True)

df_train.head()
df_test.drop(columns=["Name", "Ticket"], inplace=True)

df_test.head()
#Pclass Training set

pclass_one_hot = pd.get_dummies(df_train['Pclass'])

df_train = df_train.join(pclass_one_hot)

df_train.head()
#Pclass Test set

pclass_one_hot_t = pd.get_dummies(df_test['Pclass'])

df_test = df_test.join(pclass_one_hot)
#Embarked Training set

embarked_one_hot = pd.get_dummies(df_train.Embarked)

df_train = df_train.join(embarked_one_hot)

df_train.head()
#Embarked Test set

embarked_one_hot_t = pd.get_dummies(df_test.Embarked)

df_test = df_test.join(embarked_one_hot_t)
#Sex Training set

sex_one_hot = pd.get_dummies(df_train.Sex)

df_train = df_train.join(sex_one_hot)

df_train.head()
#Sex Test set

sex_one_hot_t = pd.get_dummies(df_test.Sex)

df_test = df_test.join(sex_one_hot_t)
# Removing "Pclass","Sex","Embarked" from Training set

df_train.drop(columns=["Pclass","Sex","Embarked"], inplace=True)

df_train.head()
# Removing "Pclass","Sex","Embarked" from Testing set

df_test.drop(columns=["Pclass","Sex","Embarked"], inplace=True)

df_test.head()
# Normalization in Training set

x = df_train.values

min_max_scaler = MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

norm_df=pd.DataFrame(x_scaled)

df_train.Age = norm_df[2]

df_train.Fare = norm_df[5]

df_train.head()
# Normalization in Testing set

x_t = df_test.values

min_max_scaler = MinMaxScaler()

x_scaled_t = min_max_scaler.fit_transform(x_t)

norm_df_t=pd.DataFrame(x_scaled_t)

df_test.Age = norm_df_t[1]

df_test.Fare = norm_df_t[4]

df_test.head()
y= df_train.Survived.values

X= df_train.drop(columns=["Survived","PassengerId"]).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
clf = xgb.XGBClassifier(max_depth=8, learning_rate=0.2, n_estimators=100)

clf.fit(X_train, y_train)
print('Accuracy on test set: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Accuracy on Train set: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)

clf_rf.fit(X_train, y_train)
print('Accuracy on test set: {:.2f}'.format(accuracy_score(y_test, clf_rf.predict(X_test))))

print('Accuracy on Train set: {:.2f}'.format(accuracy_score(y_train, clf_rf.predict(X_train))))
X_pre = df_test.drop(columns="PassengerId").values

predicted_lables = clf_rf.predict(X_pre)

df_temp = pd.DataFrame(predicted_lables)

df_temp.columns = ["Survived"]

df_submission = df_test.join(df_temp)

df_submission = df_submission[["PassengerId", "Survived"]]

df_submission.head()
df_submission.to_csv("submission.csv", index=False)