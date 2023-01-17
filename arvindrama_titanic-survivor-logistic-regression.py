import warnings

warnings.filterwarnings('ignore')



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



train = pd.read_csv("../input/train.csv")
train.info()
print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
print (train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean())
from sklearn.preprocessing import OneHotEncoder , LabelEncoder



label_encoder_embarked = LabelEncoder()

train["Sex_Encoded"] = label_encoder_embarked.fit_transform(train["Sex"])

one_hot_encoder_embarked = OneHotEncoder(sparse=False)

x = one_hot_encoder_embarked.fit_transform(train["Sex_Encoded"].values.reshape(-1, 1))

df2=pd.DataFrame(data=x)

df2.columns = ["Female","Male"]

train = train.join(df2)



label_encoder_embarked = LabelEncoder()

train["Pclass_Encoded"] = label_encoder_embarked.fit_transform(train["Pclass"])

one_hot_encoder_embarked = OneHotEncoder(sparse=False)

x = one_hot_encoder_embarked.fit_transform(train["Pclass_Encoded"].values.reshape(-1, 1))

df2=pd.DataFrame(data=x)

df2.columns = ["Class1","Class2","Class3"]

train = train.join(df2)
X = train[['Age', 'Fare', 'Class1','Class2','Class3','Female','Male']].copy()

y = train.iloc[:,[1]].values

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
print(X_train.shape,X_test.shape)

print(y_train.shape,y_test.shape)
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

X_train = imp.fit_transform(X_train)

X_test = imp.transform(X_test)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,y_pred)
test_data = pd.read_csv("../input/test.csv")

label_encoder_embarked = LabelEncoder()

test_data["Sex_Encoded"] = label_encoder_embarked.fit_transform(test_data["Sex"])

one_hot_encoder_embarked = OneHotEncoder(sparse=False)

x = one_hot_encoder_embarked.fit_transform(test_data["Sex_Encoded"].values.reshape(-1, 1))

df2=pd.DataFrame(data=x)

df2.columns = ["Female","Male"]

test_data = test_data.join(df2)



label_encoder_embarked = LabelEncoder()

test_data["Pclass_Encoded"] = label_encoder_embarked.fit_transform(test_data["Pclass"])

one_hot_encoder_embarked = OneHotEncoder(sparse=False)

x = one_hot_encoder_embarked.fit_transform(test_data["Pclass_Encoded"].values.reshape(-1, 1))

df2=pd.DataFrame(data=x)

df2.columns = ["Class1","Class2","Class3"]

test_data = test_data.join(df2)
test_data_x = test_data[['Age', 'Fare', 'Class1','Class2','Class3','Female','Male']].copy()
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

test_data_x = imp.fit_transform(test_data_x)
sc_X = StandardScaler()

test_data_x = sc_X.fit_transform(test_data_x)
test_data_y_pred = classifier.predict(test_data_x)
test_data['Survived'] = test_data_y_pred
test_data.info()
test_data_submission = test_data[['PassengerId', 'Survived']].copy()
test_data_submission
test_data_submission.to_csv("survivor_submission.csv", sep=',', encoding='utf-8',index=False)