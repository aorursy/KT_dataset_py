# setting up
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/titanic/train.csv")
df.drop(["PassengerId", "Name","Ticket", "Cabin"],inplace=True, axis = 1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df.drop(df[df['Embarked'].isna()].index, inplace = True)
from sklearn import preprocessing
l_enc1 = preprocessing.LabelEncoder()
le_df = df.copy(deep=True)
le_df['Sex'] = l_enc1.fit_transform(le_df['Sex'])
l_enc2 = preprocessing.LabelEncoder()
le_df['Embarked'] = l_enc2.fit_transform(le_df['Embarked'].astype(str))
le_df.head()
ohe_df = df.copy(deep=True)

ohe_df = df.copy(deep=True)
oh_s = pd.get_dummies(ohe_df['Sex'])
ohe_df.drop("Sex", inplace=True, axis = 1)
ohe_df = ohe_df.join(oh_s)
oh_e = pd.get_dummies(ohe_df['Embarked'].astype(str))
ohe_df.drop("Embarked", inplace=True, axis = 1)
ohe_df = ohe_df.join(oh_e)
oh_c = pd.get_dummies(ohe_df['Pclass'], prefix="class")
ohe_df.drop("Pclass", inplace=True, axis = 1)
ohe_df = ohe_df.join(oh_c)

ohe_df.head()
ohe_df_simplified = df.copy(deep=True)
oh_s = pd.get_dummies(ohe_df_simplified['Sex'],drop_first=True)
ohe_df_simplified.drop("Sex", inplace=True, axis = 1)
ohe_df_simplified = ohe_df_simplified.join(oh_s)
oh_e = pd.get_dummies(ohe_df_simplified['Embarked'].astype(str),drop_first=True)
ohe_df_simplified.drop("Embarked", inplace=True, axis = 1)
ohe_df_simplified = ohe_df_simplified.join(oh_e)
oh_c = pd.get_dummies(ohe_df_simplified['Pclass'], prefix="class",drop_first=True)
ohe_df_simplified.drop("Pclass", inplace=True, axis = 1)
ohe_df_simplified = ohe_df_simplified.join(oh_c)
ohe_df_simplified.head()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
def KNN_accuracy(dataframe):
    X = dataframe.drop('Survived', axis=1)
    y = dataframe['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    KNN = KNeighborsClassifier()
    KNN.fit(X_train,y_train)
    preds = KNN.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(acc)
print("KNN Model")
print("Label Encoder Accuracy:", end = " ")
KNN_accuracy(le_df)
print("OHE accuracy:", end = " ")
KNN_accuracy(ohe_df)
print("OHE (with first column dropped) accuracy:", end = " ")
KNN_accuracy(ohe_df_simplified)
def RF_accuracy(dataframe):
    X = dataframe.drop('Survived', axis=1)
    y = dataframe['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    RF = RandomForestClassifier(max_depth=2, random_state=0)
    RF.fit(X_train,y_train)
    preds = RF.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(acc)
print("Random Forest Model")
print("Label Encoder Accuracy:", end = " ")
RF_accuracy(le_df)
print("OHE accuracy:", end = " ")
RF_accuracy(ohe_df)
print("OHE (with first column dropped) accuracy:", end = " ")
RF_accuracy(ohe_df_simplified)