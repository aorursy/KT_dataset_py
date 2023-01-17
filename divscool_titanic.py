import numpy as np
import pandas as pd
df_train = pd.read_csv('../input/train.csv')
df_train.describe(include = 'all')
df_train = df_train.drop(["PassengerId"], axis=1)
df_train["Embarked"].unique()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df_train['Embarked'] = df_train['Embarked'].fillna('None')
df_train = df_train.drop(["Name" , "Ticket"], axis = 1)
df_train['Cabin'] = df_train['Cabin'].fillna('None')
label = LabelEncoder()
label.fit(df_train["Sex"])
df_train["Sex_code"] = label.transform(df_train["Sex"])
label1 = LabelEncoder()
label1.fit(df_train["Embarked"])
df_train["Embarked_code"] = label1.transform(df_train["Embarked"])
df_train
df_train = df_train.drop(["Sex" , "Cabin" , "Embarked"] , axis =1)
df_train.describe(include = 'all')
from numpy import median
med = np.nanmedian(df_train["Age"])
df_train["Age"] = df_train["Age"].fillna(med)
df_test = df_train["Survived"]
df_train = df_train.drop(["Survived"] , axis = 1)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(df_train , df_test)
df_real_test_initial = pd.read_csv("../input/test.csv")
df_real_test = df_real_test_initial
df_real_test = df_real_test.drop(["PassengerId"], axis=1)
df_real_test = df_real_test.drop(["Name" , "Ticket"], axis = 1)
df_real_test['Cabin'] = df_real_test['Cabin'].fillna('None')
df_real_test["Sex_code"] = label.transform(df_real_test["Sex"])
df_real_test["Embarked_code"] = label1.transform(df_real_test["Embarked"])
df_real_test = df_real_test.drop(["Sex" , "Cabin" , "Embarked"] , axis =1)
df_real_test["Age"] = df_real_test["Age"].fillna(med)
df_real_test.describe(include = 'all')
df_real_test.fillna(0)
writer = pd.ExcelWriter('after_output.xlsx')
df_real_test.to_excel(writer,'Sheet1')
writer.save()
df_real_test["Fare"] = df_real_test["Fare"].fillna(0)
predict = clf.predict(df_real_test)
df_real_test = df_real_test.drop(["Pclass" ,"Age" , "SibSp" , "Parch" , "Fare" , "Sex_code" , "Embarked_code"] , axis = 1)
df_real_test["PassengerId"] = df_real_test_initial["PassengerId"]
df_real_test["Survived"] = predict
writer = pd.ExcelWriter('result_titanic.xlsx')
df_real_test.to_excel(writer,'Sheet1')
writer.save()
df_real_test

