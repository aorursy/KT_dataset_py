import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/titanic/train.csv')
df=pd.read_csv('../input/titanic/train.csv')
df_test=pd.read_csv('../input/titanic/test.csv')
df.drop(["Cabin"],inplace=True,axis=1)
df.drop(["Ticket"],inplace=True,axis=1)
df.drop(["Name"],inplace=True,axis=1)
onehot = pd.get_dummies(df['Sex'])
df.drop(["Sex"],inplace=True,axis=1)
df=df.join(onehot)
onehot_em = pd.get_dummies(df['Embarked'])
df.drop(["Embarked"],inplace=True,axis=1)
df=df.join(onehot_em)




df_test.drop(["Cabin"],inplace=True,axis=1)
df_test.drop(["Ticket"],inplace=True,axis=1)
df_test.drop(["Name"],inplace=True,axis=1)
onehot_test = pd.get_dummies(df_test['Sex'])
df_test.drop(["Sex"],inplace=True,axis=1)
df_test=df_test.join(onehot_test)
onehot_em_test = pd.get_dummies(df_test['Embarked'])
df_test.drop(["Embarked"],inplace=True,axis=1)
df_test=df_test.join(onehot_em_test)
df=df.fillna(df.mean())
df_test=df_test.fillna(df_test.mean())
x_train=df.loc[:,df.columns!='Survived']
x_test=df_test
y_train=df["Survived"]
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=128, max_depth=None,max_features='log2', min_samples_split=3, random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

print(y_pred.reshape(-1,1))
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
train_score_RF = classifier.score(x_train, y_train)
print("Train score RandomForest :", train_score_RF)
output = pd.DataFrame({'PassengerId': x_test.PassengerId, 'Survived': y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
