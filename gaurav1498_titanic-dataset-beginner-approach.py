import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

%matplotlib inline
df_Train=pd.read_csv('../input/titanic/train.csv')

df_Test=pd.read_csv('../input/titanic/test.csv')
df_Train.head()
df_Test.head()
print(df_Train.shape)

print(df_Test.shape)
df=df_Train.append(df_Test,sort=False)

df.head()
df.isnull().sum()
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna("S")

df['Fare']=df['Fare'].fillna(df['Fare'].mode())
df.isnull().sum()
df.dtypes
df.Cabin.value_counts()
df['Cabin'] = df['Cabin'].fillna('U')
import re

df['Cabin'] = df['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
df['Cabin'].unique()
cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}

df['Cabin'] = df['Cabin'].map(cabin_category)
df.head()
df['Name'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
df['Name'].unique()
df['Name'] = df['Name'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
df['Name'].unique()
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
df["Name"]=encoder.fit_transform(df['Name'])

df["Sex"]=encoder.fit_transform(df['Sex'])

df["Embarked"]=encoder.fit_transform(df['Embarked'])
df.head()
df['TitanicNumber']=df['Ticket'].str.extract('(\d{2,})', expand=True)

df['TitanicNumber']=df['TitanicNumber'].apply(pd.to_numeric)



df.drop('Ticket', axis=1, inplace=True)
df.head()
df.dtypes
df.corr()
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()



columns_to_scale = ['Fare', 'Age', 'PassengerId','TitanicNumber']

df[columns_to_scale] = sc.fit_transform(df[columns_to_scale])
df.head()
df.corr()
df.drop('Sex', axis=1, inplace=True)
df_train=df[0:891]

df_test=df[891:]
df_train['Survived']=df_train['Survived'].astype(int)
df_train.head()
df_test=df_test.drop(['Survived'],axis=1)

df_test.head()
print(df_train.shape)

print(df_test.shape)
X=df_train.drop(['Survived'],axis=1)

y=df_train[['Survived']]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.22566,random_state=6)
from catboost import CatBoostClassifier

CB = CatBoostClassifier(iterations=18,eval_metric="F1",

                                    learning_rate=0.2991,depth=4)





CB.fit(X_train, y_train,eval_set=(X_test, y_test))
pred = CB.predict(X_test)



from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(y_test, pred))

print(confusion_matrix(y_test, pred))
y_score1 = CB.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve, roc_auc_score

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)



print('roc_auc_score for Catboost: ', roc_auc_score(y_test, y_score1))



# Plot ROC curves

plt.subplots(1, figsize=(10,10))

plt.title('Receiver Operating Characteristic - Catboost')

plt.plot(false_positive_rate1, true_positive_rate1)

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"),plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
prediction = CB.predict(df_test)

prediction
output = pd.DataFrame({'PassengerId': df_Test.PassengerId, 'Survived': prediction})

#output.to_csv('titanic2020.csv', index=False)