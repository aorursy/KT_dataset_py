import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import os

from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from sklearn.metrics import classification_report
def string_remover(df,list1=[],drop=[]):

    a = df.select_dtypes(include='object')

    for i in a.columns:

        for x in a.index:

            try:

                c = list1.index(a[i].iloc[x:x+1][x])

                a[i].iloc[x:x+1][x] = c

            except:

                list1.append(a[i].iloc[x:x+1][x])

                a[i].iloc[x:x+1][x] = len(list1)-1

    a.fillna(len(list1))

    d = df.select_dtypes(exclude='object').fillna(0)

    try:

        return pd.concat([d,a],axis=1).fillna(0).drop([drop],axis=1)

    except:

        return pd.concat([d,a],axis=1).fillna(0)
df_test = pd.read_csv('../input/titanic/test.csv')

df_train = pd.read_csv('../input/titanic/train.csv')
df_train
sns.heatmap(df_train.isnull(),cmap='viridis')
df_train.drop(['Cabin','Fare','Parch','Ticket','Name'],axis=1,inplace = True)
df_test.drop(['Cabin','Fare','Parch','Ticket','Name'],axis=1,inplace = True)
df_test.fillna(35,inplace=True)
df_train.fillna(35,inplace=True)
sns.heatmap(df_train.isnull(),cmap = 'viridis')
train = string_remover(df_train)
test = string_remover(df_test)
sns.heatmap(df_test.isnull(),cmap = 'viridis')
train = train.drop(['PassengerId'],axis=1)

test = test.drop(['PassengerId'],axis=1)


model = Sequential()



model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))



model.add(Dense(1))



model.compile(optimizer='rmsprop',loss='mse')
model.fit(train.drop(['Survived'],axis=1),train['Survived'],epochs=200)
pred = model.predict_classes(test)
idpred = []

for i in range(len(pred)):

    idpred.append(pred[i][0])
p = pd.Series(idpred)
submit = pd.concat([df_test['PassengerId'],p],axis=1)
submit.rename(columns={0:'Survived'},inplace = True)
submit.to_csv('Submission.csv',index=False)
submit