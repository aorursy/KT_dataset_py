import sys
import sklearn
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
full = pd.concat([train,test])#, ignore_index=True)
full.head(1)

#train.isnull().sum(),train.count(),

full.Embarked.mode()
full.Embarked.fillna('S', inplace=True)

full[full.Fare.isnull()]


full.Fare.fillna(full[full.Pclass==3]['Fare'].median(),inplace=True)
full.loc[full.Cabin.notnull(),'Cabin']=1
full.loc[full.Cabin.isnull(),'Cabin']=0
full.Cabin.isnull().sum()
pd.pivot_table(full,index=['Cabin'] , values=['Survived']).plot.bar(figsize=(4,2))
plt.title('Taxa de sobrevivente')
cabin = pd.crosstab(full.Cabin,full.Survived)
cabin.rename(index={0:'sem cabine',1:'com babine'}, columns={0.0:'Morto',1.0:'Sobrevivente'}, inplace=True)
cabin
cabin.plot.bar(figsize=(5,2))
plt.xticks(rotation=0,size='xx-large')
plt.title('Número de sobreviventes')
plt.xlabel('')
plt.legend()
median = train.Age.median()
train.Age.fillna(median, inplace=True)
full.info()
full.head(3)
full.groupby(['Pclass'])[['Age','Pclass']].mean().plot(kind='bar', figsize=(5,3))
plt.xticks(rotation=0)
pd.crosstab(full.Sex,full.Survived).plot.bar(stacked=True,figsize=(5,3))
plt.legend(bbox_to_anchor=(0.55,0.9))
agehist = pd.concat([full[full.Survived==1]['Age'],full[full.Survived==0]['Age']], axis=1)
agehist.columns=['Sobrevivente','Morto']
agehist.plot(kind='hist', bins=30,figsize=(15,3),alpha=0.4)
full.ageCut = pd.cut(full.Age,5)
full.fareCut = pd.cut(full.Fare,5)
full.ageCut.value_counts().sort_index()
full.fareCut.value_counts().sort_index()
# substituir faixas
full.loc[full.Age<=16.136,'AgeCut']=1
full.loc[(full.Age>16.136)&(full.Age<=32.102),'AgeCut']=2
full.loc[(full.Age>32.102)&(full.Age<=48.068),'AgeCut']=3
full.loc[(full.Age>48.068)&(full.Age<=64.034),'AgeCut']=4
full.loc[full.Age>64.034,'AgeCut']=5

full.loc[full.Fare<=7.854,'FareCut']=1
full.loc[(full.Fare>7.854)&(full.Fare<=10.5),'FareCut']=2
full.loc[(full.Fare>10.5)&(full.Fare<=21.558),'FareCut']=3
full.loc[(full.Fare>21.558)&(full.Fare<=41.579),'FareCut']=4
full.loc[full.Fare>41.579,'FareCut']=5


full[['FareCut','Survived']].groupby(['FareCut']).mean().plot.bar(figsize=(8,3))
full.corr()
predictors=['Cabin','Embarked','Parch','Pclass','Sex','SibSp','AgeCut','FareCut','Age','Fare']
full_dummies=pd.get_dummies(full[predictors])
full_dummies.count()
medianAgeCut = full_dummies.AgeCut.median()
full_dummies.AgeCut.fillna(medianAgeCut, inplace=True)
medianAge = full_dummies.Age.median()

full_dummies.Age.fillna(medianAge, inplace=True)
x=full_dummies[:891]
y=full.Survived[:891]
x_test = full_dummies[891:]

scaler = StandardScaler()
x_scaled =scaler.fit(x).transform(x)
x_test_scaled =scaler.fit(x).transform(x_test)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(1, 13)),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(1, activation=keras.activations.hard_sigmoid)                                   
])
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Nadam(), metrics=["accuracy"])

#model.fit(x=x_scaled,y=y,  epochs=100,batch_size=32, steps_per_epoch=len(x) // 32)

#p = model.predict(x_test_scaled)
#submission = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':p.reshape(-1).astype('int64')})
#submission.Survived.value_counts()
#submission.to_csv('submission.csv', index=False)

