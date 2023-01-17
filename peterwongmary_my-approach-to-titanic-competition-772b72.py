import numpy as np

import pandas as pd
dfTrain = pd.read_csv("../input/train.csv") # importing train dataset

dfTest = pd.read_csv("../input/test.csv") # importing test dataset
dfTrain.info()
dfTest.info()
dfTrain.set_index(['PassengerId'],inplace=True)

dfTest.set_index(['PassengerId'],inplace=True)
import matplotlib as plt

%matplotlib inline
dfTrain.groupby('Pclass').Survived.mean().plot(kind='bar')
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)
dfTrain.groupby('Sex').Survived.mean().plot(kind='bar')
import seaborn as sns
sns.factorplot("Sex", "Survived", hue="Pclass", data=dfTrain)
dfFull = pd.concat([dfTrain,dfTest])

dfFull['Sex'] = dfFull['Sex'].map({'male': 0, 'female': 1}).astype(int)

dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]
dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)
dfTrain.groupby('Parch').Survived.mean().plot(kind='bar')
dfTrain.Parch.value_counts()
dfFull['ParchCat'] = dfFull.Parch.copy().astype(int)

dfFull.loc[dfFull.Parch > 1,'ParchCat'] = 2

dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]
dfTrain.groupby('ParchCat').Survived.mean().plot(kind='bar')
dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex','ParchCat']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex','ParchCat']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)
dfTrain.groupby('SibSp').Survived.mean().plot(kind='bar')
dfTrain.SibSp.value_counts()
dfFull['SibSpCat'] = dfFull.SibSp.copy().astype(int)

dfFull.loc[dfFull.SibSp > 1,'SibSpCat'] = 2

dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]
dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex','ParchCat','SibSpCat']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex','ParchCat','SibSpCat']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)
dfTrain.groupby('Embarked').Survived.mean().plot(kind='bar')
dfFull.groupby(['Embarked','Sex']).Name.count()
dfFull.groupby(['Embarked','Pclass']).Name.count()
dfFull[dfFull.Embarked.isnull()]
indexEmbarked = dfFull[dfFull.Embarked.isnull()].index.tolist()

for indEmb in indexEmbarked:

    fareEmbarked = dfFull.loc[indEmb].Fare.mean()

    predictedEmbarked = dfFull[(dfFull.Fare < fareEmbarked*1.1) &

                           (dfFull.Fare > fareEmbarked*0.9) &

                           (dfFull.Pclass == dfFull.loc[indEmb].Pclass)].Embarked.mode()

    dfFull.loc[indEmb,'Embarked'] = predictedEmbarked[0]

    print(predictedEmbarked)   
dfFull['Embarked'] = dfFull['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)

dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]
dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex','ParchCat','Embarked']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex','ParchCat','Embarked']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)
nullFares = dfFull[dfFull.Fare.isnull()].index.values

dfFull.loc[nullFares]
dfFull.loc[nullFares,'Fare']

dfFull.loc[nullFares,'Fare'] = dfFull[(dfFull.ParchCat == 0) & (dfFull.Pclass ==3 ) & (dfFull.Embarked == 0)].Fare.mean()

dfFull.loc[[1044]]
import matplotlib.pyplot as plt

plt.hist([dfTrain[dfTrain['Survived']==1]['Fare'], dfTrain[dfTrain['Survived']==0]['Fare']], stacked=True, color = ['g','r'],

         bins = 10,label = ['Survived','Not Survived'])

plt.legend()
fareMean = dfFull.Fare.mean()

dfFull.loc[dfFull.Fare <= fareMean,'Fare']=0

dfFull.loc[dfFull.Fare > fareMean,'Fare']=1

dfFull.Fare.value_counts()
dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex','ParchCat','Embarked','Fare']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex','ParchCat','Embarked','Fare']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)
dfFull.Cabin.value_counts(dropna=False)
dfFull.Cabin.str[0].value_counts(dropna=False)
dfFull['CabinCat'] = dfFull.Cabin.str[0].fillna('Z')

dfFull.loc[dfFull.CabinCat=='G','CabinCat']= 'Z'

dfFull.loc[dfFull.CabinCat=='T','CabinCat']= 'Z'

dfFull['CabinCat'] = dfFull['CabinCat'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'Z': 6}).astype(int)

dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]

dfTrain.groupby('CabinCat').Survived.mean().plot(kind='bar')
dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex','ParchCat','Embarked','Fare','CabinCat']]

X_train = pd.get_dummies(X_train)

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex','ParchCat','Embarked','Fare','CabinCat']]

X_test = pd.get_dummies(X_test)

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)
dfFull['Title'] = dfFull.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

dfFull.Title.value_counts()
dfFull['TitleCat']=dfFull['Title']

dfFull.TitleCat.replace(to_replace=['Rev','Dr','Col','Major','Mlle','Ms','Countess','Capt','Dona','Don','Sir','Lady','Jonkheer','Mme'],

                        value=0, inplace=True)

dfFull.TitleCat.replace('Mr',1,inplace=True)

dfFull.TitleCat.replace('Miss',2,inplace=True)

dfFull.TitleCat.replace('Mrs',3,inplace=True)

dfFull.TitleCat.replace('Master',4,inplace=True)                                            

dfFull.TitleCat.value_counts(dropna=False)
dfFull.corr().TitleCat
dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]

dtree = DecisionTreeClassifier()

X_train = dfFull[dfFull.TitleCat!=0][['Sex','ParchCat','SibSpCat']]

y = dfFull[dfFull.TitleCat!=0]['TitleCat']

X_test = dfFull[dfFull.TitleCat==0][['Sex','ParchCat','SibSpCat']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = X_test.index.values,columns=['TitleCat'])

#print(dfPrediction)
dfFull.TitleCat

dfFull.update(dfPrediction)
dfFull.loc[dfPrediction.index,['TitleCat','Title','Sex','SibSpCat','ParchCat']]
dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]

dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex','ParchCat','Embarked','Fare','TitleCat']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex','ParchCat','Embarked','Fare','TitleCat']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)
dfFull['Age'] = dfFull.Age.round()
dfFull.corr().Age
X_train = dfFull[dfFull.Age.notnull()][['Pclass','SibSp','CabinCat','TitleCat']]

X_test = dfFull[dfFull.Age.isnull()][['Pclass','SibSp','CabinCat','TitleCat']]

y = dfFull.Age.dropna()
dtree = DecisionTreeClassifier()

dtree.fit(X_train, y)

prediction = dtree.predict(X_test)

agePrediction = pd.DataFrame(data=prediction,index=X_test.index.values,columns=['Age'])

dfFull = dfFull.combine_first(agePrediction)
dfFull.Age.isnull().sum()

dfTrain = dfFull.loc[1:891,:].groupby('Age').Survived.mean().plot(kind='line',figsize=(12,5))
dfFull.loc[dfFull.Age<18,'Age'] = 0

dfFull.loc[(dfFull.Age>=18),'Age'] = 1

dfFull.Age.value_counts()
dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]

dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex','ParchCat','Embarked','Fare','TitleCat','Age']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex','ParchCat','Embarked','Fare','TitleCat','Age']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)

dfTicket = dfFull.Ticket.value_counts()

dfTicket.head()
lstTicket = dfTicket.loc[dfTicket>1].index.tolist()

lstTicketSingle = dfTicket.loc[dfTicket==1].index.tolist()
len(lstTicket)
len(lstTicketSingle)
dfFull['TicketCat'] = dfFull['Ticket']
i=1

for ticket in lstTicket:

    dfFull.loc[dfFull.Ticket == ticket, 'TicketCat'] = i

    i+=1

i=1

for ticket in lstTicketSingle:

    dfFull.loc[dfFull.Ticket == ticket, 'TicketCat'] = 0

    i+=1
dfFull.TicketCat.value_counts().head()
dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]

dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex','ParchCat','Embarked','Fare','TitleCat','Age','TicketCat']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex','ParchCat','Embarked','Fare','TitleCat','Age','TicketCat']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

print(contentTestPredObject1)
from sklearn.ensemble import RandomForestClassifier

rforest = RandomForestClassifier(n_estimators=30)

rforest.fit(X_train,y)

prediction = rforest.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)