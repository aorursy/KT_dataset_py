import numpy as np

import pandas as pd
dfTrain = pd.read_csv("../input/train.csv") # importing train dataset

dfTest = pd.read_csv("../input/test.csv") # importing test dataset
dfTrain.head(1).info()
dfTest.head(1).info()
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
dfFull.Parch.value_counts()
dfFull['ParchCat'] = dfFull.Parch.copy().astype(int)

dfFull.loc[dfFull.Parch > 2,'ParchCat'] = 3

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
dfFull.SibSp.value_counts()
dfFull['SibSpCat'] = dfFull.SibSp.copy().astype(int)

dfFull.loc[dfFull.SibSp > 2,'SibSpCat'] = 3

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

plt.ylabel('No. of Passengers')

plt.xlabel('Fare')
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

dfFull['ageBins'] = pd.cut(dfFull['Age'],list(range(0,80,5)))
sns.factorplot("ageBins", "Survived", hue="Sex", data=dfFull.loc[1:891,:],size=7)
dfFull.loc[1:891,:].groupby(['Sex','ageBins']).Name.count()
dfFull.loc[dfFull.Age <11,'Age'] = 0

dfFull.loc[(dfFull.Age >=10),'Age'] = 1

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
lstTicket = dfTicket.loc[dfTicket > 1].index.tolist()

lstTicketSingle = dfTicket.loc[dfTicket == 1].index.tolist()
len(lstTicket)
len(lstTicketSingle)
dfFull[dfFull.Ticket=='347082'].Name
dfFull['TicketCat'] = dfFull['Ticket'].copy()
i=1

for ticket in lstTicket:

    dfFull.loc[dfFull.Ticket == ticket, 'TicketCat'] = i

    i+=1
for ticket in lstTicketSingle:

    dfFull.loc[dfFull.Ticket == ticket, 'TicketCat'] = 0
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

#print(contentTestPredObject1)
dfFull['NameLen'] = dfFull['Name'].apply(len)

dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]

dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex','ParchCat','Embarked','Fare','TitleCat','Age','TicketCat','NameLen']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex','ParchCat','Embarked','Fare','TitleCat','Age','TicketCat','NameLen']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)
dfResult = pd.Series({'Pclass':65.55,

                      '+Sex':75.59,

                      '+Parch':76.07,

                      '+Embarked':77.51,

                      '+Fare':77.99,

                      '+Title':79.42,

                      '+Age':79.90,

                      '+Ticket':80.38})

dfResult.sort_values().plot(kind='bar',title='LeaderBoard result (%)')
from sklearn.model_selection import train_test_split

X_train = dfTrain[['Pclass','Sex','ParchCat','SibSpCat','Embarked','Fare','TitleCat','Age','TicketCat']]

X_test = dfTest[['Pclass','Sex','ParchCat','SibSpCat','Embarked','Fare','TitleCat','Age','TicketCat']]

y = dfTrain['Survived']

X_NewTrain, X_NewTest,y_NewTrain, y_NewTest = train_test_split(X_train, y,

                                         test_size=0.33, # 1/3 test; 2/3 train

                                          random_state=1410) # seed
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.tree import DecisionTreeClassifier



classifiers = [

    KNeighborsClassifier(),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LogisticRegression(),

    LinearSVC()]



for clf in classifiers:

    name = clf.__class__.__name__

    clf.fit(X_NewTrain, y_NewTrain)

    prediction = clf.predict(X_NewTest)

    rank = pd.DataFrame(data=np.column_stack([prediction, y_NewTest]),

                                index=X_NewTest.index.values,columns=['Predicted','Real'])

    accurracy = np.sum(rank.Predicted.values == rank.Real.values)

    accurracy = accurracy/len(y_NewTest)

    print(accurracy, name)
dfTrain = dfFull.loc[1:891,:]

dfTest = dfFull.loc[892:,:]

dtree = DecisionTreeClassifier()

X_train = dfTrain[['Pclass','Sex','ParchCat','SibSpCat','Embarked','Fare','TitleCat','Age','TicketCat','CabinCat']]

y = dfTrain['Survived']

X_test = dfTest[['Pclass','Sex','ParchCat','SibSpCat','Embarked','Fare','TitleCat','Age','TicketCat','CabinCat']]

dtree.fit(X_train,y)

prediction = dtree.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = dfTest.index.values,columns=['Survived'])

contentTestPredObject1 = dfPrediction.to_csv()

#print(contentTestPredObject1)