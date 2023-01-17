import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

dfTrain = pd.read_csv('../input/train.csv')

dfTrain.head()
namesDF = dfTrain.drop(['Pclass','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','PassengerId'],1)
title = []

for index, row in namesDF.iterrows():

    t = row['Name'].split(' ')

    title.append(t[1])

print(Counter(title))

title = pd.Series(title)

namesDF['Name'] = title.values

namesDF.head()
namesDF = pd.get_dummies(namesDF, prefix=['Name','Sex'],columns=['Name','Sex'])

namesDF.head()
namesDF.drop(['Name_Col.','Name_Planke,','Name_Billiard,','Name_Impe,','Name_Major.',

              'Name_Gordon,','Name_Mlle.','Name_Carlo,','Name_Ms.','Name_Messemaeker,','Name_Mme.',

              'Name_Capt.','Name_Jonkheer.','Name_the','Name_Don.','Name_der',

              'Name_Mulder,','Name_Pelsmaeker,','Name_Shawah,','Name_Melkebeke,','Name_Velde,',

              'Name_Walle,','Name_Steen,','Name_Cruyssen,','Name_y'],1,inplace=True)
namesDF.columns = ['Survived','Dr','Master','Ms','Mr','Mrs','Rev','Female','Male']
namesCor = namesDF.corr()

mask = np.zeros_like(namesCor, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(9, 7))

cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(namesCor, mask=mask, cmap=cmap, vmax=.3,

            square=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

plt.title("Names Correlation")

plt.show()
namesDF = namesDF.drop(['Dr'],1)
finalDF = namesDF

finalDF.head()
ageSexDF = dfTrain.drop(['Pclass','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','PassengerId'],1)
ageSexDF.head()
ageSexDF.mean()
ageWDF = ageSexDF[ageSexDF.Sex != 'male']

ageMDF = ageSexDF[ageSexDF.Sex != 'female']

print(ageWDF.mean())

print(ageMDF.mean())
f,ax = plt.subplots(figsize=(8, 6))

logWAge = sns.regplot(x='Age',y='Survived',data=ageWDF,logistic=True,ax=ax,color="#F08080")

logMAge = sns.regplot(x='Age',y='Survived',data=ageMDF,logistic=True,ax=ax,color="#6495ED")

plt.title("LogOdds: Survival vs Age by Gender \n Women: Orange | Men: Blue")

plt.show(logWAge)

plt.show(logMAge)
ageSexDF = pd.get_dummies(ageSexDF,prefix=['Sex'],columns=['Sex'])

femAge = ageSexDF.Sex_female * ageSexDF.Age

malAge = ageSexDF.Sex_male * ageSexDF.Age

ageSexDF['FemAge'] = femAge

ageSexDF['MalAge'] = malAge

ageSexDF.drop(['Sex_female','Sex_male','Age'],1,inplace=True)

ageSexDF.head()
finalDF = pd.concat([finalDF.drop(['Female','Male'],1), ageSexDF.drop(['Survived'],1)], axis=1)

finalDF.head()
totFam = dfTrain.SibSp + dfTrain.Parch

#Just Need an empty DF

famDF = dfTrain.drop(['Pclass','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','PassengerId','Name','Sex'],1)

famDF['Family'] = totFam

famDF.head(10)
famDF.loc[famDF['Family'] > 0, 'Family'] = 1

famDF.head(10)
withFamDF = famDF[famDF.Family == 1]

noFamDF = famDF[famDF.Family ==0]

barFam = sns.factorplot(x='Family',y='Survived',data=famDF,kind='bar',palette = ["#F08080","#6495ED"])

plt.title("Survival vs Family Onboard")

plt.show(barFam)

print("With Family: ",withFamDF.mean()[0], " No Family: ",noFamDF.mean()[0])
finalDF = pd.concat([finalDF,famDF.drop(['Survived'],1)],axis=1)

finalDF.head()
wealthDF = dfTrain.drop(['Age','SibSp','Parch','Ticket','Cabin','Embarked','PassengerId','Name','Sex'],1)

wealthDF.head()
barClass = sns.factorplot('Pclass','Survived',data=wealthDF,kind='bar',palette = ["#F08080","#6495ED","#A5FEE3"])

plt.title("Survival vs Class")

plt.show(barClass)



g,ax2 = plt.subplots(figsize=(8, 6))

logFare = sns.regplot(x='Fare',y='Survived',data=wealthDF[wealthDF.Fare < 500],logistic=True,ax=ax2,color="#F08080")

plt.title("LogOdds: Survival vs Fare")

plt.show(logFare)
wealthDF['Sex'] = dfTrain['Sex']

barClassSex = sns.factorplot(x = 'Sex', y='Survived', col='Pclass',data=wealthDF

                              ,kind='bar',palette = ["#6495ED","#F08080"])



wealthDFMen = wealthDF[wealthDF.Fare < 500][wealthDF.Sex == 'female']

wealthDFWomen = wealthDF[wealthDF.Fare < 500][wealthDF.Sex == 'male']

h,ax3 = plt.subplots(figsize=(9, 7))

fig5 = sns.regplot(x='Fare',y='Survived',data=wealthDFWomen,logistic=True,ax=ax3,color="#F08080")

fig6 = sns.regplot(x='Fare',y='Survived',data=wealthDFMen,logistic=True,ax=ax3,color="#6495ED")

plt.title("LogOdds: Survival vs Fare by Sex \n Women: Orange | Men: Blue")

plt.show(fig5)

plt.show(fig6)
wealthDFDum = pd.get_dummies(wealthDF,prefix=["Pclass"],columns=['Pclass'])

wealthDFDumCor = wealthDFDum.corr()

print(wealthDFDumCor)
finalDF = pd.concat([finalDF,wealthDFDum.drop(['Survived','Pclass_2','Sex'],1)],axis=1)

finalDF.head()
cabin = []

for index, row in dfTrain.iterrows():

    try:

        c = row['Cabin']

        cabin.append(c[0])

    except Exception:

        cabin.append(c)



print(Counter(cabin))

cabin = pd.Series(cabin)

cabinDF = dfTrain.drop(['Age','SibSp','Parch','Ticket','Fare','Embarked','PassengerId','Name','Sex','Pclass'],1)

cabinDF['Cabin'] = cabin.values

print(cabinDF.head(10))
barCabin = sns.factorplot(x = 'Cabin', y = 'Survived', kind='bar',data = cabinDF)

plt.title("Survival vs Cabin")

plt.show(barCabin)
cabinDF = pd.get_dummies(cabinDF, prefix=['Cabin'], columns=['Cabin'])

cabinDF.drop(['Cabin_G','Cabin_T'],1,inplace=True)

finalDF = pd.concat([finalDF,cabinDF,],axis=1)

finalDF.head()
embarkedDF = dfTrain.drop(['Age','SibSp','Parch','Ticket','Fare','Cabin','PassengerId','Name','Sex','Pclass'],1)

embarkedDF.head()

barEmbarked = sns.factorplot('Embarked','Survived',data=embarkedDF,kind='bar',

                               palette = ["#F08080","#6495ED","#A5FEE3"])

plt.title("Survival vs Embarked")

plt.show(barEmbarked)
embarkedDFDum = pd.get_dummies(embarkedDF,prefix=['Embarked'],columns=['Embarked'])

print(embarkedDFDum.corr())
finalDF = pd.concat([finalDF,embarkedDFDum.drop(['Embarked_Q','Survived'],1)],axis=1)

finalDF.head()
finalDF = finalDF[finalDF.Fare < 500]

features = finalDF.drop(['Survived'],1)

labels = finalDF['Survived']
finalDF = finalDF.dropna()
finalDF = finalDF[finalDF.Fare < 500]

features = finalDF.drop(['Survived'],1)

labels = finalDF['Survived']
from sklearn.ensemble import RandomForestClassifier

accuracy = []

for i in range(100):

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(features,labels)

    accuracy.append(clf.score(features,labels))

print(sum(accuracy)/len(accuracy))
from sklearn import cross_validation

accuracy2 = []

for i in range(100):

    featureTrain, featureTest, labelTrain, labelTest = cross_validation.train_test_split(

        finalDF.drop(['Survived'], 1),

        finalDF['Survived'], test_size=0.20)





    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(featureTrain,labelTrain)

    accuracy2.append(clf.score(featureTest,labelTest))



print(sum(accuracy2)/len(accuracy2))
