

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv('../input/train.csv')

df['FamSize'] = df.SibSp + df.Parch + 1

plt.figure(figsize=[12,5])

plt.scatter(x = df[df['Survived']>0].Age,y=df[df['Survived']>0].Fare,s = df.FamSize*15, c='c', alpha = 0.3, label = 'survived')

plt.scatter(x = df[df['Survived']<1].Age,y=df[df['Survived']<1].Fare,s = df.FamSize*15, c='m', alpha = 0.3, label = 'dead')

plt.title('Age vs Fare - Sizes according to family members')

plt.legend()

plt.xlabel('Age')

plt.ylabel('Fare')

plt.show()



df = df[df['Age'].notnull()]

plt.figure(figsize=(12,5))

plt.hist([df[df['Survived']==1]['Age'],df[df['Survived']==0]['Age']], stacked=True, color = ['c','m'], bins = 30,label = ['Survived','Dead'],alpha = 0.5)

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()



plt.show()

plt.figure(figsize=[12,5])

plt.hist([df[df['Survived']==1]['Fare'],df[df['Survived']==0]['Fare']], stacked=True, color = ['c','m'],

         bins = 30,label = ['Survived','Dead'],alpha=0.5)

plt.legend()

plt.xlabel('Fare')

plt.ylabel('Number of Passengers')

plt.show()
import re

def title(name):

    search = re.search(' ([A-Za-z]+)\.', name)

    if search:

        return search.group(1)

    return ""



df['Title'] = df['Name'].apply(title)



aux_graph = pd.get_dummies(df, columns = ['Title'])

aux_graph = aux_graph.drop(['PassengerId','Name','Sex','Age','SibSp','Parch','Pclass','Ticket','Fare','Cabin','Embarked','FamSize'],axis = 1)



deadTitle = aux_graph[aux_graph['Survived']==0].drop(['Survived'],axis=1)

survivedTitle = aux_graph[aux_graph['Survived']==1].drop(['Survived'],axis=1)



survivedTitle.columns=['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Master','Miss','Mlle','Mme','Mr','Mrs','Ms','Rev','Sir']

deadTitle.columns=['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Master','Miss','Mlle','Mme','Mr','Mrs','Ms','Rev','Sir']



plt.figure(figsize=[12,5])

plt.subplot(121)

survivedTitle.sum().plot(kind='bar',color='c', alpha = 0.5,stacked=True)

axes = plt.gca()

axes.set_ylim([0,450])

plt.legend('Survived')

plt.ylabel('Number of Passengers')



plt.subplot(122)

deadTitle.sum().plot(kind='bar',color='m', alpha = 0.5,stacked=True)

axes = plt.gca()

axes.set_ylim([0,450])

plt.legend('Dead')

plt.ylabel('Number of Passengers')



plt.show()