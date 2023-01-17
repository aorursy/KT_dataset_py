# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data =pd.read_csv("../input/train.csv")

print(train_data.shape)

train_data.head()
test_data =pd.read_csv("../input/test.csv")

print(test_data.shape)

test_data.head()
y=train_data['Survived']



y.head(5)
train_data.drop('Survived', axis=1, inplace=True)

combined = train_data.append(test_data)

combined.reset_index(inplace=True)

combined.drop('index', inplace=True, axis=1)

combined.head(5)
combined.shape
sns.heatmap(combined.isnull(),yticklabels=False,cbar=False, cmap='viridis')
sns.distplot(combined['Age'].dropna(),kde=False, bins=30)
sns.boxplot(data=combined,x='Pclass',y='Age')
def sub_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        

        if Pclass ==1:

            return 37

        elif Pclass ==2:

            return 27

                

        else:

            return 24

    else:

        return Age

    

    
combined['Age']=combined[['Age','Pclass']].apply(sub_age,axis=1)
sns.heatmap(combined.isnull(),yticklabels=False,cbar=False, cmap='viridis')
combined.drop('Cabin',axis=1, inplace=True)
combined.head()
sns.heatmap(combined.isnull(),yticklabels=False,cbar=False, cmap='viridis')
combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

combined.head(5)
combined['Title'].unique()
title_Dictionary = {

                        "Capt":       "Officer",

                        "Col":        "Officer",

                        "Major":      "Officer",

                        "Jonkheer":   "Royalty",

                        "Don":        "Royalty",

                        "Sir" :       "Royalty",

                        "Dr":         "Officer",

                        "Rev":        "Officer",

                        "the Countess":"Royalty",

                        "Dona":       "Royalty",

                        "Mme":        "Mrs",

                        "Mlle":       "Miss",

                        "Ms":         "Mrs",

                        "Mr" :        "Mr",

                        "Mrs" :       "Mrs",

                        "Miss" :      "Miss",

                        "Master" :    "Master",

                        "Lady" :      "Royalty"



                        }
combined['Title']=combined['Title'].map(title_Dictionary)
combined['Title'].unique()
combined.drop('Name',axis=1, inplace=True)

combined.head(5)
def cleanTicket(ticket):

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        ticket = map(lambda t : t.strip(), ticket)

        ticket = list(filter(lambda t : not t.isdigit(), ticket))

        if len(list(ticket)) > 0:

            return list(ticket)[0]

        else: 

            return 'XXX'
combined['Ticket']=combined['Ticket'].map(cleanTicket)



combined['Ticket'].head(5)
combined['Ticket'].unique()
ticket_dummy=pd.get_dummies(combined['Ticket'],prefix="ticket",drop_first=True)

combined['Parch'].unique()
parch_dummy=pd.get_dummies(combined['Parch'],prefix="Parch",drop_first=True)

combined['Pclass'].unique()
pclass_dummy=pd.get_dummies(combined['Pclass'],prefix="pclass",drop_first=True)

train_data['SibSp'].unique()
sibsp_dummy=pd.get_dummies(combined['SibSp'],prefix="sibsp",drop_first=True)
sex= pd.get_dummies(combined['Sex'],drop_first=True)



embarkedcity=pd.get_dummies(combined['Embarked'],drop_first=True)
title_dummy=pd.get_dummies(combined['Title'],drop_first=True)
combined=pd.concat([combined,sex,embarkedcity,title_dummy,ticket_dummy,parch_dummy,pclass_dummy,sibsp_dummy], axis=1)

combined.head(5)

combined.drop(labels=['PassengerId','Pclass','SibSp','Parch','Ticket','Title','Embarked','Sex'],axis=1, inplace=True)

combined.head(5)
combined.columns
X=combined.head(891)

#y=train_data['Survived']

len(combined.index)
from sklearn.model_selection import train_test_split



print(X.shape)

print(y.shape)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression

print(len(X_train.index))

print(len(X_test.index))

print(X_test)
logmodel=LogisticRegression()

logmodel.fit(X_train,y_train)
predicitons= logmodel.predict(X_test)



from sklearn.metrics import classification_report

print(classification_report(y_test,predicitons))
print(len(combined.index))

sub_predict=combined[891:]

len(sub_predict.index)
print (sub_predict)
sns.heatmap(sub_predict.isnull(),yticklabels=False,cbar=False, cmap='viridis')
from sklearn.preprocessing import Imputer

my_imputer = Imputer()

data_with_imputed_values = my_imputer.fit_transform(sub_predict)
prediction2=logmodel.predict(data_with_imputed_values)

df_output = pd.DataFrame()

aux = pd.read_csv('../input/test.csv')

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = prediction2

df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

output= pd.read_csv('output.csv')

output.head()