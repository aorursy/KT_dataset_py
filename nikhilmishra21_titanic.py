import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



full = train.append(test,ignore_index = True)

titanic = full[:891]



print(f'Full Shape: {full.shape}\n Titanic shape: {titanic.shape}')
titanic.head()
titanic.describe()
#Plotting the Correlation amongst the features

sns.heatmap(titanic.corr(),annot=True,cmap='Pastel2')

plt.tight_layout()
facet = sns.FacetGrid(titanic,hue='Survived',aspect=4,row='Sex')

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,titanic['Age'].max()))

facet.add_legend()
facet = sns.FacetGrid(titanic,hue='Survived',aspect=4,row='Sex')

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim = (0,titanic['Fare'].max()))

facet.add_legend()
sns.barplot(x='Embarked',y='Survived',data=titanic,palette='Pastel2')
sns.barplot(x='Sex',y='Survived',data=titanic,palette='Pastel2')
sns.barplot(x='Pclass',y='Survived',data=titanic,palette='Pastel2')
sns.barplot(x='SibSp',y='Survived',data=titanic,palette='Pastel2')
sns.barplot(x='Parch',y='Survived',data=titanic,palette='Pastel2')
full.info()
imputed = pd.DataFrame()



imputed['Age'] = full['Age'].fillna(full['Age'].mean())



imputed['Fare'] = full['Fare'].fillna(full['Fare'].mean())



imputed.head()
sex = pd.Series(np.where(full.Sex == 'male',0,1),name = 'Sex')

embarked = pd.get_dummies(full.Embarked,prefix='Embarked')

pclass = pd.get_dummies(full.Pclass,prefix = 'Pclass')



print(f'{sex.head()}\n')

print(f'{embarked.head()}\n')

print(pclass.head())
#Extracting title out of the Name



title = pd.DataFrame()

title['Title'] = [full.Name[i].split(',')[1].split('.')[0].strip() for i in range(len(full.Name))]

# a map of more aggregated titles

Title_Dictionary = {

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

title['Title'] = title['Title'].map(Title_Dictionary) 

title = pd.get_dummies(title.Title)



#print the change 

title.head()
#Extracting Cabin's from Cabin

cabin = pd.DataFrame()



#filling null values

cabin['Cabin'] = full.Cabin.astype('object').fillna('U')



#Creating the categorical variables

cabin['Cabin'] = [cabin['Cabin'][i][0] for i in range(len(cabin['Cabin']))]

cabin = pd.get_dummies(cabin['Cabin'],prefix='Cabin')



#print the result

cabin.head()
full['Ticket'] = full['Ticket'].astype('object')
def cleanticket(ticket):

    ticket = ticket.replace('.','')

    ticket = ticket.replace('/','')

    ticket = ticket.split()

    ticket = map(lambda t: t.strip(),ticket)

    ticket = list(filter(lambda t:not t.isdigit(),ticket))

    if len(ticket) > 0:

        return ticket[0]

    else:

        return 'XXX'

ticket = pd.DataFrame()



ticket['Ticket'] = full['Ticket'].map(cleanticket)

ticket = pd.get_dummies(ticket['Ticket'])



ticket.head()
family = pd.DataFrame()

family['FamilySize'] = full.Parch + full.SibSp + 1



family['Family_Single'] = family['FamilySize'].map(lambda s:1 if s==0 else 0)

family['Family_Small'] = family['FamilySize'].map(lambda s:1 if 2<=s<=4 else 0)

family['Family_Large'] = family['FamilySize'].map(lambda s:1 if 5<=s else 0)



family.head()
full_X = pd.concat([imputed,sex,embarked,pclass,title,cabin,family],axis=1)

full_X.head()