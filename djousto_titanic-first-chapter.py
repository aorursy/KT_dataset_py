# Pandas pour manipuler les tableaux de données

import pandas as pd

pd.set_option('display.max_columns', 500)



# Numpy pour les listes de données numériques et les fonctions classiques mathématiques

import numpy as np



# scipy (librairie scientifique) pour les fonctions statistiques et autres utilisaires

import scipy



# scikit learn pour les outils de machine learning

import sklearn

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler



# librairies pour la visualisation de données

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl



# et quelques options visuelles

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

sns.set(style="whitegrid", color_codes=True)

sns.set(rc={'figure.figsize':(15,8)})
# cette fonction évalue la corrélation entre variables qualitatives en 

# - élaboration du tableau de contingence des valeurs

# - calcul du chi2 de cet tableau 

# - calcul du coefficient de cramer qui est une normalisation du coefficient chi2

def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
# élaboration de deux listes de variables aléatoires

x = np.random.randint(0,20, size=1000)

y = np.random.randint(0,20, size=1000)
# test pour deux listes décorrélées

cramers_v(x,y)
# test pour deux listes totalement corrélées

cramers_v(x,x)
def eta_squared(x,y):

    moyenne_y = y.mean()

    classes = []

    for classe in x.unique():

        yi_classe = y[x==classe]

        classes.append({'ni': len(yi_classe),

                        'moyenne_classe': yi_classe.mean()})

    SCT = sum([(yj-moyenne_y)**2 for yj in y])

    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])

    return SCE/SCT
# les données ont été téléchargées sur kaggle et enregistrées en local 

# dans un répertoire similaire que celui en ligne sur kaggle, ainsi le script

# pourra tourner aussi bien en ligne qu'en local

train = pd.read_csv('../input/titanic/train.csv')

test =  pd.read_csv('../input/titanic/test.csv')

train.head()
# affichage des 5 premières lignes du jeu de test

test.head()
test['Survived']=np.nan

data=pd.concat([train,test],keys=['train','test'], join='inner')

data.index=data.index.droplevel(level=1)
len(data)
len(data.loc['test'])
# affichage de la liste et format des différentes variables

data.info()
# est-ce qu'il y a des doublons dans les identifiants

data.PassengerId.duplicated().sum()
# describe permet d'avoir quelques éléments statistiques sur les données

data.PassengerId.describe()
# liste des valeurs prises par Survived, on utilise value_counts() qui permet de lister les valeurs prise 

# pas une variable donnée ainsi que l'occurence de chaque valeur

data.Survived.value_counts()
sns.set(rc={'figure.figsize':(15,8)})

_ = sns.countplot(x="Survived", data=data)
# liste des valeurs prises par Pclass

data.Pclass.value_counts()
_ = sns.barplot(x="Pclass",y="Survived", data=data)
correlation = {}

correlation['Pclass']=cramers_v(data.loc['train','Pclass'],data.loc['train','Survived'])

print(correlation['Pclass'])
pd.crosstab(data["Survived"],data["Pclass"])
data.Name[0]
data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
data.Title.value_counts()
pd.crosstab(data.Title,data.Survived)
data['Title']=data['Title'].replace(['Major','Don','Jonkheer','Capt','Sir'], 'Mr')

data['Title']=data['Title'].replace(['Mlle'], 'Miss')

data['Title']=data['Title'].replace(['Lady','the Countess','Mme','Ms','Dona'], 'Mrs')

data.Title.value_counts()
# voyons si le titre a une influence sur la survie

pd.crosstab(data['Title'], data['Survived']).apply(lambda r: r/r.sum(), axis=1)
correlation['Title']=cramers_v(data.loc['train']['Title'],data.loc['train']['Survived'])

print(correlation['Title'])
data['Last Name'] = data['Name'].str.split(", ", expand=True)[0]
data['Last Name'].describe()
data.loc['train'].groupby('Last Name').agg({'Survived': 'var'}).mean()
data['randNumCol'] = np.random.choice(2,len(data))
data.loc['train'].groupby('Last Name').agg({'randNumCol': 'var'}).mean()
data=data.drop('randNumCol',axis=1)
from sklearn import preprocessing

MyEncoder=preprocessing.LabelEncoder()

MyEncoder.fit(data['Last Name'].sort_values())

data['LastNameNum'] = MyEncoder.transform(data['Last Name'])

data.head(1)
correlation['LastNameNum']=cramers_v(data['LastNameNum'],data['Survived'])

print(correlation['LastNameNum'])
data['First Name'] = data['Name'].str.split("\. ", expand=True)[1].str.split(" ", expand=True)[0]
data['First Name'].head()
gender = {'male': 1,'female': 0}

data.Sex = [gender[item] for item in data.Sex]
correlation['Sex']=cramers_v(data.loc['train']['Sex'],data.loc['train']['Survived'])

print(correlation['Sex'])
_ = sns.barplot(x="Sex",y="Survived", data=data)
sns.distplot(data.Age.dropna())
data.Age.describe()
scipy.stats.shapiro(data.Age.dropna())
sns.boxplot(x="Sex", y="Age",hue="Survived", data=data)
# voyons en détail la répartition des survivants par sexe et par age

sns.set(style="darkgrid", palette="pastel", color_codes=True)

ax = sns.violinplot(x="Sex", y="Age", hue="Survived",

               palette={0.0: "grey", 1.0: "white"},

               split=True, inner="quart",

               data=data)

ax.set_xticklabels(['Femme','Homme'])
dataTemp = data.loc['train'].dropna(subset=['Age'])



correlation['Age']=eta_squared(dataTemp.Age,dataTemp.Survived)

print(correlation['Age'])
data['TicketSibling'] = data.groupby('Ticket')['Ticket'].transform('count')
data.loc[(data.Age.isna())&(data['TicketSibling']==1),'Age']=30
data.loc[(data.Age.notna())&(data.Title=="Mrs"),'Age'].mean()
data.loc[(data.Age.isna())&(data.Title=="Mrs"),'Age']=37
# les données ont été enregistrées dans le fichier age.csv

age = pd.read_csv('../input/titanicdata/age.csv')
import unicodedata



def text_normalize(text):

    text = unicodedata.normalize('NFD', text)

    text = text.encode('ascii', 'ignore')

    text = text.decode("utf-8")

    return str(text)



# on va parcourir la liste des passagers et regarder si les nom de famille correspond, en cas de correspondance unique

# on sait qu'on a trouvé le bon passager et on lui attribue la valeur 1 en colonne Servant

# sinon on croise aussi la donnée prenom pour être sûr



for _,passager in data[data.Age.isna()].iterrows():

    for _,line in age.iterrows():

        x = line['Surname'].split(' ')[0]

        x = text_normalize(x)

        LastName = passager['Last Name']

        FirstName = passager['First Name']

        y = line['First Name'].split(' ')[0]

        y = text_normalize(y)

        if ((x in LastName)&(y in FirstName)):

            data.loc[data.PassengerId==passager.PassengerId,'Age'] = int(line['Age'])
data.Age.isna().sum()
data[data.Ticket=='2661']
# ce sont tous les deux des enfants

data.loc[(data.Age.isna())&(data.Ticket=='2661'),'Age']=10
data[data.Ticket=='371110']
# il s'agit de l'épouse de Mr MORAN, on va rire qu'elle a à peu près le même age

data.loc[(data.Age.isna())&(data.Ticket=='371110'),'Age']=25
data[data.Ticket=='2668']
# ce sont tous les deux des enfants

data.loc[(data.Age.isna())&(data.Ticket=='2668'),'Age']=10
data[data.Ticket=='4133']
# ce sont tous des enfants

data.loc[(data.Age.isna())&(data.Ticket=='4133'),'Age']=10
data[data.Ticket=='2665']
# ce sont des enfants, deux soeurs

data.loc[(data.Age.isna())&(data.Ticket=='2665'),'Age']=10
data[data.Ticket=='367230']
# ce sont des adultes, deux soeurs

data.loc[(data.Age.isna())&(data.Ticket=='367230'),'Age']=25
data[data.Ticket=='370365']
# ce sont des adultes, mari et femme

data.loc[(data.Age.isna())&(data.Ticket=='370365'),'Age']=30
data[data.Ticket=='2627']
# on ne peut pas être sur mais on peut imaginer que ce sont deux adultes

data.loc[(data.Age.isna())&(data.Ticket=='2627'),'Age']=18
data[data.Ticket=='PC 17757']
# on ne peut pas être sur mais on peut imaginer que c'est un adulte voyageant avec le reste de la famille

data.loc[(data.Age.isna())&(data.Ticket=='PC 17757'),'Age']=25
data[data.Ticket=='367226']
# certainement un adulte

data.loc[(data.Age.isna())&(data.Ticket=='367226'),'Age']=25
data[data.Ticket=='370371']
# certainement un adulte

data.loc[(data.Age.isna())&(data.Ticket=='370371'),'Age']=20
data[data.Ticket=='CA. 2343']
# certainement un enfant

data.loc[(data.Age.isna())&(data.Ticket=='CA. 2343'),'Age']=10
data[data.Ticket=='A/5. 851']
# certainement un enfant

data.loc[(data.Age.isna())&(data.Ticket=='A/5. 851'),'Age']=10
data.Age.isna().sum()
data['isParent']=np.nan

data['isEnfant']=np.nan
data.loc[(data.Parch==0),'isEnfant']=0

data.loc[(data.Parch==0),'isParent']=0



data.loc[(data.Age<14),'isEnfant']=1

data.loc[(data.Age<14),'isParent']=0



data.loc[(data.Parch>0)&(data.Title=='Mrs'),'isEnfant']=0

data.loc[(data.Parch>0)&(data.Title=='Mrs'),'isParent']=1



data.loc[(data.Parch>0)&(data.Title=='Mr'),'isEnfant']=0

data.loc[(data.Parch>0)&(data.Title=='Mr'),'isParent']=1
for x in data.LastNameNum.unique():

    if len(data[(data.LastNameNum==x)&(data.Title=='Mrs')]):

        data.loc[(data.LastNameNum==x)&(data.Title=='Miss'),'isParent']=0

        data.loc[(data.LastNameNum==x)&(data.Title=='Miss'),'isEnfant']=1

        data.loc[(data.LastNameNum==x)&(data.Title=='Master'),'isParent']=0

        data.loc[(data.LastNameNum==x)&(data.Title=='Master'),'isEnfant']=1

    

    ageMax = data[data.LastNameNum==x].Age.max()

    ageMin = data[data.LastNameNum==x].Age.min()

    data.loc[(data.LastNameNum==x)&(data.Age<(ageMax-18)),'isEnfant']=1

    data.loc[(data.LastNameNum==x)&(data.Age>(ageMin+18)),'isEnfant']=0

    data.loc[(data.LastNameNum==x)&(data.Age<(ageMax-18)),'isParent']=0

    data.loc[(data.LastNameNum==x)&(data.Age>(ageMin+18)),'isParent']=1

for x in data.Ticket.unique():

    if len(data[(data.Ticket==x)&(data.Title=='Mrs')]):

        data.loc[(data.Ticket==x)&(data.Title=='Miss'),'isParent']=0

        data.loc[(data.Ticket==x)&(data.Title=='Miss'),'isEnfant']=1

        data.loc[(data.Ticket==x)&(data.Title=='Master'),'isParent']=0

        data.loc[(data.Ticket==x)&(data.Title=='Master'),'isEnfant']=1

    

    ageMax = data[data.Ticket==x].Age.max()

    ageMin = data[data.Ticket==x].Age.min()

    data.loc[(data.Ticket==x)&(data.Age<(ageMax-18)),'isEnfant']=1

    data.loc[(data.Ticket==x)&(data.Age>(ageMin+18)),'isEnfant']=0

    data.loc[(data.Ticket==x)&(data.Age<(ageMax-18)),'isParent']=0

    data.loc[(data.Ticket==x)&(data.Age>(ageMin+18)),'isParent']=1

data.loc[(data.SibSp>1),'isParent']=0

data.loc[(data.SibSp>1),'isEnfant']=1



data.loc[(data.Parch>2),'isParent']=1

data.loc[(data.Parch>2),'isEnfant']=0
data.isParent.isna().sum()
data[data.isParent.isna()]
data.loc[data.PassengerId==137,'isParent']=0

data.loc[data.PassengerId==137,'isEnfant']=1



data.loc[data.PassengerId==418,'isParent']=0

data.loc[data.PassengerId==418,'isEnfant']=1



data.loc[data.PassengerId==540,'isParent']=0

data.loc[data.PassengerId==540,'isEnfant']=1



data.loc[data.PassengerId==1130,'isParent']=0

data.loc[data.PassengerId==1130,'isEnfant']=1



data.loc[data.PassengerId==1041,'isParent']=1

data.loc[data.PassengerId==1041,'isEnfant']=0
_ = sns.barplot(x="isEnfant",y="Survived", data=data)
correlation['isEnfant']=cramers_v(data.loc['train']['isEnfant'],data.loc['train']['Survived'])

print(correlation['isEnfant'])
data['AgeBin']=pd.cut(data['Age'],bins=[0,5,30,65,100],labels=[0,1,2,3],include_lowest=True).astype(int)
cramers_v(data.loc['train']['AgeBin'],data.loc['train']['Survived'])
sns.barplot(x="AgeBin",y="Survived", data=data)
correlation['AgeBin']=cramers_v(data.loc['train']['AgeBin'],data.loc['train']['Survived'])

print(correlation['AgeBin'])
data['Age']=StandardScaler().fit_transform(data['Age'].values.reshape(-1, 1))
correlation['Age']=cramers_v(data.loc['train']['Age'],data.loc['train']['Survived'])

print(correlation['Age'])
data.SibSp.describe()
data.SibSp.value_counts()
sns.barplot(x="SibSp",y="Survived", data=data)
correlation['SibSp']=cramers_v(data.loc['train']['SibSp'],data.loc['train']['Survived'])

print(correlation['SibSp'])
data.Parch.value_counts()
sns.barplot(x="Parch",y="Survived", data=data)
correlation['Parch']=cramers_v(data.loc['train']['Parch'],data.loc['train']['Survived'])

print(correlation['Parch'])
data[['Cabin','PassengerId']].info()
data['Cabin']=data['Cabin'].fillna('X').astype(str).str[0]
sns.barplot(x="Cabin",y="Survived", data=data)
_ = sns.countplot(x="Cabin", data=data)
correlation['Cabin']=cramers_v(data.loc['train']['Cabin'],data.loc['train']['Survived'])

print(correlation['Cabin'])
data = data.drop(['Cabin'],axis=1)
data.Ticket.describe()
data.groupby('Ticket').agg({'Survived': 'var'}).Survived.value_counts()
correlation['Ticket']=cramers_v(data.loc['train']['Ticket'],data.loc['train']['Survived'])

print(correlation['Ticket'])
data.Embarked.value_counts()
data.Embarked.isna().sum()
data[data.Embarked.isna()]
data[data.Ticket==42]
# pareil pour le port d'embarquement (Embarked), il n'en manque que deux (dans train pas dans test)

# plutot que de faire de fausses hypothèses on supprime les données manquantes

data = data.dropna(subset=['Embarked'])
_ = sns.barplot(x="Embarked",y="Survived", data=data)
_=sns.countplot(x="Embarked", data=data)
correlation['Embarked']=cramers_v(data.loc['train']['Embarked'],data.loc['train']['Survived'])

print(correlation['Embarked'])
port = {'C': 0,'Q': 1,'S':2}

data.Embarked = [port[item] for item in data.Embarked]
data.Fare.describe()
# nombre de valeurs manquantes

data.Fare.isna().sum()
data[data.Fare.isna()]
data[data.Pclass==3].Fare.describe()
# on remplace donc la valeur manquante par la moyenne de la 3e classe

data.Fare=data.Fare.fillna(13)
_= sns.distplot(data.Fare)
(data.Fare==0.0).sum()
data[data.Fare==0]
ax = sns.boxplot(x="Pclass", y="Fare", data=data[(data.Title=="Mr")&(data.Embarked==2)&(data.AgeBin==1)&(data.SibSp==0)&(data.Parch==0)])
for classe in data.Pclass.unique():

    PrixMoyen = data[(data.Title=="Mr")&

         (data.Embarked==2)&

         (data.AgeBin==1)&

         (data.SibSp==0)&

         (data.Parch==0)&

         (data.Fare!=0.0)&

         (data.Pclass==classe)].Fare.mean()

    print("Prix moyen de la classe ",classe," = ",PrixMoyen)
data.loc[(data.Fare==0.0)&

         (data.Pclass==1),'Fare']=37

data.loc[(data.Fare==0.0)&

         (data.Pclass==2),'Fare']=16

data.loc[(data.Fare==0.0)&

         (data.Pclass==3),'Fare']=9
_= sns.distplot(data.Fare)
eta_squared(data.loc['train'].Survived,data.loc['train'].Fare)
scipy.stats.skew(data.Fare.dropna())
_= sns.distplot(np.log1p(data.Fare))
eta_squared(data.loc['train'].Survived,np.log1p(data.loc['train'].Fare))
data.Fare=np.log1p(data.Fare)
data['Fare']=StandardScaler().fit_transform(data['Fare'].values.reshape(-1, 1))
correlation['Fare']=cramers_v(data.loc['train']['Fare'],data.loc['train']['Survived'])

print(correlation['Fare'])
data['FareBin']=pd.qcut(data.Fare,5, labels=False)
sns.barplot(x=data['FareBin'],y=data.Survived)
correlation['FareBin']=cramers_v(data.loc['train']['FareBin'],data.loc['train']['Survived'])

print(correlation['FareBin'])
# on regroupe par tickets, calcule le nombre d'occurences de chaque tickets et si l'occurence est > 1

# on renvoie 1 (personne ne voyageant pas seule)

VoyageSeul = (data.Ticket.groupby(data.Ticket).transform('count')==1)*1
data['IsAlone']=(((data.SibSp+data.Parch)==0)&VoyageSeul)*1
sns.barplot(x="IsAlone",y="Survived", data=data)
correlation['IsAlone']=cramers_v(data['IsAlone'],data['Survived'])

print(correlation['IsAlone'])
data['FamilySize']=data.Parch+data.SibSp+1
data['FamilySize']=data[['FamilySize', 'TicketSibling']].values.max(1)
correlation['FamilySize']=cramers_v(data['FamilySize'],data['Survived'])

print(correlation['FamilySize'])
# les données ont été enregistrées dans le fichier accompagnants.csv

accompagnants = pd.read_csv('../input/titanicdata/accompagnants.csv')
data["Servant"]=0
import unicodedata



def strip_accents(text):

    text = unicodedata.normalize('NFD', text)

    text = text.encode('ascii', 'ignore')

    text = text.decode("utf-8")

    return str(text)



# on va parcourir la liste des passagers et regarder si les nom de famille correspond, en cas de correspondance unique

# on sait qu'on a trouvé le bon passager et on lui attribue la valeur 1 en colonne Servant

# sinon on croise aussi la donnée prenom pour être sûr

for _,line in accompagnants.iterrows():

    x = line['Surname'].split(' ')[0]

    x = strip_accents(x)

    y = line['First Name'].split(' ')[0]

    y = strip_accents(y)

    result = len(data[(data['Last Name'].str.contains(x))&(data['FamilySize']==1)])

    if result==0:next

    if result==1:

        data.loc[(data['Last Name'].str.contains(x))&(data['FamilySize']==1),'Servant']=1

    if result>1:

        data.loc[(data['First Name'].str.contains(y))&(data['Last Name'].str.contains(x))&(data['FamilySize']==1),'Servant']=1

       

correlation['Servant']=cramers_v(data['Servant'],data['Survived'])

print(correlation['Servant'])
pd.crosstab(data['Servant'],data['Survived'])
# on va créer de nouvelles variables pour indiquer si la mère a survécu, le père a survécu et les enfants ont survécu,

# ou s'ils sont décédés

data['parentSurvived']=0

data['childrenSurvived']=0

data['parentDied']=0

data['childrenDied']=0

data['SiblingDied']=0

data['SiblingSurvived']=0
for x in data.Ticket.unique():

    if len(data[data.Ticket==x])<2:continue

        

    # on créer une variable childrenSurvived qui pour chaque famille correspond au nombre d'enfants ayant survécu

    if len(data.loc['train'][(data.loc['train'].Ticket==x)&(data.loc['train'].isEnfant==1)]):

        y = data.loc['train'][(data.loc['train'].Ticket==x)&(data.loc['train'].isEnfant==1)].Survived.sum()



        # si x est défini

        if y==y:

            if y==0:

                data.loc[(data.Ticket==x)&(data.isParent==1),'childrenDied'] = 1

            if y>0:

                data.loc[(data.Ticket==x)&(data.isParent==1),'childrenSurvived'] = y



                

    if len(data.loc['train'][(data.loc['train'].Ticket==x)&(data.loc['train'].isParent==1)]):

        # on créer une variable parentSurvived qui pour chaque famille correspond à la survie d'au moins un des parents

        y = data.loc['train'][(data.loc['train'].Ticket==x)&(data.loc['train'].isParent==1)].Survived.max()

        # si x est défini

        if y==y:

            if y==0:

                data.loc[(data.Ticket==x)&(data.isEnfant==1),'parentDied'] = 1

            if y>0:

                data.loc[(data.Ticket==x)&(data.isEnfant==1),'parentSurvived'] = y

# et pareil pour la survie des freres et soeurs

for x in data.Ticket.unique():

    if len(data[data.Ticket==x])<2:continue

    # il faut aussi qu'il y ait plusieurs enfants 

    if len(data[(data.Ticket==x)&(data.isEnfant==1)])<2:continue

    if len(data.loc['train'][(data.loc['train'].Ticket==x)&(data.loc['train'].isEnfant==1)])==0:continue



    y = data.loc['train'][(data.loc['train'].Ticket==x)&(data.loc['train'].isEnfant==1)].Survived.sum()



    # si y est défini

    if y==y:

        data.loc[(data.Ticket==x)&(data.isEnfant==1),'SiblingSurvived'] = y

        

    y = (data.loc['train'][(data.loc['train'].Ticket==x)&(data.loc['train'].isEnfant==1)].Survived==0).sum()



    # si y est défini

    if y==y:

        data.loc[(data.Ticket==x)&(data.isEnfant==1),'SiblingDied'] = y

cramers_v(data.loc[(data.isParent==1)&(data.Sex==0),'childrenSurvived'],data.loc[(data.isParent==1)&(data.Sex==0),'Survived'])
pd.crosstab(data.loc[(data.isParent==1)&(data.Sex==0),'childrenSurvived'],data.loc[(data.isParent==1)&(data.Sex==0),'Survived'])
cmap = sns.diverging_palette(h_neg=10,h_pos=240,as_cmap=True)

corr = data.drop(['PassengerId'],axis=1).corr(method='pearson')

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask,center=0, cmap=cmap, linewidths=1,annot=True, fmt=".2f")
dataForML=data.drop(['Name','Last Name','First Name'],axis=1)

dataForML.head(2)
MyEncoder=preprocessing.LabelEncoder()

MyEncoder.fit(data['Ticket'].sort_values())

dataForML['Ticket'] = MyEncoder.transform(dataForML['Ticket'])

MyEncoder.fit(data['Title'].sort_values())

dataForML['Title'] = MyEncoder.transform(dataForML['Title'])

dataForML.head(2)
# dataForML.to_csv('../input/titanicdata/dataForML.csv')