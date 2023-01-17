# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import seaborn as sns

sns.set(style='white',context='notebook',palette='muted')

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train_data.info())
print(test_data.info())
full_data =train_data.append(test_data, ignore_index=True)

full_data.describe()
full_data.info()
plt.figure(figsize=(14, 6))

plt.title("Distribution of Survival Result")

sns.countplot(x=train_data['Survived'])

plt.show()



train_data['Survived'].value_counts()
# Pclass

plt.figure(figsize=(14, 6))

plt.title("Distribution of Passenger Class")

sns.countplot(x=train_data['Pclass'])

plt.show()



train_data['Pclass'].value_counts()
# Sex

plt.figure(figsize=(14, 6))

plt.title("Distribution of Passengers' Genders")

sns.countplot(x=train_data['Sex'])

plt.show()



train_data['Sex'].value_counts()
# Age

plt.figure(figsize=(14, 6))

plt.title("Distribution of Passengers' Ages")

sns.distplot(a=train_data['Age'], kde=False)

plt.show()



train_data['Age'].describe()
# sibsp



plt.figure(figsize=(14, 6))

plt.title("Distribution of Passengers' number of siblings onboard")

sns.countplot(x=train_data['SibSp'])

plt.show()



train_data['SibSp'].value_counts()
# parch

plt.figure(figsize=(14, 6))

plt.title("Distribution of Passengers' number of parents and children onboard")

sns.countplot(x=train_data['Parch'])

plt.show()



train_data['Parch'].value_counts()
# Embark

plt.figure(figsize=(14, 6))

plt.title("Distribution of Passengers' Embared location")

sns.countplot(x=train_data['Embarked'])

plt.show()



train_data['Embarked'].value_counts()
# Pclass



plt.figure(figsize=(14, 6))

plt.title("Passenger Class vs. Survival Outcome")

sns.barplot(x=train_data['Pclass'], y=train_data['Survived'])

plt.show()



print( train_data[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean() )
# Sex



plt.figure(figsize=(14, 6))

plt.title("Passenger Gender vs. Survival Outcome")

sns.barplot(x=train_data['Sex'], y=train_data['Survived'])

plt.show()



print( train_data[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean() )
# Age



ageFacet=sns.FacetGrid(train_data,hue='Survived',aspect=3)



ageFacet.map(sns.kdeplot,'Age',shade=True)



ageFacet.set(xlim=(0,train_data['Age'].max()))

ageFacet.add_legend()
# Parch



plt.figure(figsize=(14, 6))

plt.title("Number of Parents and Children vs. Survive Outcome")

sns.barplot(x=train_data['Parch'], y=train_data['Survived'])

plt.show()



for num in train_data['Parch'].dropna().unique():

    if (num == 4) or (num == 6):

        print(f"No info for {num} relatives")

    else:

        print(f"Survival rate if having {num} relatives: ", train_data['Survived'][train_data['Parch']==num].value_counts(normalize=True)[1])
# SibSp



plt.figure(figsize=(14, 6))

plt.title("Number of Siblings and Spouses vs. Survive Outcome")

sns.barplot(x=train_data['SibSp'], y=train_data['Survived'])

plt.show()



for num in train_data['SibSp'].dropna().unique():

    if (num == 5) or (num == 8):

        print(f"No info for {num} siblings and spouses")

    else:

        print(f"Survival rate if having {num} relatives: ", train_data['Survived'][train_data['SibSp']==num].value_counts(normalize=True)[1])
# Embarked



plt.figure(figsize=(14, 6))

plt.title("Number of Parents and Children vs. Survive Outcome")

sns.barplot(x=train_data['Embarked'], y=train_data['Survived'])

plt.show()



for location in train_data['Embarked'].dropna().unique():

    print(f"Survival rate if embarked from {location}: ", train_data['Survived'][train_data['Embarked']==location].value_counts(normalize=True)[1])
# Fare



ageFacet=sns.FacetGrid(train_data,hue='Survived',aspect=3)

ageFacet.map(sns.kdeplot,'Fare',shade=True)

ageFacet.set(xlim=(0,150))

ageFacet.add_legend()



farePlot=sns.distplot(full_data['Fare'][full_data['Fare'].notnull()],label='skewness:%.2f'%(full_data['Fare'].skew()))

farePlot.legend(loc='best')
full_data['Fare']=full_data['Fare'].map(lambda x: np.log(x) if x>0 else 0)
full_data['Cabin'] = full_data['Cabin'].fillna('Unkown')

full_data['Cabin'].head()
full_data[full_data['Embarked'].isnull()]
full_data['Embarked'].value_counts()
full_data['Embarked']=full_data['Embarked'].fillna('S')
full_data[full_data['Fare'].isnull()]
full_data['Fare'] = full_data['Fare'].fillna(full_data[(full_data['Pclass']==3)&(full_data['Embarked']=='S')&(full_data['Cabin']=='U')]['Fare'].mean()) 
full_data['Title']=full_data['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())



full_data['Title'].value_counts()
# Group similar titles together



TitleDict={}

TitleDict['Mr']='Mr'

TitleDict['Mlle']='Miss'

TitleDict['Miss']='Miss'

TitleDict['Master']='Master'

TitleDict['Jonkheer']='Master'

TitleDict['Mme']='Mrs'

TitleDict['Ms']='Mrs'

TitleDict['Mrs']='Mrs'

TitleDict['Don']='Royalty'

TitleDict['Sir']='Royalty'

TitleDict['the Countess']='Royalty'

TitleDict['Dona']='Royalty'

TitleDict['Lady']='Royalty'

TitleDict['Capt']='Officer'

TitleDict['Col']='Officer'

TitleDict['Major']='Officer'

TitleDict['Dr']='Officer'

TitleDict['Rev']='Officer'



full_data['Title']=full_data['Title'].map(TitleDict)

full_data['Title'].value_counts()
sns.barplot(data=full_data,x='Title',y='Survived')
full_data['Family'] = full_data['SibSp'] + full_data['Parch'] + 1



sns.barplot(data=full_data,x='Family',y='Survived')
def familysize(familyNum):

    if familyNum == 1:

        return 0

    elif (familyNum>=2) & (familyNum<=4):

        return 1

    else:

        return 2



full_data['familySize'] = full_data['Family'].map(familysize)

sns.barplot(data=full_data,x='familySize',y='Survived')

full_data['familySize'].value_counts()
full_data['Deck'] = full_data['Cabin'].map(lambda x:x[0])



sns.barplot(data=full_data,x='Deck',y='Survived')
TickCountDict = {}

TickCountDict = full_data['Ticket'].value_counts()

TickCountDict.head()
full_data['TickGroup'] = full_data['Ticket'].map(TickCountDict)

full_data['TickGroup'].head()
sns.barplot(data=full_data,x='TickGroup',y='Survived')
def TickCountGroup(num):

    if (num>=2) & (num<=4):

        return 1

    elif (num==1)|((num>=5)&(num<=8)):

        return 0

    else :

        return 2



full_data['TickCate']=full_data['TickGroup'].map(TickCountGroup)



sns.barplot(data=full_data,x='TickCate',y='Survived')
AgePre = full_data[['Age','Parch','Pclass','SibSp','Title','familySize','TickGroup']]



AgePre = pd.get_dummies(AgePre)

ParAge = pd.get_dummies(AgePre['Parch'],prefix='Parch')

SibAge = pd.get_dummies(AgePre['SibSp'],prefix='SibSp')

PclAge = pd.get_dummies(AgePre['Pclass'],prefix='Pclass')



AgeCorrDf = pd.DataFrame()

AgeCorrDf = AgePre.corr()

AgeCorrDf['Age'].sort_values()
AgePre = pd.concat([AgePre,ParAge,SibAge,PclAge],axis=1)

AgePre.head()
Age_train = AgePre[AgePre['Age'].notnull()]

Age_test = AgePre[AgePre['Age'].isnull()]



AgeKnown_X = Age_train.drop(['Age'],axis=1)

AgeKnown_Y = Age_train['Age']
Age_X_test = Age_test.drop(['Age'],axis=1)
from sklearn.ensemble import RandomForestRegressor



rfr=RandomForestRegressor(random_state=None,n_estimators=500,n_jobs=-1)

rfr.fit(AgeKnown_X,AgeKnown_Y)



print(rfr.score(AgeKnown_X,AgeKnown_Y))
AgeUnKnown_Y = rfr.predict(Age_X_test)



full_data.loc[full_data['Age'].isnull(),['Age']] = AgeUnKnown_Y

full_data.info()
fullSel = full_data.drop(['Cabin','Name','Ticket','PassengerId'],axis=1)



corrDf = pd.DataFrame()

corrDf = fullSel.corr()

corrDf['Survived'].sort_values(ascending=True)
plt.figure(figsize=(8,8))

sns.heatmap(fullSel[['Survived','Age','Embarked','Fare','Parch','Pclass',

                    'Sex','SibSp','Title','Family','familySize','Deck',

                     'TickGroup','TickCate']].corr(),cmap='BrBG',annot=True,

           linewidths=.5)

plt.xticks(rotation=45)
fullSel = fullSel.drop(['Family','SibSp','TickGroup','Parch'],axis=1)



fullSel = pd.get_dummies(fullSel)

PclassDf = pd.get_dummies(full_data['Pclass'],prefix='Pclass')

TickGroupDf = pd.get_dummies(full_data['TickGroup'],prefix='TickGroup')

familySizeDf = pd.get_dummies(full_data['familySize'],prefix='familySize')



fullSel=pd.concat([fullSel,PclassDf,TickGroupDf,familySizeDf],axis=1)
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold



kfold = StratifiedKFold(n_splits=10)



experData = fullSel[fullSel['Survived'].notnull()]

preData = fullSel[fullSel['Survived'].isnull()]



experData_X = experData.drop('Survived',axis=1)

experData_y = experData['Survived']

preData_X = preData.drop('Survived',axis=1)



modelLR = LogisticRegression()

LR_param_grid = {'C' : [1,2,3],

                'penalty':['l1','l2']}

modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold, 

                                     scoring="accuracy", n_jobs= -1, verbose = 1)

modelgsLR.fit(experData_X,experData_y)
print("Accuracy: ", modelgsLR.best_score_)
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }

modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, 

                                     scoring="accuracy", n_jobs= -1, verbose = 1)

modelgsGBC.fit(experData_X,experData_y)
print('Accuracy: ', modelgsGBC.best_score_)
GBCpreData_y=modelgsGBC.predict(preData_X)

GBCpreData_y=GBCpreData_y.astype(int)



GBCpreResultDf=pd.DataFrame()

GBCpreResultDf['PassengerId']=full_data['PassengerId'][full_data['Survived'].isnull()]

GBCpreResultDf['Survived']=GBCpreData_y

GBCpreResultDf



GBCpreResultDf.to_csv('submission.csv',index=False)