# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')
import seaborn as sns

sns.set(style='white',context='notebook',palette='muted')

import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()
print('the shape of train data:', train.shape)

print('the shape of test data:', test.shape)
full = train.append(test, ignore_index=True)

full.describe()
full.info()
sns.barplot(data=train,x='Embarked',y='Survived')
print('Survive_rate of Embarked in S is %.2f'%full['Survived'][full['Embarked']=='S'].value_counts(normalize=True)[1])

print('Survive_rate of Embarked in S is %.2f'%full['Survived'][full['Embarked']=='C'].value_counts(normalize=True)[1])

print('Survive_rate of Embarked in S is %.2f'%full['Survived'][full['Embarked']=='Q'].value_counts(normalize=True)[1])
sns.factorplot('Pclass',col='Embarked',data=train,kind='count',size=3)
sns.barplot(data=train,x='Parch',y='Survived')

sns.barplot(data=train,x='SibSp',y='Survived')
sns.barplot(data=train,x='Pclass',y='Survived')
sns.barplot(data=train,x='Sex',y='Survived')
age_Facet=sns.FacetGrid(data=train,hue='Survived',aspect=3)

age_Facet.map(sns.kdeplot,'Age',shade=True)

age_Facet.set(xlim=(0,train['Age'].max()))

age_Facet.add_legend()
fare_Facet = sns.FacetGrid(data=train,hue='Survived',aspect=3)

fare_Facet.map(sns.kdeplot,'Fare',shade=True)

fare_Facet.set(xlim=(0,150))

fare_Facet.add_legend()
fare_plot = sns.distplot(full['Fare'][full['Fare'].notnull()], label='skewness:%.2f'%(full['Fare'].skew()))

fare_plot.legend(loc='best')
full['Fare'] = full['Fare'].map(lambda x: np.log(x) if x>0 else 0)
full['Cabin'] = full['Cabin'].fillna('U')

full['Cabin'].head()
full[full['Embarked'].isnull()]
full['Embarked'].value_counts()
full['Embarked'] = full['Embarked'].fillna('S')
full['Title'] = full['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

full['Title'].value_counts()
Titeldict = {}

Titeldict['Mr'] = 'Mr'

Titeldict['Mlle'] = 'Miss'

Titeldict['Miss'] = 'Miss'

Titeldict['Master'] = 'Master'

Titeldict['Dr'] = 'Dr'

Titeldict['Jonkheer'] = 'Master'

Titeldict['Mme'] = 'Mrs'

Titeldict['Mrs'] = 'Mrs'

Titeldict['Don'] = 'Royalty'

Titeldict['Sir'] = 'Royalty'

Titeldict['the Countess'] = 'Royalty'

Titeldict['Dona'] = 'Royalty'

Titeldict['Lady'] = 'Royalty'

Titeldict['Capt'] = 'Officer'

Titeldict['Col'] = 'Officer'

Titeldict['Major'] = 'Officer'

Titeldict['Rev'] = 'Officer'

full['Title'] = full['Title'].map(Titeldict)

full['Title'].value_counts()



sns.barplot(data=full,x='Title',y='Survived')
full['FamilyNum'] = full['Parch']+full['SibSp'] + 1

sns.barplot(data=full,x='FamilyNum',y='Survived')
def familysize(familyNum):

    if familyNum == 1:

        return 0 

    elif (familyNum>=2) & (familyNum<=4):

        return 1

    else:

        return 2

    

full['familySize'] = full['FamilyNum'].map(familysize)

full['familySize'].value_counts()
sns.barplot(data=full,x='familySize',y='Survived')
full['Deck'] = full['Cabin'].map(lambda x:x[0])

sns.barplot(data=full,x='Deck',y='Survived')
TickCountDict = {}

TickCountDict = full['Ticket'].value_counts()

TickCountDict.head()

full['TickCot'] = full['Ticket'].map(TickCountDict)

full['TickCot'].head()
sns.barplot(data=full,x='TickCot',y='Survived')
def TickCountGroup(num):

    if (num>=2) & (num<=4):

        return 0

    elif (num==1) | ((num>=5) & (num<=8)):

        return 1

    else:

        return 2

    

full['TickGroup'] = full['TickCot'].map(TickCountGroup)

sns.barplot(data=full,x='TickGroup',y='Survived')
full[full['Age'].isnull()].head()
Age_pre = full[['Age','Parch','Pclass','SibSp','Title','FamilyNum','TickCot']]

Age_pre = pd.get_dummies(Age_pre)

ParAge = pd.get_dummies(Age_pre['Parch'],prefix='Parch')

SibAge = pd.get_dummies(Age_pre['SibSp'],prefix='Sibsp')

PclAge = pd.get_dummies(Age_pre['Pclass'],prefix='Pclass')

Cor_age = pd.DataFrame()

Cor_age = Age_pre.corr()

Cor_age['Age'].sort_values()
Age_Pre = pd.concat([Age_pre,ParAge,SibAge,PclAge],axis=1)

Age_Pre.head()
AgeKnown = Age_Pre[Age_Pre['Age'].notnull()]

AgeUnknown = Age_Pre[Age_Pre['Age'].isnull()]
AgeKnown_x = AgeKnown.drop(['Age'],axis=1)

AgeKnown_y = AgeKnown['Age']



AgeUnknown_x = AgeUnknown.drop(['Age'],axis=1)
from sklearn.ensemble import RandomForestRegressor 

rfr = RandomForestRegressor(random_state=None,n_estimators=500,n_jobs=-1)

rfr.fit(AgeKnown_x,AgeKnown_y)



rfr.score(AgeKnown_x,AgeKnown_y)
AgeUnknown_y = rfr.predict(AgeUnknown_x)

full.loc[full['Age'].isnull(),['Age']] = AgeUnknown_y

full.info

full.head()
fullSel = full.drop(['PassengerId','Name','Cabin'],axis=1)

corr = pd.DataFrame()

corr = fullSel.corr()

corr['Survived'].sort_values(ascending=True)
plt.figure(figsize=(8,8))

sns.heatmap(fullSel[['Survived','Age','Embarked','Fare','Parch','Pclass','Sex','SibSp','Title','FamilyNum','familySize','Deck','TickCot','TickGroup']].corr(), cmap='BrBG',annot=True,linewidth=0.5)

plt.xticks(rotation=45)
fullSel=fullSel.drop(['FamilyNum','TickCot','Parch','SibSp'],axis=1)
fullSel
fullSel = pd.get_dummies(fullSel)

PclassDf = pd.get_dummies(full['Pclass'],prefix='Pclass')

TickGroupDf = pd.get_dummies(full['TickGroup'],prefix='TickGroup')

familySizeDf = pd.get_dummies(full['familySize'],prefix='familySize')



fullSel = pd.concat([fullSel,PclassDf,TickGroupDf,familySizeDf],axis=1)

fullSel
experData = fullSel[fullSel['Survived'].notnull()]

preData = fullSel[fullSel['Survived'].isnull()]



train_x = experData.drop('Survived',axis=1)

train_y = experData['Survived']



test_x = preData.drop('Survived',axis=1)



from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold



RFC = RandomForestClassifier()

param_grid = {'criterion' : ['gini', 'entropy'],

              'n_estimators' : np.arange(20,101,10),

              'max_depth': np.arange(2,16,2),

              'min_samples_split': [2,3,4,5,6,7],

              }

#RFC_rs = GridSearchCV(RFC,param_grid=param_grid,n_jobs=-1)

#RFC_rs.fit(train_x,train_y)

#RFC_rs.best_estimator_
param = {'criterion' : ['gini'],

              'n_estimators' : [60],

              'max_depth': [14],

              'min_samples_split': [7],    

        }

rfc_rs = GridSearchCV(RFC,param_grid=param,n_jobs=-1)

rfc_rs.fit(train_x,train_y)



test_y = rfc_rs.predict(test_x)

test_y = test_y.astype(int)

df = pd.DataFrame({'PassengerId':full['PassengerId'][full['Survived'].isnull()],'Survived':test_y})



df

df.to_csv('submission.csv',index=False)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

param = {'C' : [1,2,3],

        'penalty':['l1','l2']}

lr_rs = GridSearchCV(lr, param_grid=param,cv=StratifiedKFold(n_splits=10),scoring="accuracy", n_jobs= -1, verbose = 1)

lr_rs.fit(train_x,train_y)
lr_rs.best_estimator_
param={'C':[2],

      'penalty':['l2']}

lr_rs = GridSearchCV(lr, param_grid=param,cv=StratifiedKFold(n_splits=10),scoring="accuracy", n_jobs= -1, verbose = 1)

lr_rs.fit(train_x,train_y)
pre_y = lr_rs.predict(test_x)

pre_y = pre_y.astype(int)

df = pd.DataFrame({'PassengerId':full['PassengerId'][full['Survived'].isnull()],'Survived':pre_y})



df

df.to_csv('Submission_lr.csv',index=False)