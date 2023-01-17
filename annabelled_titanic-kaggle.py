# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

PassengerId=test['PassengerId']

all_data = pd.concat([train, test], ignore_index = True)
train.head()
train.info()
train['Survived'].value_counts()
sns.barplot(x="Sex", y="Survived", data=train)
sns.barplot(x="Pclass", y="Survived", data=train)
sns.barplot(x="SibSp", y="Survived", data=train)
sns.barplot(x="Parch", y="Survived", data=train)
facet = sns.FacetGrid(train, hue="Survived",aspect=2)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlabel('Age') 

plt.ylabel('density') 
sns.countplot('Embarked',hue='Survived',data=train)
all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())

Title_Dict = {}

Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))

Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))

Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))

Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))

Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))

Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))



all_data['Title'] = all_data['Title'].map(Title_Dict)

sns.barplot(x="Title", y="Survived", data=all_data)
all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1

sns.barplot(x="FamilySize", y="Survived", data=all_data)
def Fam_label(s):

    if (s >= 2) & (s <= 4):

        return 2

    elif ((s > 4) & (s <= 7)) | (s == 1):

        return 1

    elif (s > 7):

        return 0

all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)

sns.barplot(x="FamilyLabel", y="Survived", data=all_data)
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')

all_data['Deck']=all_data['Cabin'].str.get(0)

sns.barplot(x="Deck", y="Survived", data=all_data)
Ticket_Count = dict(all_data['Ticket'].value_counts())

all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])

sns.barplot(x='TicketGroup', y='Survived', data=all_data)
def Ticket_Label(s):

    if (s >= 2) & (s <= 4):

        return 2

    elif ((s > 4) & (s <= 8)) | (s == 1):

        return 1

    elif (s > 8):

        return 0



all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)

sns.barplot(x='TicketGroup', y='Survived', data=all_data)
from sklearn.ensemble import RandomForestRegressor

age_df = all_data[['Age', 'Pclass','Sex','Title']]

age_df=pd.get_dummies(age_df)

known_age = age_df[age_df.Age.notnull()].iloc[:,:].values

unknown_age = age_df[age_df.Age.isnull()].iloc[:,:].values

y = known_age[:, 0]

X = known_age[:, 1:]

rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)

rfr.fit(X, y)

predictedAges = rfr.predict(unknown_age[:, 1::])

all_data.loc[ (all_data.Age.isnull()), 'Age' ] = predictedAges 
all_data[all_data['Embarked'].isnull()]
all_data.groupby(by=["Pclass","Embarked"]).Fare.median()
all_data['Embarked'] = all_data['Embarked'].fillna('C')
all_data[all_data['Fare'].isnull()]
fare=all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()

all_data['Fare']=all_data['Fare'].fillna(fare)
all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())

Surname_Count = dict(all_data['Surname'].value_counts())

all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])

Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]

Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())

Female_Child.columns=['GroupCount']

Female_Child
sns.barplot(x=Female_Child.index, y=Female_Child["GroupCount"]).set_xlabel('AverageSurvived')
Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())

Male_Adult.columns=['GroupCount']

Male_Adult
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()

Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)

print(Dead_List)

Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()

Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

print(Survived_List)
train=all_data.loc[all_data['Survived'].notnull()]

test=all_data.loc[all_data['Survived'].isnull()]

test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'

test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60

test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'

test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'

test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5

test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'
all_data=pd.concat([train, test])

all_data=all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]

all_data=pd.get_dummies(all_data)

all_data
train=all_data[all_data['Survived'].notnull()]

test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)

X = train.iloc[:,:].values[:,1:]

y = train['Survived']
#训练模型

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectKBest

select = SelectKBest(k = 20)

clf = RandomForestClassifier(random_state = 10, warm_start = True, 

                                  n_estimators = 26,

                                  max_depth = 6, 

                                  max_features = 'sqrt')

pipeline = make_pipeline(select, clf)

pipeline.fit(X, y)
#交叉验证

from sklearn import model_selection, metrics

cv_score = model_selection.cross_val_score(pipeline, X, y, cv= 10)

print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
#预测并输出

predictions = pipeline.predict(test)

submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})

submission.to_csv("submission.csv", index=False)