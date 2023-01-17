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
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.feature_selection import SelectKBest

#from sklearn.model_selection import cross_validation, metrics

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import warnings

warnings.filterwarnings('ignore')

 

train = pd.read_csv('../input/titanic/train.csv',dtype={"Age": np.float64})

test = pd.read_csv('../input/titanic/test.csv',dtype={"Age": np.float64})

PassengerId=test['PassengerId']

all_data = pd.concat([train, test], ignore_index = True)
train.info()
#数据可视化
sns.barplot(x='Sex',y='Survived',data=train)
sns.barplot(x='Pclass',y='Survived',data=train,palette='Set3')
facet=sns.FacetGrid(train,hue='Survived',aspect=2)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()
sns.barplot(x='SibSp',y='Survived',data=train,palette='Set3')
sns.barplot(x='Parch',y='Survived',data=train,palette='Set3')
train['Fare'].describe()
facet=sns.FacetGrid(train,hue='Survived',aspect=2)

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim=(0,200))

facet.add_legend()
train['fareclass']=train['Fare'].apply(lambda x: 1 if x>20 else 0)

train['fareclass'].value_counts()
sns.barplot(x='fareclass',y='Survived',data=train,palette='Set3')
sns.barplot(x='Embarked',y='Survived',data=train,palette='Set3')
train['hasCabin']=train['Cabin'].apply(lambda x: 1 if pd.notnull(x) else 0)

sns.barplot(x='hasCabin',y='Survived',data=train,palette='Set3')
all_data['Cabin']=all_data['Cabin'].fillna('Unknow')

all_data['Deck']=all_data['Cabin'].str.get(0)

sns.barplot(x='Deck',y='Survived',data=all_data,palette='Set3')
all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())

Title_Dict = {}

Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))

Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))

Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))

Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))

Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))

Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))

all_data['Title'] = all_data['Title'].map(Title_Dict)

sns.barplot(x="Title", y="Survived", data=all_data, palette='Set3')

# 2 创作新特征

all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1

sns.barplot(x="FamilySize", y="Survived", data=all_data, palette='Set3')
def Fam_label(s):

    if (s >= 2) & (s <= 4):

        return 2

    elif ((s > 4) & (s <= 7)) | (s == 1):

        return 1

    elif (s > 7):

        return 0

all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)

sns.barplot(x="FamilyLabel", y="Survived", data=all_data, palette='Set3')
Ticket_Count = dict(all_data['Ticket'].value_counts())

all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])

sns.barplot(x='TicketGroup', y='Survived', data=all_data, palette='Set3')
def ticket_label(s):

    if (s >= 2) & (s <= 4):

        return 2

    elif ((s > 4) & (s <= 8)) | (s == 1):

        return 1

    elif (s > 8):

        return 0

all_data['TicketLabel']=all_data['TicketGroup'].apply(ticket_label)

sns.barplot(x="TicketLabel", y="Survived", data=all_data, palette='Set3')
#确实值的填充

for col in all_data.columns:

    if all_data[col].isnull().sum()>0:

        print("{} is lack of {}".format(col,all_data[col].isnull().sum()))
from sklearn.model_selection import train_test_split

 

train = all_data[all_data['Survived'].notnull()]

test  = all_data[all_data['Survived'].isnull()]

# 分割数据，按照 训练数据:cv数据 = 1:1的比例

train_split_1, train_split_2 = train_test_split(train, test_size=0.5, random_state=0)

 

 

def predict_age_use_cross_validationg(df1,df2,dfTest):

    age_df1 = df1[['Age', 'Pclass','Sex','Title']]

    age_df1 = pd.get_dummies(age_df1)

    age_df2 = df2[['Age', 'Pclass','Sex','Title']]

    age_df2 = pd.get_dummies(age_df2)

    

    known_age = age_df1[age_df1.Age.notnull()].as_matrix()

    unknow_age_df1 = age_df1[age_df1.Age.isnull()].as_matrix()

    unknown_age = age_df2[age_df2.Age.isnull()].as_matrix()

    

    print (unknown_age.shape)

    

    y = known_age[:, 0]

    X = known_age[:, 1:]

    

    rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)

    rfr.fit(X, y)

    predictedAges = rfr.predict(unknown_age[:, 1::])

    df2.loc[ (df2.Age.isnull()), 'Age' ] = predictedAges 

    predictedAges = rfr.predict(unknow_age_df1[:,1::])

    df1.loc[(df1.Age.isnull()),'Age'] = predictedAges

    

   # rf1 ---> rf1,rf2      #  rf2+dftest---> dftest

    age_Test = dfTest[['Age', 'Pclass','Sex','Title']]

    age_Test = pd.get_dummies(age_Test)

    age_Tmp = df2[['Age', 'Pclass','Sex','Title']]

    age_Tmp = pd.get_dummies(age_Tmp)

    

    age_Tmp = pd.concat([age_Test[age_Test.Age.notnull()],age_Tmp])

    

    known_age1 = age_Tmp.as_matrix()

    unknown_age1 = age_Test[age_Test.Age.isnull()].as_matrix()

    y = known_age1[:,0]

    x = known_age1[:,1:]

 

    rfr.fit(x, y)

    predictedAges = rfr.predict(unknown_age1[:, 1:])

    dfTest.loc[ (dfTest.Age.isnull()), 'Age' ] = predictedAges 

    

    return dfTest

    

t1 = train_split_1.copy()

t2 = train_split_2.copy()

tmp1 = test.copy()

t5 = predict_age_use_cross_validationg(t1,t2,tmp1)

t1 = pd.concat([t1,t2])

 

t3 = train_split_1.copy()

t4 = train_split_2.copy()

tmp2 = test.copy()

t6 = predict_age_use_cross_validationg(t4,t3,tmp2)

t3 = pd.concat([t3,t4])

 

train['Age'] = (t1['Age'] + t3['Age'])/2

 

 

test['Age'] = (t5['Age'] + t6['Age']) / 2

all_data = pd.concat([train,test])

print (train.describe())

print (test.describe())

all_data[all_data['Embarked'].isnull()]
all_data['Embarked'].value_counts()
all_data['Embarked'] = all_data['Embarked'].fillna('C')
fare=all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()

all_data['Fare']=all_data['Fare'].fillna(fare)

fare
# 同一组的识别
all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())

Surname_Count = dict(all_data['Surname'].value_counts())

all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])

Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]

Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())

Female_Child.columns=['GroupCount']

Female_Child
Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts()
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
all_data.columns
all_data=pd.concat([train, test])

all_data=all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketLabel']]

all_data=pd.get_dummies(all_data)

train=all_data[all_data['Survived'].notnull()]

test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)

X = train.as_matrix()[:,1:]

y = train.as_matrix()[:,0]

train.shape
from sklearn.model_selection import GridSearchCV,StratifiedKFold

n_fold=StratifiedKFold(n_splits=3)

pipe=Pipeline([('select',SelectKBest(k=20)), 

               ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt',oob_score=True))])

 

param_test = {'classify__n_estimators':[30], 

              'classify__max_depth':[6],

               }

gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=n_fold,n_jobs= 10, verbose = 1)

gsearch.fit(X,y)

RFC_best=gsearch.best_estimator_

print(gsearch.best_params_, gsearch.best_score_)
RFC_best.fit(X,y)

RFC_best.score(X,y)
select = SelectKBest(k = 20)

clf = RandomForestClassifier(random_state = 10, warm_start = True, 

                                  n_estimators =30,

                                  max_depth = 6, 

                                  max_features = 'sqrt',

                           )

pipeline = make_pipeline(select, clf)

pipeline.fit(X, y)

predictions = pipeline.predict(test)

submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})

submission.to_csv("submission.csv", index=False)