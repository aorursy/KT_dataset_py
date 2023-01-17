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
train_data=pd.read_csv("../input/titanic/train.csv")

test_data=pd.read_csv("../input/titanic/test.csv")

all_data=pd.concat([train_data,test_data],ignore_index=True,sort=False)
#import pandas_profiling as pdp

#pdp.ProfileReport(train)
test_data['Survived'] = np.nan

all_data.info()
all_data.head()
#ランダムフォレストでageを推定

from sklearn.ensemble import RandomForestRegressor



age_data=all_data[["Age","Pclass","Sex","Parch","SibSp"]]



age_data=pd.get_dummies(age_data)

age_data.head()
know_age=age_data[age_data.Age.notnull()].values

unknow_age=age_data[age_data.Age.isnull()].values
X=know_age[:,1:]

y=know_age[:,0]

rfr=RandomForestRegressor(random_state=0,n_estimators=100,n_jobs=-1)

rfr.fit(X,y)
predictedAges=rfr.predict(unknow_age[:,1::])

all_data.loc[(all_data.Age.isnull()),"Age"]=predictedAges
import seaborn as sns

import matplotlib.pyplot as plt

facet=sns.FacetGrid(all_data[0:890],hue="Survived",aspect=2)

facet.map(sns.kdeplot,"Age",shade=True)

facet.set(xlim=(0,all_data.loc[0:890,"Age"].max()))

facet.add_legend()

plt.show()
#lambda=無名関数

all_data["Title"]=all_data["Name"].map(lambda x:x.split(", ")[1].split(". ")[0])

all_data["Title"].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'],"Officer",inplace=True)

all_data['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)

all_data['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)

all_data['Title'].replace(['Mlle'], 'Miss', inplace=True)

all_data['Title'].replace(['Jonkheer'], 'Master', inplace=True)

sns.barplot(x='Title', y='Survived', data=all_data)
all_data["Surname"]=all_data["Name"].map(lambda x:x.split(",")[0].strip()) #苗字

all_data["Family_group"]=all_data["Surname"].map(all_data["Surname"].value_counts()) #出現回数カウント

print(all_data["Surname"],all_data["Family_group"])
#1０才以下の子供、または女性の生存率グループ化

Female_child_group=all_data.loc[(all_data["Family_group"]>=2)&((all_data["Age"]<=16)|(all_data["Sex"]=="female"))]

Female_child_list=Female_child_group.groupby('Surname')['Survived'].mean()

print(Female_child_list.value_counts())
#10才越えかつ男性の生存率

Male_adult_group=all_data.loc[(all_data["Family_group"]>=2)&(all_data["Age"]>16)&(all_data["Sex"]=="male")]

Male_adult_list=Male_adult_group.groupby("Surname")["Survived"].mean()

print(Male_adult_list.value_counts())
Dead_list=set(Female_child_list[Female_child_list.apply(lambda x:x==0)].index)

Survived_list=set(Male_adult_list[Male_adult_list.apply(lambda x:x==1)].index)



print(Dead_list,"\n")

print(Survived_list)
all_data.loc[(all_data['Survived'].isnull()) & (all_data['Surname'].apply(lambda x:x in Dead_list)),['Sex','Age','Title']] = ['male',28.0,'Mr']

all_data.loc[(all_data["Survived"].isnull())&(all_data["Surname"].apply(lambda x:x in Survived_list)), ["Sex","Age","Title"]] = ["female",5.0,"Mrs"]
fare=all_data.loc[(all_data["Pclass"]==3)&(all_data["Embarked"]=="S"),"Fare"].median()

all_data["Fare"]=all_data["Fare"].fillna(fare)
all_data.info()
all_data["Family"]=all_data["SibSp"]+all_data["Parch"]+1



sns.barplot(x='Family', y='Survived', data=all_data)
all_data.loc[(all_data["Family"]>=2)&(all_data["Family"]<=4),"Family_label"]=2

all_data.loc[(all_data["Family"]==1)|(all_data["Family"]>=5)&(all_data["Family"]<=7),"Family_label"]=1

all_data.loc[(all_data["Family"]>=8),"Family_label"]=0
Ticket_count=dict(all_data["Ticket"].value_counts())

all_data["Ticket_group"]=all_data["Ticket"].map(Ticket_count)

sns.barplot(x='Ticket_group', y='Survived', data=all_data)

plt.show()
all_data.info()
all_data.loc[(all_data["Ticket_group"]>=2)&(all_data["Ticket_group"]<=4),"Ticket_label"]=2

all_data.loc[(all_data["Ticket_group"]>=5)&(all_data["Ticket_group"]<=8)|(all_data["Ticket_group"]==1),"Ticket_label"]=1

all_data.loc[(all_data["Ticket_group"]>=11),"Ticket_label"]=0

sns.barplot(x='Ticket_label', y='Survived', data=all_data)

plt.show()
all_data["Cabin"]=all_data["Cabin"].fillna("U")

all_data["Cabin_first"]=all_data["Cabin"].str.get(0)

sns.barplot(x='Cabin_first', y='Survived', data=all_data)

plt.show()
all_data["Embarked"]=all_data["Embarked"].fillna("S")
all_data['Fareround'] = pd.qcut(all_data['Fare'], 4)

all_data[['Fareround', 'Survived']].groupby(['Fareround'], as_index=False).mean().sort_values(by='Fareround', ascending=True)
all_data.info()
all_data.loc[(all_data['Fare']<=7.19), 'Fare'] = 0

all_data.loc[(all_data['Fare']>=7.18) & (all_data['Fare']<=14.454),'Fare'] = 1

all_data.loc[(all_data['Fare']>14.454) & (all_data['Fare']<=31.0),'Fare'] = 2

all_data.loc[(all_data['Fare']>31.0),'Fare'] = 3



all_data = all_data.drop(['Fareround'], axis=1)
all_data=all_data[["Survived","Pclass","Age","Sex","Fare","Embarked","Title","Family_label","Ticket_label","Cabin_first"]]

all_data=pd.get_dummies(all_data)



train=all_data[all_data["Survived"].notnull()]

test=all_data[all_data["Survived"].isnull()].drop("Survived",axis=1)

train.info()
X=train.values[:,1:]

y=train.values[:,0]

test_x=test.values
from sklearn.feature_selection import SelectKBest

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_validate



select = SelectKBest(k=20) #特徴量絞り込む　25→20



clt=RandomForestClassifier(random_state=10, warm_start=True, n_estimators=26,max_depth=6,max_features="sqrt")

pipeline = make_pipeline(select,clt)

pipeline.fit(X,y)
cv_result=cross_validate(pipeline,X,y,cv=10)

print('mean_score = ', np.mean(cv_result['test_score']))

print('mean_std = ', np.std(cv_result['test_score']))
mask=select.get_support()



list_col=list(all_data.columns[1:])



for i,j in enumerate(list_col):

    print('No'+str(i+1), j,'=',  mask[i])



X_selected = select.transform(X)

print('X.shape={}, X_selected.shape={}'.format(X.shape, X_selected.shape))

test.head()
test_data.info()
PassengerId=test_data['PassengerId']

predictions = pipeline.predict(test_x)

submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})

submission.to_csv("my_submission.csv", index=False)