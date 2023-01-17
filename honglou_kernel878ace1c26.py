# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np

import seaborn as sas

import os

import warnings

warnings.filterwarnings('ignore')



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
test.head()
train.head()
test['Survived']=np.zeros(len(test))+5



test.head()
new_train=pd.concat([train,test],ignore_index=True)

new_train.head()
new_train.info()
# new_train=new_train.drop(columns='Cabin')

# new_train.head()
new_train.shape
test.shape,train.shape

gend=pd.read_csv('../input/gender_submission.csv')
gend.head()
sas.barplot(x='Pclass',y='Survived',data=train)







sas.barplot(x='Sex',y='Survived',data=train)
sas.barplot(x='Embarked',y='Survived',data=train)
new_train.info()
new_train[new_train['Embarked'].isnull()]
new_train.groupby(by=['Pclass','Embarked']).Fare.median()
new_train['Embarked']=new_train['Embarked'].fillna('C')
new_train.info()
new_train[new_train['Fare'].isnull()]
fare=new_train[(new_train['Embarked']=='S')&(new_train['Pclass']==3)].Fare.median()

new_train['Fare']=new_train['Fare'].fillna(fare)

new_train.head()
new_train['Embarked']=new_train['Embarked'].factorize()[0]
# new_train['Fare']=new_train['Fare'].factorize()[0]

new_train['Sex']=new_train['Sex'].factorize()[0]
new_train['Cabin']=new_train['Cabin'].fillna('Unknow')
new_train.head(10)
def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sas.diverging_palette(220, 10, as_cmap = True)

    

    _ = sas.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(train)
new_train.info()
sas.barplot(x='SibSp',y='Survived',data=train)
sas.barplot(x='Parch',y='Survived',data=train)
new_train['family_size']=new_train['SibSp']+new_train['Parch']+1

sas.barplot(x='family_size',y='Survived',data=new_train)
def Fam_label(s):

    if (s>=1 and s<=3)or s==11:

        return 2

    elif (s>=5 and s<=7):

        return 1

    elif s==8:

        return 0

    elif s==6:

        return 3

    elif s==4:

        return 4

new_train['family_label']=new_train['family_size'].apply(Fam_label)

sas.barplot(x='family_label',y='Survived',data=new_train)
new_train[new_train['family_label'].isnull()]
new_train.info()
new_train['Deck']=new_train['Cabin'].str.get(0)

sas.barplot(x='Deck',y='Survived',data=new_train)
Ticket_count=dict(new_train['Ticket'].value_counts())

new_train['TicketGroup']=new_train['Ticket'].apply(lambda x:Ticket_count[x])

new_train['TicketGroup'].head()
new_train['Ticket'].value_counts()
sas.barplot(x='TicketGroup',y='Survived',data=new_train)
def ticket_label(s):

    if (s>0 and s<=5) |s==7|s==11:

        return 0

    elif s==6| s==8:

        return 1

    else:

        return 2

new_train['TicketGroup']=new_train['TicketGroup'].apply(ticket_label)

sas.barplot(x='TicketGroup',y='Survived',data=new_train)
new_train[new_train['Name'].str.contains('')==True].Name
new_train['Title']=new_train['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())

Title_Dicket={}
Title_Dicket.update(dict.fromkeys(['Capt','Col','Major','Dr','Rev'],'officer'))

Title_Dicket.update(dict.fromkeys(['Don','Sir','the Countess','Dona','Lady'],'Royalty'))

Title_Dicket.update(dict.fromkeys(['Mme','Ms','Mrs'],'Mrs'))

Title_Dicket.update(dict.fromkeys(['Mlle','Miss'],'Miss'))

Title_Dicket.update(dict.fromkeys(['Mr'],'Mr'))

Title_Dicket.update(dict.fromkeys(['Master','Jonkheer'],'Master'))

new_train['Title']=new_train['Title'].map(Title_Dicket)
sas.barplot(x='Title',y='Survived',data=new_train)
new_train.Title
from sklearn.ensemble import RandomForestRegressor

new_train['Title']=new_train['Title'].factorize()[0]

age_df=new_train[['Age','Pclass','Sex','Title','family_size']]

agg_df=pd.get_dummies(age_df)

agg_df.head()
know_age=age_df[age_df.Age.notnull()].as_matrix()

unknow_age=age_df[age_df.Age.isnull()].as_matrix()
y=know_age[:,0]

X=know_age[:,1:]

rfr=RandomForestRegressor(random_state=0,n_estimators=100,n_jobs=-1)

rfr.fit(X,y)

pre=rfr.predict(unknow_age[:,1::])

new_train.loc[(new_train.Age.isnull()),'Age']=pre
new_train['Surname']=new_train['Name'].apply(lambda x:x.split(',')[0].strip())

Surname_count=dict(new_train['Surname'].value_counts())

new_train['Family_Grp']=new_train['Surname'].apply(lambda x:Surname_count[x])

Female_Chi_Grp=new_train.loc[(new_train['Family_Grp']>=2) & ((new_train['Age']<=12)|(new_train['Sex']==1))]

Male_Adult_Grp=new_train.loc[(new_train['Family_Grp']>=2)&((new_train['Age']>12)|(new_train['Sex']==0))]
Female_Chl=pd.DataFrame(Female_Chi_Grp.groupby('Surname')['Survived'].mean().value_counts())

Female_Chl.head()
Female_Chl.columns=['Grp_count']

Female_Chl.head()
sas.barplot(x=Female_Chl.index,y=Female_Chl['Grp_count']).set_xlabel('Avg_survied')
Male_Adult=pd.DataFrame(Male_Adult_Grp.groupby('Surname')['Survived'].mean().value_counts())

Male_Adult.columns=['Grp_count']

Male_Adult.head()
def length_name(s):

    return len(s)

new_train['len_name']=new_train['Surname'].apply(length_name)
new_train['first_Surname']=new_train['Surname'].str.get(0)

sas.barplot(x='first_Surname',y='family_size',data=new_train)
sas.barplot(x='len_name',y='Survived',data=new_train[(new_train['Survived']!=5) &(new_train['Sex']==1)])
sas.barplot(x='len_name',y='Survived',data=new_train[(new_train['Survived']!=5) &(new_train['Sex']==0)])
sas.barplot(x='len_name',y='SibSp',data=new_train)
new_train['len_Tic']=new_train['Ticket'].apply(lambda x:len(x))

sas.barplot(x='len_Tic',y='Survived',data=new_train[new_train['Survived']!=5])
alone=new_train['family_size']==1

new_train['Alone']=alone.factorize()[0]
new_train['Sex']=new_train['Sex']-1

new_train['man_len_name']=new_train['Sex']*new_train['len_name']

new_train['Sex']=new_train['Sex']+1
new_train['first_Surname']

first_surnCount=new_train['first_Surname'].value_counts()/len(new_train["first_Surname"])

first_Surname_dit=dict(first_surnCount)

new_train['first_Surname']=new_train['first_Surname'].apply(lambda x:first_Surname_dit.get(x))

new_train['first_Surname']
Female_Child_Group=Female_Chi_Grp.groupby('Surname')['Survived'].mean()

Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)

print(Dead_List)

Male_Adult_List=Male_Adult_Grp.groupby('Surname')['Survived'].mean()

Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

print(Survived_List)

train=new_train.loc[new_train['Survived']!=5]

test=new_train.loc[new_train['Survived']==5]

test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'

test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60

test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'

test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'

test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5

test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'
train.loc[(train['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'

train.loc[(train['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60

train.loc[(train['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'

train.loc[(train['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'

train.loc[(train['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5

train.loc[(train['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'
new_train.head()
all_data_2=pd.concat([train, test])

all_data_2['man_len_name']=abs(all_data_2['man_len_name'])

all_data_2.columns
all_data_1=all_data_2[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','family_label','Deck','TicketGroup','len_name','len_Tic','Alone','man_len_name','first_Surname']]

all_data=pd.get_dummies(all_data_1)

# all_data=all_data_1

# all_data['Deck']=all_data['Deck'].factorize()[0]


# =all_data['Deck'].factorize()[0]

train=all_data[all_data['Survived']!=5]

test=all_data[all_data['Survived']==5].drop('Survived',axis=1)

X = train.as_matrix()[:,1:]

y = train.as_matrix()[:,0]
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest



pipe=Pipeline([('select',SelectKBest(k=20)), 

               ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])



param_test = {'classify__n_estimators':list(range(20,50,2)), 

              'classify__max_depth':list(range(3,60,3))}

gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10)

gsearch.fit(X,y)

print(gsearch.best_params_, gsearch.best_score_)

from sklearn.pipeline import make_pipeline

select = SelectKBest(k = 20)

clf = RandomForestClassifier(random_state = 10, warm_start = True, 

                                  n_estimators = 26,

                                  max_depth = 12, 

                                  max_features = 'sqrt')

pipeline = make_pipeline(select, clf)

pipeline.fit(X, y)

from sklearn.model_selection import cross_val_score

from sklearn import  metrics

cross_val_score(pipeline,X,y,cv=10)
predictions =pipeline.predict(test)

PassengerID=gend.PassengerId

submission = pd.DataFrame({"PassengerId":PassengerID, "Survived": predictions.astype(np.int32)})

submission.to_csv(r"submission1.csv", index=False)