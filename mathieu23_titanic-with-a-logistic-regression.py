import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
print(train.info())

print("="*50)

print(test.info())
pd.crosstab(train.Pclass,train.Survived,margins=True)
pd.crosstab(train.Sex,train.Survived,margins=True)
train.Embarked.fillna('C',inplace=True)
pd.crosstab(train.Embarked,train.Survived,margins=True)
def graph_bar(category):

        

    train_survived = train[train["Survived"]==1]

    nb_train_survived=train_survived.groupby(category)['PassengerId'].nunique()



    train_not_survived = train[train["Survived"]==0]

    nb_train_not_survived=train_not_survived.groupby(category)['PassengerId'].nunique()

    

    Survived = list(nb_train_survived)

    Notsurvived = list(nb_train_not_survived)



    N = range(len(Survived))



    p1=plt.bar(N, Survived,color='#BCF0C6')

    p2=plt.bar(N, Notsurvived,bottom=Survived,color='#F98888')



    groups=(list(train[category].unique()))

    groups.sort()        

    plt.xticks(N, groups)

    plt.title("Survivors by the {} variable".format(category))

    plt.xlabel("{}".format(category))

    plt.ylabel("Number of persons")

    plt.legend((p1[0], p2[0]), ('Survived', 'Not Survived'))
plt.figure(figsize=(12,12))

plt.subplot(2, 2, 1)  

graph_bar("Embarked")

plt.subplot(2, 2, 2)

graph_bar("Sex")

plt.subplot(2, 1, 2)  

graph_bar("Pclass")

plt.show()
def name_sep(data):

    titles=[]

    for i in range(len(data)):

        title=data.iloc[i]

        title=title.split('.')[0].split(' ')[-1]

        titles.append(title)

    return titles
train['Title']=name_sep(train.Name)

test['Title']=name_sep(test.Name)
train.Title.value_counts()
train=train.replace(['female','male'],[0,1])

test=test.replace(['female','male'],[0,1])
train=pd.concat((

    train,

    pd.get_dummies(train.Embarked,drop_first=False),

),axis=1)



test=pd.concat((

    test,

    pd.get_dummies(test.Embarked,drop_first=False),

),axis=1)
train=pd.concat((

    train,

    pd.get_dummies(train.Pclass,drop_first=False),

),axis=1)



test=pd.concat((

    test,

    pd.get_dummies(test.Pclass,drop_first=False),

),axis=1)
for i in range(len(train)) :

    if train.loc[train.index[i], 'Title'] not in ["Mr","Miss","Mrs","Master"]:

        train.loc[train.index[i], 'Title']="Autre"

for i in range(len(test)) :

    if test.loc[test.index[i], 'Title'] not in ["Mr","Miss","Mrs","Master"]:

        test.loc[test.index[i], 'Title']="Autre"
pd.crosstab(train.Title,train.Survived,margins=True)
graph_bar('Title')
train=pd.concat((

    train,

    pd.get_dummies(train.Title,drop_first=False),

),axis=1)



test=pd.concat((

    test,

    pd.get_dummies(test.Title,drop_first=False),

),axis=1)
train.pivot_table(

    values='SibSp',

    index='Survived',

    columns='Pclass',

    aggfunc=np.mean)
train.pivot_table(

    values='Parch',

    index='Survived',

    columns='Pclass',

    aggfunc=np.mean)
train.pivot_table(

    values='Fare',

    index='Survived',

    columns='Pclass',

    aggfunc=np.mean)
test['Fare'].fillna(test['Fare'].mean(),inplace=True )
train.pivot_table(

    values='Age',

    index='Survived',

    columns='Pclass',

    aggfunc=np.mean)
for i in range(len(train)) :

    if train.loc[train.index[i], 'Age'] <10:

        train.loc[train.index[i], 'Age2']="[0-10["

    elif train.loc[train.index[i], 'Age'] <20:

        train.loc[train.index[i], 'Age2']="[10-20["

    elif train.loc[train.index[i], 'Age'] <30:

        train.loc[train.index[i], 'Age2']="[20-30["

    elif train.loc[train.index[i], 'Age'] <40:

        train.loc[train.index[i], 'Age2']="[30-40["

    elif train.loc[train.index[i], 'Age'] <50:

        train.loc[train.index[i], 'Age2']="[40-50["

    elif train.loc[train.index[i], 'Age'] <60:

        train.loc[train.index[i], 'Age2']="[50-60["

    else: 

        train.loc[train.index[i], 'Age2']="[60 et +"
graph_bar("Age2")
from sklearn.linear_model import LinearRegression

#We extract the persons where we know their age :

train_age=train[~train["Age"].isnull()]

#We split in two parts : the dependants variables and the independant variable.

train_x_age=train_age[["Sex","Parch","SibSp","Fare","C","Q","S","Autre","Master","Miss","Mr","Mrs"]]

train_y_age=train_age["Age"]

#This is the datasets where we will apply the age prediction 

train1=train[["Sex","Parch","SibSp","Fare","C","Q","S","Autre","Master","Miss","Mr","Mrs"]]



model = LinearRegression()

model.fit(train_x_age, train_y_age)

age_pred = model.predict(train1)



#We concat the orinal dataset with the age and the new colomn: the age predict  

df=pd.DataFrame(list(age_pred),columns=["Age3"])

train2=pd.concat((

    train1,

    df,

    train["Age"]

),axis=1)



# if we known the age then we let it, but if we don't known the age, we take the age predict. 

for i in range(len(train2)) :

    if train2.loc[train2.index[i], 'Age']>=0:

        train.loc[train2.index[i], 'Age']=train2.loc[train2.index[i], 'Age']

    else: 

        train2.loc[train2.index[i], 'Age']=train2.loc[train2.index[i], 'Age3']
#We extract the persons where we know their age :

test_age=test[~test["Age"].isnull()]

#We split in two parts : the dependants variables and the independant variable.

test_x_age=test_age[["Sex","Parch","SibSp","Fare","C","Q","S","Autre","Master","Miss","Mr","Mrs"]]

test_y_age=test_age["Age"]

#This is the datasets where we will apply the age prediction 

test1=test[["Sex","Parch","SibSp","Fare","C","Q","S","Autre","Master","Miss","Mr","Mrs"]]



model = LinearRegression()

model.fit(test_x_age, test_y_age)

age_pred = model.predict(test1)



#We concat the orinal dataset with the age and the new colomn: the age predict  

df=pd.DataFrame(list(age_pred),columns=["Age3"])

test2=pd.concat((

    test1,

    df,

    test["Age"]

),axis=1)



# if we known the age then we let it, but if we don't known the age, we take the age predict. 

for i in range(len(test2)) :

    if test2.loc[test2.index[i], 'Age']>=0:

        test2.loc[test2.index[i], 'Age']=test2.loc[test2.index[i], 'Age']

    else: 

        test2.loc[test2.index[i], 'Age']=test2.loc[test2.index[i], 'Age3']
train_X=train2[['Sex','Parch', "SibSp",'Fare', 'Q', 'S','Autre','Master','Miss','Mrs','Age']]

train_Y=train[['Survived']]

test_X=test2[['Sex','Parch',"SibSp",'Fare', 'Q', 'S','Autre','Master','Miss','Mrs','Age']]
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver="liblinear", random_state=42)

log_reg.fit(train_X, train_Y)
y_proba = log_reg.predict_proba(test_X)

survived=[]

for i in y_proba:

    if i[0]<0.5:

        survived.append(1)

    else :

        survived.append(0)

        

survived=pd.DataFrame(survived,columns=["Survived"])



titanic=pd.concat((

    test['PassengerId'],

    survived

),axis=1)
titanic.head()
titanic.to_csv('titanic.csv',index=False)