import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
df=pd.read_csv('../input/train.csv')
df.head()
df.isnull().any()
df=df.dropna()

df.head()
df.shape
outcome=df[['Survived']].copy()
outcome
features=['Pclass','Age','Sex','Cabin','Embarked']
x=df[features].copy()

x.head()

del x['Cabin']
x.head()

def replace(x):

    Sex=x['Sex']

    if Sex in ['female']:

        return 0

    else:

        return 1

x['Sex']=x.apply(replace,axis=1)
x.head()

def replace_1(x):

    Embarked=x['Embarked']

    if Embarked in ['E']:

        return 0

    elif Embarked in ['C']:

        return 1

    else:

        return 2

x['Embarked']=x.apply(replace_1,axis=1)
result = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

result.fit(x,outcome)
#predict=result.predict()

test=pd.read_csv('../input/test.csv')
#Outcomes=result.predict(test)

test.head()


def replace(x):

    Sex=x['Sex']

    if Sex in ['female']:

        return 0

    else:

        return 1

test['Sex']=test.apply(replace,axis=1)


def replace_1(x):

    Embarked=x['Embarked']

    if Embarked in ['E']:

        return 0

    elif Embarked in ['C']:

        return 1

    else:

        return 2

test['Embarked']=test.apply(replace_1,axis=1)
test.dropna()

test.head()

#outcomes=result.predict(test)


features=['Pclass','Sex','Age','Embarked']

test=test[features]

test.isnull().any()
test=test.dropna()
test=test.dropna()

#del test['PassengerId']

test.head()
outcomes=result.predict(test)

outcomes
outcome.shape
df.shape
d=df[['PassengerId']].copy()
d.shape
d['outcomes']=outcome
d.head()