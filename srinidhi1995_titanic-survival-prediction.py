import matplotlib

from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')

%matplotlib inline

import seaborn as sns



import numpy as np

import pandas as pd

pd.options.display.max_columns = 100

pd.options.display.max_rows = 100



from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import preprocessing

from sklearn.cross_validation import cross_val_score
data=pd.read_csv('../input/train.csv')

data.head()
data.describe()
data['Age'].fillna(data['Age'].median(),inplace=True)
survived= data[data['Survived']==1]['Sex'].value_counts()

dead=data[data['Survived']==0]['Sex'].value_counts()

print(survived)

print(dead)

df= pd.DataFrame([survived,dead])

df.index =['Survived','Dead']

df.plot(kind='bar',figsize=(15,8))
afigure = plt.figure(figsize=(15,8))

plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], color = ['g','r'],bins = 10,label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
fare_figure = plt.figure(figsize=(15,8))

plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], color = ['b','y'],bins = 10,label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='green',s=40, alpha=0.4)

ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40,  alpha=0.4)

ax.set_xlabel('Age')

ax.set_ylabel('Fare')

ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=20,)
plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.set_ylabel('Survived')

ax.set_xlabel('Pclass')

ax.hist([data[data['Survived']==1]['Pclass'],data[data['Survived']==0]['Pclass']],color = ['g','r'],)
ax = plt.subplot()

ax.set_ylabel('Average fare')

data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)
survived_embark = data[data['Survived']==1]['Embarked'].value_counts()

dead_embark = data[data['Survived']==0]['Embarked'].value_counts()

df = pd.DataFrame([survived_embark,dead_embark])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(15,8))
def get_combined_data():

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

    # extracting and then removing the targets from the training data 

    targets = train.Survived

    train.drop('Survived',1,inplace=True)

    

    combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop('index',inplace=True,axis=1)

    return combined

combined = get_combined_data()
combined.describe()
combined.info()
combined.Cabin.fillna('U',inplace=True)

combined.Embarked.fillna('S',inplace=True)

combined.Fare.fillna(data.Fare.mean(),inplace=True)

combined.Age.fillna(data.Age.median(), inplace=True)
# The size of families (including the passenger)

combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

# Introducing other features based on the family size

combined['Alone'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)

combined['Small'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)

combined['Large'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.set_ylabel('Survived')

ax.set_xlabel('FamilySize')

ax.hist([data[data['Survived']==1]['FamilySize'],data[data['Survived']==0]['FamilySize']],color = ['g','r'],)
if 'Title' not in combined.columns:

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

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

    combined['Title'] = combined.Title.map(Title_Dictionary)

    combined.drop('Name',axis=1,inplace=True)

    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')

    combined.drop('Title',axis=1,inplace=True)

    combined = pd.concat([combined,titles_dummies],axis=1)
data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

data['Title'] = data.Title.map(Title_Dictionary)

data = pd.concat([data,pd.get_dummies(data['Title'],prefix='Title')],axis=1)
plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.set_ylabel('Survived')

ax.set_xlabel('Titles')

ax.hist([data[data['Survived']==1]['Title_Officer'],

         data[data['Survived']==0]['Title_Officer']

        ],color = ['g','r'],)
combined['20-40'] = combined['Age'].apply(lambda x: 1 if x>=20 and x<=40 else 0)

combined['70-80'] = combined['Age'].apply(lambda x: 1 if x>=70 and x<=80 else 0)

combined['below-80'] = combined['Fare'].apply(lambda x: 1 if x<80 else 0)
def get_one_hot_encoding(dt, features):

    for feature in features:

        if feature in dt.columns:

            dummies = pd.get_dummies(dt[feature],prefix=feature)

            dt = pd.concat([dt,dummies],axis=1)

    return dt
combined = get_one_hot_encoding(combined,['Embarked','Cabin','Pclass','Embarked','Title'])

combined['Sex'] = combined['Sex'].map({'male':0,'female':1})

combined.drop(['Embarked','Cabin','Pclass','Embarked','Title'],inplace=True,axis=1)
def cleanTicket(ticket):

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        ticket = map(lambda t : t.strip() , ticket)

        ticket = filter(lambda t : not t.isdigit(), ticket)

        ticket = list(ticket)

        if (len(ticket)) > 0:

            return ticket[0]

        else: 

            return 'XXX'



combined['Ticket'] = combined['Ticket'].map(cleanTicket)
combined = get_one_hot_encoding(combined,'Ticket')

combined.drop('Ticket',axis=1,inplace=True)
columns = combined.columns

combined_new = pd.DataFrame(preprocessing.normalize(combined, axis=0, copy=True), columns=columns)

combined_new['PassengerId'] = combined['PassengerId']

combined = combined_new
combined.head()
combined.describe()
train0 = pd.read_csv('../input/train.csv')

targets = train0.Survived

train = combined[0:891]

test = combined[891:]
clf = ExtraTreesClassifier(n_estimators=200)

clf = clf.fit(train, targets)

features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_

cols =  features.sort(['importance'],ascending=False)['feature']

model = SelectFromModel(clf, prefit=True)

train_new = model.transform(train)

test_new = model.transform(test)
cols