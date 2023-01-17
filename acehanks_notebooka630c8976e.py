import pandas as pd

import matplotlib as plt

import numpy as np

import seaborn as sns
data = pd.read_csv('../input/train.csv')
data.head()
data.describe()
data['Age'].fillna(data['Age'].median(), inplace='TRUE')
data.describe()
headers= data.columns.tolist()

headers
sns.barplot(data= data, x='Sex', y='Survived')
survived_sex= data[data['Survived']==1]['Sex'].value_counts()

dead_sex= data[data['Survived']==0]['Sex'].value_counts()

DF1= pd.DataFrame([survived_sex, dead_sex])



DF1.index= ['Survived', 'Dead']

DF1.plot(kind='bar', stacked='TRUE')
g= sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=data, size=6, kind='bar' )

g.set_ylabels('Survival Probability')
survived_age= data[data['Survived']==1]['Age'].value_counts()

dead_age= data[data['Survived']==0]['Age'].value_counts()



sns.distplot(survived_age, kde=False, rug=True, bins=40)
sns.distplot(dead_age, kde=False, rug=True, bins=40)
sns.distplot([data[data['Survived']==1]['Fare'].value_counts()], bins = 50, kde=False)
sns.distplot([data[data['Survived']==0]['Fare'].value_counts()], bins = 50, kde=False)
survived_embark = data[data['Survived']==1]['Embarked'].value_counts()

dead_embark = data[data['Survived']==0]['Embarked'].value_counts()



df = pd.DataFrame([survived_embark,dead_embark])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(15,8))
def get_combined_data():

    train= pd.read_csv('../input/train.csv')

    test= pd.read_csv('../input/test.csv')

    

    #drop survived

    targets= train.Survived

    train.drop('Survived',inplace=True, axis=1)

    

    #combine train and test data, drop index column

    combined= train.append(test)

    combined.reset_index(inplace=True)

    combined.drop('index',inplace=True, axis=1)

    

    return combined

    
combined = get_combined_data()
combined.shape
combined.head()
def get_titles():

    global combined

    

    combined['Title']= combined['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

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

    
get_titles()
combined.head()
groupAge= combined.groupby(['Sex', 'Pclass', 'Title'])

groupAge.median()
def process_age():

    

    global combined

    

    # a function that fills the missing values of the Age variable

    

    def fillAges(row):

        if row['Sex']=='female' and row['Pclass'] == 1:

            if row['Title'] == 'Miss':

                return 30

            elif row['Title'] == 'Mrs':

                return 45

            elif row['Title'] == 'Officer':

                return 49

            elif row['Title'] == 'Royalty':

                return 39



        elif row['Sex']=='female' and row['Pclass'] == 2:

            if row['Title'] == 'Miss':

                return 20

            elif row['Title'] == 'Mrs':

                return 30



        elif row['Sex']=='female' and row['Pclass'] == 3:

            if row['Title'] == 'Miss':

                return 18

            elif row['Title'] == 'Mrs':

                return 31



        elif row['Sex']=='male' and row['Pclass'] == 1:

            if row['Title'] == 'Master':

                return 6

            elif row['Title'] == 'Mr':

                return 41.5

            elif row['Title'] == 'Officer':

                return 52

            elif row['Title'] == 'Royalty':

                return 40



        elif row['Sex']=='male' and row['Pclass'] == 2:

            if row['Title'] == 'Master':

                return 2

            elif row['Title'] == 'Mr':

                return 30

            elif row['Title'] == 'Officer':

                return 41.5



        elif row['Sex']=='male' and row['Pclass'] == 3:

            if row['Title'] == 'Master':

                return 6

            elif row['Title'] == 'Mr':

                return 26

    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)

combined.info()
def process_names():

    

    global combined

    # we clean the Name variable

    combined.drop('Name',axis=1,inplace=True)

    

    # encoding in dummy variable

    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')

    combined = pd.concat([combined,titles_dummies],axis=1)

    

    # removing the title variable

    combined.drop('Title',axis=1,inplace=True)
process_names()
combined.head()
def process_fares():

    

    global combined

    # there's one missing fare value - replacing it with the mean.

    combined.Fare.fillna(combined.Fare.median(),inplace=True)
process_fares()
def process_embarked():

    

    global combined

    # two missing embarked values - filling them with the most frequent one (S)

    combined.Embarked.fillna('S',inplace=True)

    

    # dummy encoding 

    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')

    combined = pd.concat([combined,embarked_dummies],axis=1)

    combined.drop('Embarked',axis=1,inplace=True)
process_embarked()
def process_cabin():

    

    global combined

    

    # replacing missing cabins with U (for Uknown)

    combined.Cabin.fillna('U',inplace=True)

    

    # mapping each Cabin value with the cabin letter

    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])

    

    # dummy encoding ...

    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')

    

    combined = pd.concat([combined,cabin_dummies],axis=1)

    

    combined.drop('Cabin',axis=1,inplace=True)
process_cabin()
combined.info()
combined.head()
def process_sex():

    

    global combined

    # mapping string values to numerical one 

    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})

    
process_sex()
def process_pclass():

    

    global combined

    # encoding into 3 categories:

    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")

    

    # adding dummy variables

    combined = pd.concat([combined,pclass_dummies],axis=1)

    

    # removing "Pclass"

    

    combined.drop('Pclass',axis=1,inplace=True)
process_pclass()
def process_ticket():

    

    global combined

    

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

    def cleanTicket(ticket):

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        ticket = ticket.map(lambda t : t.strip() , ticket)

        ticket = ticket.filter(lambda t : not t.isdigit(), ticket)

        if len(ticket) > 0:

            return ticket[0]

        else: 

            return 'XXX'

    



    # Extracting dummy variables from tickets:



    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')

    combined = pd.concat([combined, tickets_dummies],axis=1)

    combined.drop('Ticket',inplace=True,axis=1)
process_ticket()
def process_family():

    

    global combined

    # introducing a new feature : the size of families (including the passenger)

    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    

    # introducing other features based on the family size

    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)

    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)

    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
process_family()
combined.shape

combined.head()
def scale_all_features():

    

    global combined

    

    features = list(combined.columns)

    features.remove('PassengerId'), features.remove('Ticket')

    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)

    

    print ('Features scaled successfully !')
scale_all_features()
combined.drop('Ticket', inplace=True, axis=1)

combined.head()
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score
def compute_score(clf, X, y,scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)

    return np.mean(xval)
def recover_train_test_target():

    global combined

    

    train0 = pd.read_csv('../input/train.csv')

    

    targets = train0.Survived

    train = combined.ix[0:890]

    test = combined.ix[891:]

    

    return train,test,targets
train,test,targets = recover_train_test_target()
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=200)

clf = clf.fit(train, targets)
features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances
features.sort(['importance'],ascending=False)
model = SelectFromModel(clf, prefit=True)

train_new = model.transform(train)

train_new.shape
forest = RandomForestClassifier(max_features='sqrt')



parameter_grid = {

                 'max_depth' : [4,5,6,7,8],

                 'n_estimators': [200,210,240,250],

                 'criterion': ['gini','entropy']

                 }



cross_validation = StratifiedKFold(targets, n_folds=5)



grid_search = GridSearchCV(forest,

                           param_grid=parameter_grid,

                           cv=cross_validation)



grid_search.fit(train_new, targets)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
output = grid_search.predict(test_new).astype(int)

df_output = pd.DataFrame()

df_output['PassengerId'] = test['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('../output.csv',index=False)