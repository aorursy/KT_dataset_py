# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt



import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/titanic/train.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
def status(feature):

    print('Processing', feature, ': ok')
women = train.loc[train.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)

men = train.loc[train.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of Women who survived:", rate_women)

print("% of Men who survived:", rate_men)
class1 = train.loc[train.Pclass == 1]["Survived"]

class2 = train.loc[train.Pclass == 2]["Survived"]

class3 = train.loc[train.Pclass == 3]["Survived"]





rate_class1 = sum(class1)/len(class1)

rate_class2 = sum(class2)/len(class2)

rate_class3 = sum(class3)/len(class3)





print("% of first class passenger who survived:", rate_class1)

print("% of second class passenger who survived:", rate_class2)

print("% of third class passenger who survived:", rate_class3)
age_survived = train.loc[train['Survived']==1, "Age"]

age_survived.plot.hist(fc=(0, 0, 1, 0.5), normed=1, label="Survived")

age_did_not_survive = train.loc[train['Survived']==0, "Age"]

age_did_not_survive.plot.hist(fc=(1, 0, 0, 0.5), normed=1, label="Did not Survive")

plt.xlabel("Age")

plt.ylabel("Percentage of Passengers")

plt.legend(loc='upper right')

plt.title("Distribution of Age of Survivors and Non-Survivors")
for dataT in train:

    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1 

    

pd.crosstab(train['FamilySize'], train['Survived']).plot(kind='bar', stacked=True, title="Survived by family size")

pd.crosstab(train['FamilySize'], train['Survived'], normalize='index').plot(kind='bar', stacked=True, title="Survived by family size (%)")
female = train[train['Sex'] == 'female']

male = train[train['Sex'] == 'male']

 

# Total number

fig, [ax1, ax2] = plt.subplots(1,2, sharey=True)

fig.set_figwidth(12)

pd.crosstab(female['FamilySize'], female['Survived']).plot(kind='bar', stacked=True, title="Female", ax=ax1)

pd.crosstab(male['FamilySize'], male['Survived']).plot(kind='bar', stacked=True, title="Male", ax=ax2)



# Percentage

fig, [ax1, ax2] = plt.subplots(1,2)

fig.set_figwidth(12)

pd.crosstab(female['FamilySize'], female['Survived'], normalize = 'index').plot(kind='bar', stacked=True, title="Female", ax=ax1)

pd.crosstab(male['FamilySize'], male['Survived'], normalize = 'index').plot(kind='bar', stacked=True, title="Male", ax=ax2)
kidsmale = male[male['Age'] < 15]

adultsmale = male[male['Age'] >=15 ]



print ("Number of male kids: ")

print (kidsmale.groupby(['FamilySize']).size())

print ("")

print ("Number of male adults: ")

print (adultsmale.groupby(['FamilySize']).size())



# Size of samples

fig, [ax1, ax2] = plt.subplots(1,2)

fig.set_figwidth(12)

sns.countplot(x='FamilySize', data=kidsmale, ax=ax1)

ax1.set_title('Number of male kids')

sns.countplot(x='FamilySize', data=adultsmale, ax=ax2)

ax2.set_title('Number of male adults')



# Percentage

fig, [ax1, ax2] = plt.subplots(1,2)

fig.set_figwidth(12)

pd.crosstab(kidsmale['FamilySize'], kidsmale['Survived'], normalize = 'index').plot(kind='bar', stacked=True, title="Kids male", ax=ax1)

pd.crosstab(adultsmale['FamilySize'], adultsmale['Survived'], normalize = 'index').plot(kind='bar', stacked=True, title="Adults male", ax=ax2)



train.drop('FamilySize', axis=1)
print(train.Cabin.isnull().sum())
print(train.Embarked.isnull().sum())
print(train.Ticket.isnull().sum())
def get_combined_data():



    # extracting and then removing the targets from the training data 

    targets = train.Survived

    train.drop(['Survived'], 1, inplace=True)

    



    # merging train data and test data for future feature engineering

    # we'll also remove the PassengerID since this is not an informative feature

    combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop(['index', 'PassengerId'], inplace=True, axis=1)

    

    return combined
combined = get_combined_data()
print(combined.shape)
combined.head()
titles = set()

for name in data['Name']:

    titles.add(name.split(',')[1].split('.')[0].strip())
print(titles)
Title_Dictionary = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Sir" : "Royalty",

    "Dr": "Officer",

    "Rev": "Officer",

    "the Countess":"Royalty",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Royalty"

}



def get_titles():

    # we extract the title from each name

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated title

    # we map each title

    combined['Title'] = combined.Title.map(Title_Dictionary)

    status('Title')

    return combined
combined = get_titles()
combined.head()
combined[combined['Title'].isnull()]
print(combined.iloc[:891].Age.isnull().sum())
print(combined.iloc[891:].Age.isnull().sum())
grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])

grouped_median_train = grouped_train.median()

grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
grouped_median_train.head()
def fill_age(row):

    condition = (

        (grouped_median_train['Sex'] == row['Sex']) & 

        (grouped_median_train['Title'] == row['Title']) & 

        (grouped_median_train['Pclass'] == row['Pclass'])

    ) 

    return grouped_median_train[condition]['Age'].values[0]





def process_age():

    global combined

    # a function that fills the missing values of the Age variable

    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

    status('age')

    return combined
combined = process_age()
def process_names():

    global combined

    # we clean the Name variable

    combined.drop('Name', axis=1, inplace=True)

    

    # encoding in dummy variable

    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')

    combined = pd.concat([combined, titles_dummies], axis=1)

    

    # removing the title variable

    combined.drop('Title', axis=1, inplace=True)

    

    status('names')

    return combined
combined = process_names()
combined.head()
def process_fares():

    global combined

    # there's one missing fare value - replacing it with the mean.

    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)

    status('fare')

    return combined
combined = process_fares()
def process_embarked():

    global combined

    # two missing embarked values - filling them with the most frequent one in the train  set(S)

    combined.Embarked.fillna('S', inplace=True)

    # dummy encoding 

    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')

    combined = pd.concat([combined, embarked_dummies], axis=1)

    combined.drop('Embarked', axis=1, inplace=True)

    status('embarked')

    return combined
combined = process_embarked()
combined.head()
train_cabin, test_cabin = set(), set()



for c in combined.iloc[:891]['Cabin']:

    try:

        train_cabin.add(c[0])

    except:

        train_cabin.add('U')

        

for c in combined.iloc[891:]['Cabin']:

    try:

        test_cabin.add(c[0])

    except:

        test_cabin.add('U')
print(train_cabin)
print(test_cabin)
def process_cabin():

    global combined    

    # replacing missing cabins with U (for Uknown)

    combined.Cabin.fillna('U', inplace=True)

    

    # mapping each Cabin value with the cabin letter

    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

    

    # dummy encoding ...

    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    

    combined = pd.concat([combined, cabin_dummies], axis=1)



    combined.drop('Cabin', axis=1, inplace=True)

    status('cabin')

    return combined
combined = process_cabin()
combined.head()
print(combined.shape)
def process_sex():

    global combined

    # mapping string values to numerical one 

    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})

    status('Sex')

    return combined
combined = process_sex()
def process_pclass():

    

    global combined

    # encoding into 3 categories:

    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    

    # adding dummy variable

    combined = pd.concat([combined, pclass_dummies],axis=1)

    

    # removing "Pclass"

    combined.drop('Pclass',axis=1,inplace=True)

    

    status('Pclass')

    return combined
combined = process_pclass()
combined.drop('Ticket', inplace=True, axis=1)
def process_family():

    

    global combined

    # introducing a new feature : the size of families (including the passenger)

    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    

    # introducing other features based on the family size

    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    

    status('family')

    return combined
combined = process_family()
print(combined.shape)
combined.head()
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)

    return np.mean(xval)
def recover_train_test_target():

    global combined

    

    targets = pd.read_csv('/kaggle/input/titanic/train.csv', usecols=['Survived'])['Survived'].values

    train = combined.iloc[:891]

    test = combined.iloc[891:]

    

    return train, test, targets
train, test, targets = recover_train_test_target()
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

clf = clf.fit(train, targets)
features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(25, 25))
model = SelectFromModel(clf, prefit=True)

train_reduced = model.transform(train)

print(train_reduced.shape)
test_reduced = model.transform(test)

print(test_reduced.shape)
logreg = LogisticRegression()

logreg_cv = LogisticRegressionCV()

rf = RandomForestClassifier()

gboost = GradientBoostingClassifier()



models = [logreg, logreg_cv, rf, gboost]
for model in models:

    print('Cross-validation of : {0}'.format(model.__class__))

    score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')

    print('CV score = {0}'.format(score))

    print('****')
#model = LogisticRegression()

#model = LogisticRegressionCV()

#model = RandomForestClassifier()

#model = GradientBoostingClassifier()

model.fit(train, targets)
output = model.predict(test).astype(int)

df_output = pd.DataFrame()

aux = pd.read_csv('/kaggle/input/titanic/test.csv')

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('my_submission.csv', index=False)