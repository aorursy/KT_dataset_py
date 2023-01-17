from IPython.display import display



import re

import pandas as pd

import numpy as np

import xgboost as xgb



from sklearn import preprocessing

from sklearn import cross_validation

from sklearn.model_selection import KFold

from sklearn.feature_selection import RFECV

from sklearn.grid_search import GridSearchCV
def extract_maritial(name):

    """ extract the person's title, and bin it to Mr. Miss. and Mrs.

    assuming a Miss, Lady or Countess has more change to survive than a regular married woman."""

    

    re_maritial = r' ([A-Za-z]+\.) '   # use regular expressions to extract the persons title

    found = re.findall(re_maritial, name)[0]

    replace = [['Dr.','Sir.'],

               ['Rev.','Sir.'],

               ['Major.','Officer.'],

               ['Mlle.','Miss.'],

               ['Col.','Officer.'],

               ['Master.','Sir.'],

               ['Jonkheer.','Sir.'],

               ['Sir.','Sir.'],

               ['Don.','Sir.'],

               ['Countess.','High.'],

               ['Capt.','Officer.'],

               ['Ms.','High.'],

               ['Mme.','High.'],

               ['Dona.','High.'],

               ['Lady.','High.']]

                

    for i in range(0,len(replace)):

        if found == replace[i][0]:

            found = replace[i][1]

            break

    return found





def father(sex, age, parch):

    if sex == 'male' and age > 16 and parch > 0:

        return 1

    else:

        return 0

        

        

def mother(sex, age, parch):

    if sex == 'female' and age > 16 and parch > 0:

        return 1

    else:

        return 0

        

        

def parent(sex, age, parch):

    if mother(sex, age, parch) == 1 or father(sex, age, parch) == 1:

        return 1

    else:

        return 0

        

        

def extract_cabin_nr(cabin):

    """ Extracts the cabin number.  If there no number found, return NaN """

    if not pd.isnull(cabin):

        cabin = cabin.split(' ')[-1]    # if several cabins on ticket, take last one

        re_numb = r'[A-Z]([0-9]+)'

        try:

            number = int(re.findall(re_numb, cabin)[0])

            return number

        except:

            return np.nan

    else:

        return np.nan

    

    

def extract_cabin_letter(cabin):

    """ Extracts the cabin letter.  If there no letter found, return NaN """

    if not pd.isnull(cabin):

        cabin = cabin.split(' ')[-1]    # if several cabins on ticket, take last one

        re_char = r'([A-Z])[0-9]+'

        try:

            character = re.findall(re_char, cabin)[0]

            return character

        except:

            return np.nan

    else:

        return np.nan

        

        

def expand_sex(sex, age):

    """ this expands male/female with kid.  Cause below 14 years old, male or female is irrelevant"""

    if age < 14:

        return 'kid'

    else:

        return sex
def feat_eng(data):

    # create feature 'Title', which extracts the persons title from their name.

    data['Title'] = list(map(extract_maritial, data['Name']))



    # Extract features from cabins

    data['Cabin_char'] = list(map(extract_cabin_letter, data['Cabin']))

    data['Cabin_nr'] = list(map(extract_cabin_nr, data['Cabin']))

    data['Cabin_nr_odd'] = data.Cabin_nr.apply(lambda x: np.nan if x == np.nan else x%2)

    

    # Family features

    data['Father'] = list(map(father, data.Sex, data.Age, data.Parch))

    data['Mother'] = list(map(mother, data.Sex, data.Age, data.Parch))

    data['Parent'] = list(map(parent, data.Sex, data.Age, data.Parch))

    data['has_parents_or_kids'] = data.Parch.apply(lambda x: 1 if x > 0 else 0)

    data['FamilySize'] = data.SibSp + data.Parch

    

    # Extend the male/female feature with kid.  Cause for kids gender doesn't matter.

    data['Sex'] = list(map(expand_sex, data['Sex'], data['Age']))

    

    # Create bins for Fare and Age

    data['FareBin'] = pd.cut(data.Fare, bins=(-1000,0,8.67,16.11,32,350,1000))

    data['AgeBin'] = pd.cut(data.Age, bins=(0,15,25,60,90))



    data.head(8)

    return data
def missing(data):

    # If Age is null, we impute it with the median Age for their title.

    data.loc[(data.Age.isnull()) & (data.Title == 'Sir.'), 'Age'] = data.loc[data.Title == 'Sir.', 'Age'].median()        

    data.loc[(data.Age.isnull()) & (data.Title == 'Officer.'), 'Age'] = data.loc[data.Title == 'Officer.', 'Age'].median()

    data.loc[(data.Age.isnull()) & (data.Title == 'Miss.'), 'Age'] = data.loc[data.Title == 'Miss.', 'Age'].median()

    data.loc[(data.Age.isnull()) & (data.Title == 'High.'), 'Age'] = data.loc[data.Title == 'High.', 'Age'].median()

    data.loc[(data.Age.isnull()) & (data.Title == 'Mrs.'), 'Age'] = data.loc[data.Title == 'Mrs.', 'Age'].median()

    data.loc[(data.Age.isnull()) & (data.Title == 'Mr.'), 'Age'] = data.loc[data.Title == 'Mr.', 'Age'].median()



    # There is one row without a Fare...

    median_fare = data['Fare'].median()

    data['Fare'].fillna(value=median_fare, inplace=True)



    # ... and 2 rows without Embarked.

    mode_embarked = data['Embarked'].mode()[0]

    data['Embarked'].fillna(value=mode_embarked, inplace=True)



    # deal with the NaN's in some of our newly created columns

    data['Cabin_char'].fillna(value=-9999, inplace=True)

    data['Cabin_nr'].fillna(value=-9999, inplace=True)

    data['Cabin_nr_odd'].fillna(value=-9999, inplace=True)



    # after our feature engineering, we don't need some of the original features anymore

    data = data.drop(['Name','Cabin','Fare','Age','Ticket'], 1)



    data.head(8)

    return data
# read the training set

train = pd.read_csv('../input/train.csv')

display("Unaltered training set:")

display(train.head(8))



# feature engineering

train = feat_eng(train)

display("After feature engineering:")

display(train.head(8))



# treat missing values

train = missing(train)

display("After handling missing values:")

display(train.head(8))



# convert categorical values to numerical

train = pd.get_dummies(train, drop_first=True)

display("After handling categorical values:")

display(train.head(8))
X = np.array(train.drop(['Survived','PassengerId'], 1))

training_features = np.array(train.drop(['Survived','PassengerId'], 1).columns)

#X = preprocessing.scale(X)  --- not needed for XGboost?

y = np.array(train['Survived'])
clf = xgb.XGBClassifier()

cv = cross_validation.KFold(len(X), n_folds=20, shuffle=True, random_state=1)

scores = cross_validation.cross_val_score(clf, X, y, cv=cv, n_jobs=1, scoring='accuracy')

clf.fit(X,y)

print(scores)

print('Accuracy: %.3f stdev: %.2f' % (np.mean(np.abs(scores)), np.std(scores)))
featselect = RFECV(estimator=clf, cv=cv, scoring='accuracy')

featselect.fit(X,y)



print("features used during training: ")

print(training_features)

print("")

print("features proposed by RFECV: "),

print(training_features[featselect.support_])



# Note that for our feature Sex, which consists of male/female/kid, the classifier only needs to

# know if a person is male or not.  The classifier expects women and children to have equal

# chance of survival.  Which makes sense when we think about "Women and children first!".
selection = np.append(training_features[featselect.support_], ['Survived','PassengerId'])

train2 = train[selection]



X = np.array(train2.drop(['Survived','PassengerId'], 1))

training_features = np.array(train2.drop(['Survived','PassengerId'], 1).columns)

#X = preprocessing.scale(X)  --- not needed for XGboost?

y = np.array(train2['Survived'])



clf = xgb.XGBClassifier()

cv = cross_validation.KFold(len(X), n_folds=20, shuffle=True, random_state=1)

scores = cross_validation.cross_val_score(clf, X, y, cv=cv, n_jobs=1, scoring='accuracy')

print(scores)

print('Accuracy: %.3f stdev: %.2f' % (np.mean(np.abs(scores)), np.std(scores)))

clf.fit(X,y)
# just as an example, tuning 2 parameters.

# first try a wide range, e.g. [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# and then narrow it down.

grid = {'learning_rate':[0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.010], 

       'reg_lambda':[0, 0.01, 0.10, 0.50, 1]}



search = GridSearchCV(estimator=clf, param_grid=grid, scoring='accuracy', n_jobs=1, refit=True, cv=cv)

search.fit(X,y)



print(search.best_params_)

print(search.best_score_)
# read test set

test = pd.read_csv('../input/test.csv')



# pull the test set through our feature engineering and missing values functions

test = feat_eng(test)

test = missing(test)



# deal with categorical values

test = pd.get_dummies(test, drop_first=True)



# remove features deemed unworthy by our feature selection (RFECV)

test2 = test[training_features]

# the above line removes several features incl. PassengerId.

# So we prefer to keep our 'test' variable as it is, cause a few lines below

# we will need the passengerid feature.



X = np.array(test2)

#X = preprocessing.scale(X)

y_predict = clf.predict(X)

dfresult = pd.DataFrame(y_predict, test.PassengerId)
dfresult.columns = ['Survived']

dfresult.to_csv('predictions.csv')

print("done.")