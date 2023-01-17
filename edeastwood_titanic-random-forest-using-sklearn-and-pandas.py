import pandas

import numpy

import re

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn import preprocessing

from sklearn import linear_model

import matplotlib.pyplot as plt

%matplotlib notebook
TRAIN_PATH = "../input/train.csv"

TEST_PATH = "../input/test.csv"

train = pandas.read_csv(TRAIN_PATH)

test = pandas.read_csv(TEST_PATH)
train.isnull().any()
test.isnull().any()
def deriveTitles(s):

    title = re.search('(?:\S )(?P<title>\w*)',s).group('title')

    if title == "Mr": return "adult"

    elif title == "Don": return "gentry"

    elif title == "Dona": return "gentry"

    elif title == "Miss": return "miss" # we don't know whether miss is an adult or a child

    elif title == "Col": return "military"

    elif title == "Rev": return "other"

    elif title == "Lady": return "gentry"

    elif title == "Master": return "child"

    elif title == "Mme": return "adult"

    elif title == "Captain": return "military"

    elif title == "Dr": return "other"

    elif title == "Mrs": return "adult"

    elif title == "Sir": return "gentry"

    elif title == "Jonkheer": return "gentry"

    elif title == "Mlle": return "miss"

    elif title == "Major": return "military"

    elif title == "Ms": return "miss"

    elif title == "the Countess": return "gentry"   

    else: return "other"

    

train['title'] = train.Name.apply(deriveTitles)

test['title'] = test.Name.apply(deriveTitles)



# and encode these new titles for later

le = preprocessing.LabelEncoder()

titles = ['adult', 'gentry', 'miss', 'military', 'other', 'child']

le.fit(titles)

train['encodedTitle'] = le.transform(train['title']).astype('int')

test['encodedTitle'] = le.transform(test['title']).astype('int')
train.Embarked.fillna(value = 'S', inplace=True)
# not expected to add significant value because cabin data is so sparse
combined = pandas.concat([train, test])

# combining train and test casts Survived from int to float because all Survived values in test are blank

combined.ParChCategories = combined.Parch > 2
combined.boxplot(column='Age', by='Pclass')
combined = combined.assign(SibSpGroup1 = combined['SibSp'] < 2)

combined = combined.assign(SibSpGroup2 = combined['SibSp'].between(2, 3, inclusive=True))

combined = combined.assign(SibSpGroup3 = combined['SibSp'] > 2)

combined = combined.assign(ParChGT2 = combined['Parch'] > 2)
age_train, age_validation = train_test_split(combined[combined.Age.notnull()], test_size = 0.2)

age_learn = combined[combined.Age.isnull()]
age_rf = RandomForestRegressor()

age_rf.fit(age_train[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']], age_train['Age'])

age_validation = age_validation.assign(rf_age = age_rf.predict(age_validation[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']]))

mean_absolute_error(age_validation['Age'], age_validation['rf_age'], sample_weight=None, multioutput='uniform_average')
age_encoder = preprocessing.OneHotEncoder().fit(combined[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']])

age_training_encoded = age_encoder.transform(age_train[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']])

age_validation_encoded = age_encoder.transform(age_validation[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']])

age_model = linear_model.RidgeCV(alphas = [0.1, 0.2, 0.3, 0.4, 0.5])

age_estimator = age_model.fit(age_training_encoded, age_train['Age'])

linear_age_predictions = age_estimator.predict(age_validation_encoded)

mean_absolute_error(age_validation['Age'], linear_age_predictions, sample_weight=None, multioutput='uniform_average')
age_learn = age_learn.assign(Age = age_rf.predict(age_learn[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']]))
age_learn.set_index('PassengerId', inplace=True, drop=False)

combined.set_index('PassengerId', inplace=True, drop=False)

combined.update(age_learn, join = 'left', overwrite = False)

# careful here... update changes int columns to floats

# https://github.com/pandas-dev/pandas/issues/4094

# this could be problematic later if they're not changed back since

# int features are treated as categorical and floats are not
combined = combined.assign(familySize = combined['Parch'] + combined['SibSp'])



def deriveChildren(age, parch):

    if(age < 18): return parch

    else: return 0



combined = combined.assign(children = combined.apply(lambda row: deriveChildren(row['Age'], row['Parch']), axis = 1))

# train['children'] = train.apply(lambda row: deriveChildren(row['Age'], row['Parch']), axis = 1)

# test['children'] = test.apply(lambda row: deriveChildren(row['Age'], row['Parch']), axis = 1)

# I think (but am not certain) the commented code above is functionally equivalent to the preceeding two lines,

# but the commented lines gave settingdwithcopy warnings. I think these were false postives but am not certain.



def deriveParents(age, parch):

    if(age > 17): return parch

    else: return 0

    

combined['parents'] = combined.apply(lambda row: deriveParents(row['Age'], row['Parch']), axis = 1)

    

def deriveResponsibleFor(children, SibSp):

    if(children > 0): return children / (SibSp + 1)

    else: return 0

    

combined['responsibleFor'] = combined.apply(lambda row: deriveResponsibleFor(row['children'], row['SibSp']), axis = 1)

    

def deriveAccompaniedBy(parents, SibSp):

    if(parents > 0): return parents / (SibSp + 1)

    else: return 0

    

combined['accompaniedBy'] = combined.apply(lambda row: deriveAccompaniedBy(row['parents'], row['SibSp']), axis = 1)

    

def unaccompaniedChild(age, parch):

    if((age < 16) & (parch == 0)): return True

    else: return False



combined['unaccompaniedChild'] = combined.apply(lambda row: unaccompaniedChild(row['Age'], row['Parch']), axis = 1)
# may not be worth doing given how sparsely populated cabin data is
# drop unused columns

combined = combined.drop(['Name', 'Cabin', 'Fare', 'Parch', 'SibSp', 'Ticket', 'title'], axis=1)

# confirm types

combined.dtypes
# label encode string features

categorical_names = {}

categorical_features = ['Embarked', 'Sex']

for feature in categorical_features:

    le = preprocessing.LabelEncoder()

    le.fit(combined[feature])

    combined[feature] = le.transform(combined[feature])

    categorical_names[feature] = le.classes_

    

#combined = combined.assign(encodedTitleInt = combined['encodedTitle'].astype(int, copy=False))

combined['title'] = combined['encodedTitle'].astype(int, copy=False)

combined['class'] = combined['Pclass'].astype(int, copy=False)

combined = combined.drop(['Pclass'], axis=1)

combined = combined.drop(['encodedTitle'], axis=1)



train = combined[combined.PassengerId < 892]

test = combined[combined.PassengerId > 891]

test = test.drop(['Survived'], axis=1)



train['Survived'] = train['Survived'].astype(int, copy=False)

# the warning below is a false positive since the copy input is set to false
test.dtypes
rf = RandomForestClassifier()

rf.fit(train[['title', 

              'Age', 

              'Embarked', 

              'class', 

              'Sex', 

              'SibSpGroup1', 

              'SibSpGroup2', 

              'SibSpGroup3', 

              'familySize', 

              'children', 

              'parents', 

              'responsibleFor', 

              'accompaniedBy', 

              'unaccompaniedChild']], train['Survived'])



test = test.assign(Survived = rf.predict(test[['title', 

              'Age', 

              'Embarked', 

              'class', 

              'Sex', 

              'SibSpGroup1', 

              'SibSpGroup2', 

              'SibSpGroup3', 

              'familySize', 

              'children', 

              'parents', 

              'responsibleFor', 

              'accompaniedBy', 

              'unaccompaniedChild']]))
test[['Survived']].to_csv(path_or_buf='~/output.csv')