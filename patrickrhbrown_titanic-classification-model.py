# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
print('The mean survival rate across the whole dataset, and therefore our baseline survival only prediction model is approx: ' + str(round(train_df['Survived'].mean(),3) * 100) + '%')
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
matplotlib.style.use('fivethirtyeight')

gender_survival = train_df.groupby(['Sex']).mean()
gender_survival['Survived'].plot.bar()
gender_and_class_survival = train_df.groupby(['Pclass','Sex']).mean()
gender_and_class_survival['Survived'].plot.bar()
gender_and_class_df = pd.DataFrame(gender_and_class_survival['Survived'])
gender_and_class_df
missing_values = pd.DataFrame({'Number of Missing Values': train_df.isnull().sum()}).T
missing_values
age_groups = pd.cut(train_df["Age"], bins=12, precision=0, right = False)
group_age = train_df.groupby(age_groups).mean()
group_age['Survived'].plot.bar()
#first let's create a new column that sums Parch and SibSp to give a num of additional family members onboard

train_df['family_onboard'] = train_df['Parch'] + train_df['SibSp']
test_df['family_onboard'] = test_df['Parch'] + test_df['SibSp']
train_df.head()
#check we've not accidentally introduced any missing values

train_df['family_onboard'].isna().sum()
#next let's establish some further value from the new column by binning the values so that a) we create a categorical column and b) we begin to standardise some family sizes
#we need to do the same for our training and test sets as before.  We'll have to include an upper bin limit to the right to cope with any particularly large families (hence '50' as the rightward bin limit)

train_df['family_onboard'] = pd.cut(train_df.family_onboard, bins = [0,1,2,3,4,5,50], right = False, labels = [1,2,3,4,5,6])
test_df['family_onboard'] = pd.cut(test_df.family_onboard, bins = [0,1,2,3,4,5,50], right = False, labels = [1,2,3,4,5,6])
train_df.head(10)
#okay we're good

train_df.isnull().sum()
#we plot a bar chart of survival rates for each family size category and across gender of passenger
#we see that single female and females with one other family member showing good survival rates but 
#male passengers and larger family members do not do so well

family_model = train_df.groupby(['family_onboard','Sex']).sum()
family_model['Survived'].plot.bar()
#create a helper function to get the titles from the name column where they are currently embedded in a string.  We'll use strip to 
#reduce the string to a list of strings and pick out the string we want

honorifics_train = set()
honorifics_test = set()

for n in train_df['Name']:
    honorifics_train.add(n.split(',')[1].split('.')[0].strip())

for n in test_df['Name']:
    honorifics_test.add(n.split(',')[1].split('.')[0].strip())
        
master_list = honorifics_train | honorifics_test
master_list
#we create a title map which creates keys for each title in our existing training and test sets
#and maps to each a smaller set of categories that we will use to create mappings in the next step

title_map = {
    "Capt" : "services",
    "Col" : "services",
    "Don" : "gentry",
    "Dona" : "gentry",
    "Dr" : "profession",
    "Jonkheer" : "gentry",
    "Lady" : "gentry",
    "Major" : "services",
    "Master" : "master",
    "Miss" : "miss",
    "Mlle" : 'miss',
    "Mr" : "mr",
    "Mrs": "mrs",
    "Ms" : "ms",
    "Rev" : "profession",
    "Sir" : "gentry",
    "the Countess" : "gentry"
}


#the next bit involves two steps.  First, we repeat the extraction process we ran to pull together a list of titles repeated in the training and test sets
#with a view to replacing the current names field in each observation with just the title.  Second, we will then use our dictionary to map each 
#title to our dictionary of title groups so that we can create a title category column
#given the two steps should ideally happen relatively seamlessly, we'll process them via a helper function

def bin_titles():
    train_df['Honorific'] = train_df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    test_df['Honorific'] = test_df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    
    train_df['Honorific'] = train_df.Honorific.map(title_map)
    test_df['Honorific'] = test_df.Honorific.map(title_map)
bin_titles()
train_df.head(20)
class_model = train_df.groupby(['Honorific']).mean()
class_model_survival = pd.DataFrame(class_model['Survived'])
class_model_survival
#First, we'll fill any nans with X to denote that we don't know the cabin details for that occupier.  We'll then map the train and the test sets such that we'll reduce each feature
#to the 

train_df['Cabin'].fillna('X', inplace = True)
test_df['Cabin'].fillna('X', inplace = True)
train_df['Cabin'] = train_df['Cabin'].map(lambda x: x[0])
test_df['Cabin'] = test_df['Cabin'].map(lambda x: x[0])
test_df.head(20)

# reached this point.  Need to sort age, fare columns and also rejig model preparation
train_age_groups = train_df.groupby(['Sex', 'Pclass', 'Honorific'])
pd.DataFrame(train_age_groups.Age.median())

test_age_groups = test_df.groupby(['Sex', 'Pclass', 'Honorific'])
pd.DataFrame(test_age_groups.Age.median())

train_df['Age'] = train_age_groups.Age.apply(lambda y: y.fillna(y.median()))
test_df['Age'] = test_age_groups.Age.apply(lambda x: x.fillna(x.median()))

train_df.head()
fare_survival = train_df[train_df['Survived'] == 1]['Fare'] 
fare_not_survival = train_df[train_df['Survived'] == 0]['Fare']
fare_comparison = pd.DataFrame({'Mean Fare Paid by Survivors': fare_survival.mean(), 'Median Fare Paid by Survivors': fare_survival.median(), 'Mean Fare Paid by Non-Survivors': fare_not_survival.mean(), 'Median Fare Paid by Non-Survivors': fare_not_survival.median()}, index = ['Values'])
fare_comparison
embark_point_survived = train_df[train_df['Survived'] == 1]['Embarked'].value_counts()
embark_point_not_survived = train_df[train_df['Survived'] == 0]['Embarked'].value_counts()
embark_point_combined = pd.DataFrame({'Survival by Embarcation Point': embark_point_survived, 'Non-Survival by Embarcation Point': embark_point_not_survived}, index = ['Q', 'S', 'C'])
embark_point_combined
#Let's tidy up some features that are currently framed as strings so that they can be dummified in the next steps

train_df['Pclass'] = train_df['Pclass'].astype('object')
test_df['Pclass'] = test_df['Pclass'].astype('object')
train_df['Embarked'] = train_df['Embarked'].astype('object')
test_df['Embarked'] = test_df['Embarked'].astype('object')
train_df['family_onboard'] = train_df['family_onboard'].astype('object')
test_df['family_onboard'] = test_df['family_onboard'].astype('object')
train_df['Sex'] = train_df.Sex.map({'male': 0, 'female': 1})
test_df['Sex'] = test_df.Sex.map({'male': 0, 'female': 1})
#load initial dependencies

from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#create our target

train_y = train_df.Survived

#create our predictors and drop some cols from which we've extracted derived features

train_x = train_df.drop(['Survived','Ticket','Name'], axis = 1)
test_x = test_df.drop(['Ticket','Name'], axis = 1)

#call some descriptive statistics about the data

train_x.dropna()
test_x.dropna()
test_x.isnull().sum()
train_predictors_numeric = train_x.select_dtypes(exclude = ['object'])
test_predictors_numeric = test_x.select_dtypes(exclude = ['object'])
train_predictors_categorical = train_x.select_dtypes(['object'])
test_predictors_categorical = test_x.select_dtypes(['object'])
train_x_one_hot_encoded = pd.get_dummies(train_predictors_categorical)
test_one_hot_encoded = pd.get_dummies(test_predictors_categorical)
coded_train, coded_test = train_x_one_hot_encoded.align(test_one_hot_encoded, join = 'inner', axis = 1)
train_x = train_predictors_numeric.merge(coded_train, left_index = True, right_index = True)
train_x.isnull().sum()
test_x = test_predictors_numeric.merge(coded_test, left_index = True, right_index = True)
test_x.head()
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline

#my_pipeline = make_pipeline(Imputer(axis = 1, strategy = 'median'), XGBClassifier(learning_rate = 0.02, n_estimators = 600, objective = 'binary:logistic', silent = True, nthread = 1))

my_pipeline = XGBClassifier(learning_rate = 0.02, n_estimators = 600, objective = 'binary:logistic', silent = True, nthread = 1)

#We'll perform a GridSearch to find the best hyperparameters for our model.  
# First, we need to assemble a dictionary of parameters

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

params = {
    'max_depth': [3,4,5],
    'min_child_weight': [1,5,10],
    'gamma': [0.5,1,1.5,2,5],
    'colsample_bytree': [0.6,0.8,1.0],
    'subsample': [0.6,0.8,1.0]
}
folds = 3
param_comb = 5

xgb_skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)

random_search = RandomizedSearchCV(my_pipeline, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=xgb_skf.split(train_x,train_y), verbose=3, random_state=42)

random_search.fit(train_x, train_y)
print(random_search.best_params_)
my_pipeline = make_pipeline(Imputer(axis = 1, strategy = 'median'), XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=2, learning_rate=0.02, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=600,
       n_jobs=1, nthread=1, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(my_pipeline, train_x, train_y, scoring = 'neg_mean_absolute_error')
print(scores)
print('Mean across scores: {}'.format(-1 * scores.mean()))
my_pipeline.fit(train_x, train_y)
my_predictions = my_pipeline.predict(test_x)
predicted_survival = my_pipeline.predict(test_x)
test_x['PassengerId'] = test_x['PassengerId'].astype('int64')
my_submission = pd.DataFrame({'PassengerId': test_x.PassengerId, 'Survived': predicted_survival})

my_submission.to_csv('submission.csv', index=False)