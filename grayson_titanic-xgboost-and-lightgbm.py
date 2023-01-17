import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer # missing data imputing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

import os
print(os.listdir("../input"))
# read train and test data
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_train.head()
data_test.head()
# check column datatypes
data_train.info()
# check for missing values
print("Missing Values in data_train: ", data_train.isnull().sum(), sep = "\n")
print()
print("Missing Values in data_test: ", data_test.isnull().sum(), sep = "\n")
# check age distribution in both datasets
plt.subplot(1, 2, 1)
data_train.Age.hist()
plt.xlabel('Age (data_train)')
plt.ylabel('# of passengers')

plt.subplot(1, 2, 2)
data_test.Age.hist()
plt.xlabel('Age (data_test)')
plt.ylabel('# of passengers')

# Ages are continuously distributed with a single mode. Filling the missing values
# with the most frequent value is appropriate.

# fill missing values for Age with the most frequent value
imp_age = SimpleImputer(missing_values = np.nan, strategy='most_frequent')

# for data_train
imp_age.fit(data_train[['Age']])
data_train['Age'] = imp_age.transform(data_train[['Age']])

# for data_test
imp_age.fit(data_test[['Age']])
data_test['Age'] = imp_age.transform(data_test[['Age']])
# check cabin info in both datasets
print("Cabin values in data_train: ", data_train.Cabin.unique().size)
print()
print("Cabin values in data_test: ", data_test.Cabin.unique().size)

# Cabin has many discrete values. Adding a new discrete value "Unknown" to this
# category will not have significant impact.

# fill missing values for Cabin with "Unknown"
data_train.Cabin.fillna("Unknown", inplace = True)
data_test.Cabin.fillna("Unknown", inplace = True)
# check embark info in data_train
print(data_train.Embarked.value_counts())

# Only 3 categories found in this column. Missing values are filled by the
# most frequent value.
data_train.Embarked.fillna("S", inplace = True)
# check fair info in data_test
data_test.Fare.hist()
plt.xlabel('Fare (data_test)')
plt.ylabel('# of passengers')

# A large amount of passangers didn't pay their fare.
# The missing values in the Fare column are filled by 0.
data_test.Fare.fillna(0, inplace = True)
# Check the missing values again after imputing
print("Missing Values in data_train: ", data_train.isnull().sum(), sep = "\n")
print()
print("Missing Values in data_test: ", data_test.isnull().sum(), sep = "\n")
# visulize the relationship between pclass and survive
pclass_survive_crosstbl = pd.crosstab(data_train.Pclass, data_train.Survived)

# print(pclass_survive_crosstbl)

passanger_num_pclass = pclass_survive_crosstbl.sum(axis = 1)

# calculate survivor rate for each pclass
pclass_survive_crosstbl = pclass_survive_crosstbl.divide(passanger_num_pclass, axis = 0).round(2)

pclass_survive_crosstbl.plot(kind = "bar", stacked = True)
plt.xlabel('pclass (data_train)')
plt.ylabel('Survival Rate')
# extract titles for the passengers
title_train = data_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
title_test = data_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(title_train.unique())
print()
print(title_test.unique())
print()

# merge the titles by social status
title_to_replace = {'Mrs': 'Ordinary_female', 'Miss': 'Ordinary_female', 
                    'Mme': 'Ordinary_female', 'Ms': 'Ordinary_female', 
                    'Mlle': 'Ordinary_female', 'Mr': 'Ordinary_male', 
                    'Master': 'Ordinary_male', 'Capt': 'Official', 
                    'Major': 'Official', 'Dr': 'Official', 
                    'Col': 'Official', 'Rev': 'Official', 
                    'Don': 'Noble_male', 'Jonkheer': 'Noble_male', 
                    'Sir': 'Noble_male', 'Dona': 'Noble_female', 
                    'Lady': 'Noble_female', 'Countess': 'Noble_female'}
# add title column in data_train
data_train['Title'] = title_train.map(title_to_replace)
data_test['Title'] = title_test.map(title_to_replace)

# check if the Tile column matches the Name column
print(data_train[['Name', 'Title']].head())
print()
print(data_test[['Name', 'Title']].head())

# remove the Name column
data_train = data_train.drop('Name', axis = 1)
data_test = data_test.drop('Name', axis = 1)
# visulize the relationship between Title and survive
title_survive_crosstbl = pd.crosstab(data_train.Title, data_train.Survived)

# print(title_survive_crosstbl)

passanger_num_title = title_survive_crosstbl.sum(axis = 1)

# calculate survivor rate for each title
title_survive_crosstbl = title_survive_crosstbl.divide(passanger_num_title, axis = 0).round(2)

title_survive_crosstbl.plot(kind = "bar", stacked = True)
plt.xlabel('Title (data_train)')
plt.ylabel('Survival Rate')
# visulize the relationship between sex and survive
sex_survive_crosstbl = pd.crosstab(data_train.Sex, data_train.Survived)

# print(sex_survive_crosstbl)
# calculate survivor rate for each sex
sex_survive_crosstbl = sex_survive_crosstbl.divide(sex_survive_crosstbl.sum(axis = 1), axis = 0)

sex_survive_crosstbl.plot(kind = "bar", stacked = True)
plt.xlabel('Sex (data_train)')
plt.ylabel('Survival Rate')
# Segment the Age column
age_qcut_train = pd.cut(data_train.Age, [0, 20, 40, 60, 80])
age_qcut_test = pd.cut(data_test.Age, [0, 20, 40, 60, 80])

# encode the age bins for data_train
le = LabelEncoder()
le.fit(age_qcut_train)
data_train['Age_bins'] = le.transform(age_qcut_train)

# for data_test
data_test['Age_bins'] = le.transform(age_qcut_test)

 # visulize the relationship between age and survive
age_survive_crosstbl = pd.crosstab(data_train.Age_bins, data_train.Survived)
# print(age_survive_crosstbl)
# calculate survivor rate for each age
age_survive_crosstbl = age_survive_crosstbl.divide(age_survive_crosstbl.sum(axis = 1), axis = 0)
age_survive_crosstbl.plot(kind = "bar", stacked = True)
plt.xlabel('Age_bins (data_train)')
plt.ylabel('Survival Rate')
plt.xticks(np.arange(0, 4), ('0~20', '20~40', '40~60', '60~80'))
# create new family size column by combining SibSp and Parch column
data_train['famsz'] = data_train.SibSp + data_train.Parch + 1
data_test['famsz'] = data_test.SibSp + data_test.Parch + 1

# visulize the relationship between famsz and survive
famsz_survive_crosstbl = pd.crosstab(data_train.famsz, data_train.Survived)

# print(famsz_survive_crosstbl)
# calculate survivor rate for each famsz
famsz_survive_crosstbl = famsz_survive_crosstbl.divide(famsz_survive_crosstbl.sum(axis = 1), axis = 0)

famsz_survive_crosstbl.plot(kind = "bar", stacked = True)
plt.xlabel('Family Size (data_train)')
plt.ylabel('Survival Rate')
# calculate fare/person
data_train['FarePP'] = data_train['Fare'] / data_train['famsz']
data_test['FarePP'] = data_test['Fare'] / data_test['famsz']

# Segment the FarePP column
farepp_qcut_train = pd.cut(data_train.FarePP, [0, 5, 10, 20, 30, 600], include_lowest = True)
farepp_qcut_test = pd.cut(data_test.FarePP, [0, 5, 10, 20, 30, 600], include_lowest = True)

farepp_qcut_train.value_counts()
# encode the farepp bins for data_train
le = LabelEncoder()
le.fit(farepp_qcut_train)
data_train['FarePP_bins'] = le.transform(farepp_qcut_train)

# for data_test
data_test['FarePP_bins'] = le.transform(farepp_qcut_test)
# visulize the relationship between farepp and survive
farepp_survive_crosstbl = pd.crosstab(data_train.FarePP_bins, data_train.Survived)

# print(farepp_survive_crosstbl)
# calculate survivor rate for each farepp
farepp_survive_crosstbl = farepp_survive_crosstbl.divide(farepp_survive_crosstbl.sum(axis = 1), axis = 0)

farepp_survive_crosstbl.plot(kind = "bar", stacked = True)
plt.xlabel('FarePP_bins (data_train)')
plt.ylabel('Survival Rate')
plt.xticks(np.arange(0, 5), ('0~5', '5~10', '10~20', '20~30', '30~600'))
# Extract Deck info from the Cabin column
data_train['Deck'] = data_train.Cabin.str.slice(0,1)
data_test['Deck'] = data_test.Cabin.str.slice(0,1)

# visulize the relationship between deck and survive
deck_survive_crosstbl = pd.crosstab(data_train.Deck, data_train.Survived)

# calculate survivor rate for each deck
deck_survive_crosstbl = deck_survive_crosstbl.divide(deck_survive_crosstbl.sum(axis = 1), axis = 0)

deck_survive_crosstbl.plot(kind = "bar", stacked = True)
plt.xlabel('Deck (data_train)')
plt.ylabel('Survival Rate')
# visulize the relationship between embark and survive
embark_survive_crosstbl = pd.crosstab(data_train.Embarked, data_train.Survived)

# calculate survivor rate for each embark
embark_survive_crosstbl = embark_survive_crosstbl.divide(embark_survive_crosstbl.sum(axis = 1), axis = 0)

embark_survive_crosstbl.plot(kind = "bar", stacked = True)
plt.xlabel('Embark (data_train)')
plt.ylabel('Survival Rate')
# drop non-informative or redundant columns
PassengerId_train = data_train.PassengerId
PassengerId_test = data_test.PassengerId

data_train = data_train.drop(['PassengerId', 'Age', 'Ticket', 'Fare', 'Cabin', 'FarePP'], axis = 1)
data_test = data_test.drop(['PassengerId', 'Age', 'Ticket', 'Fare', 'Cabin', 'FarePP'], axis = 1)
# check processed data (train)
data_train.head()
# check processed data (test)
data_test.head()
# one hot encoding for Pclass
Pclass_one_hot_train = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
Pclass_one_hot_test = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

# one hot encoding for Sex
Sex_one_hot_train = pd.get_dummies(data_train['Sex'], prefix='Sex')
Sex_one_hot_test = pd.get_dummies(data_test['Sex'], prefix='Sex')

# one hot encoding for SibSp
SibSp_one_hot_train = pd.get_dummies(data_train['SibSp'], prefix='SibSp')
SibSp_one_hot_test = pd.get_dummies(data_test['SibSp'], prefix='SibSp')

# one hot encoding for Parch
Parch_one_hot_train = pd.get_dummies(data_train['Parch'], prefix='Parch')
Parch_one_hot_test = pd.get_dummies(data_test['Parch'], prefix='Parch')

# one hot encoding for Embarked
Embarked_one_hot_train = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
Embarked_one_hot_test = pd.get_dummies(data_test['Embarked'], prefix='Embarked')

# one hot encoding for Title
Title_one_hot_train = pd.get_dummies(data_train['Title'], prefix='Title')
Title_one_hot_test = pd.get_dummies(data_test['Title'], prefix='Title')

# one hot encoding for Age_bins
Age_bins_one_hot_train = pd.get_dummies(data_train['Age_bins'], prefix='Age_bins')
Age_bins_one_hot_test = pd.get_dummies(data_test['Age_bins'], prefix='Age_bins')

# one hot encoding for famsz
famsz_one_hot_train = pd.get_dummies(data_train['famsz'], prefix='famsz')
famsz_one_hot_test = pd.get_dummies(data_test['famsz'], prefix='famsz')

# one hot encoding for FarePP_bins
FarePP_bins_one_hot_train = pd.get_dummies(data_train['FarePP_bins'], prefix='FarePP_bins')
FarePP_bins_one_hot_test = pd.get_dummies(data_test['FarePP_bins'], prefix='FarePP_bins')

# one hot encoding for Deck
Deck_one_hot_train = pd.get_dummies(data_train['Deck'], prefix='Deck')
Deck_one_hot_test = pd.get_dummies(data_test['Deck'], prefix='Deck')

# join the data frames
# for data_train
one_hot_train = pd.concat([Pclass_one_hot_train, Sex_one_hot_train, SibSp_one_hot_train,
                           Parch_one_hot_train, Embarked_one_hot_train, Title_one_hot_train,
                           Age_bins_one_hot_train, famsz_one_hot_train, FarePP_bins_one_hot_train,
                          Deck_one_hot_train, data_train.Survived], axis = 1, sort = False)

# for data_test
one_hot_test = pd.concat([Pclass_one_hot_test, Sex_one_hot_test, SibSp_one_hot_test,
                           Parch_one_hot_test, Embarked_one_hot_test, Title_one_hot_test,
                           Age_bins_one_hot_test, famsz_one_hot_test, FarePP_bins_one_hot_test,
                          Deck_one_hot_test], axis = 1, sort = False)
# check if the columns are same between one_hot_train and one_hot_test
set(list(one_hot_train)) ^ set(list(one_hot_test))
# check the one hot encoded data (train)
# no Deck_T  & Title_Noble_male in data_test
one_hot_train = one_hot_train.drop('Deck_T', axis = 1)
one_hot_train = one_hot_train.drop('Title_Noble_male', axis = 1)

print(one_hot_train.shape)
one_hot_train.head()
# check the one hot encoded data (test)
# no Parch_9 in the data_train
one_hot_test = one_hot_test.drop('Parch_9', axis = 1)

print(one_hot_test.shape)
one_hot_test.head()
# split data_train
y = one_hot_train.Survived
X = one_hot_train.drop('Survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 12)
# xgboost parameter tuning (RandomizedSearchCV)

params_xgb = {'min_child_weight': range(5,10),
           'gamma': [i/10.0 for i in range(0,10, 2)],
           'subsample': [i/10.0 for i in range(5, 10)],
           'colsample_bytree': [i/10.0 for i in range(5, 10)],
           'max_depth': range(5,10),
           'n_estimators': [400, 600, 1000, 1500],
           'learning_rate': [0.1, 0.01, 0.001]}

my_xgb = xgb.XGBClassifier(silent = 1, nthread = 1)

# split train for parameter tuning
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 12)

# tuning with random search
random_search_xgb = RandomizedSearchCV(my_xgb, param_distributions = params_xgb,
                                   n_iter = 10,
                                   scoring = 'accuracy', n_jobs = 4,
                                   cv = skf.split(X_train,y_train),
                                   verbose = False, random_state = 12)

# predict with tuned xgboost
random_search_xgb.fit(X_train, y_train)
predictions_xgb = random_search_xgb.predict(X_test)

# accuracy check
accuracy_xgb = accuracy_score(y_test, predictions_xgb)

precision_xgb = precision_score(y_test, predictions_xgb)

print('Accuracy (xgb): ', accuracy_xgb)
print('Precision (xgb): ', precision_xgb)
print('Best Parameters (xgb): ', random_search_xgb.best_params_)
# lgbm parameter tuning (RandomizedSearchCV)

params_lgbm = {'max_depth' : range(5,10),
               'num_leaves': [2**i for i in range(5, 10)],
               'max_bin': [100, 300, 500],
               'subsample_for_bin': [50, 100, 200],
               'min_child_weight': range(5,10),
               'min_child_samples': range(5,10),
               'min_split_gain': [i/10.0 for i in range(0, 10)],
               'colsample_bytree': [i/10.0 for i in range(5, 10)],
               'n_estimators': [400, 600, 1000, 1500],
               'subsample': [i/10.0 for i in range(5, 10)]}

my_lgbm = lgb.LGBMClassifier(boosting_type = 'dart', objective = 'binary', nthread = 1,
                             learning_rate = 0.01, scale_pos_weight = 1.1,
                             num_class = 1, metric = 'accuracy')

random_search_lgbm = RandomizedSearchCV(my_lgbm, param_distributions = params_lgbm,
                                   n_iter = 10,
                                   scoring = 'accuracy', n_jobs = 4,
                                   cv = skf.split(X_train,y_train),
                                   verbose = False, random_state = 12)

# predict with tuned lgbm
random_search_lgbm.fit(X_train, y_train)
predictions_lgbm = random_search_lgbm.predict(X_test)

# classification_report(y_test, predictions)
accuracy_lgbm = accuracy_score(y_test, predictions_lgbm)
precision_lgbm = precision_score(y_test, predictions_lgbm)

print('Accuracy: ', accuracy_lgbm)
print('Precision: ', precision_lgbm)
print('Best Parameters (lgbm): ', random_search_lgbm.best_params_)
# make prediction for data_test
predictions_xgb_test = random_search_xgb.predict(one_hot_test)

predictions_xgb_test_df = pd.DataFrame({'PassengerId': PassengerId_test,
                                         'Survived': predictions_xgb_test})

# write the results
predictions_xgb_test_df.to_csv('titanic_submission.csv', sep=',', index = False)