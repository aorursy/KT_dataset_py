# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

X_test_full = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['Survived'], inplace=True)

y = X_full.Survived

X_full.drop(['Survived'], axis=1, inplace=True)
X_full.columns
X_full.head()
#https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if big_string.find(substring) != -1:

            return substring

    #print(big_string)

    return np.nan
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']

X_full['Title']=X_full['Name'].map(lambda x: substrings_in_string(x, title_list))

X_test_full['Title']=X_test_full['Name'].map(lambda x: substrings_in_string(x, title_list))

 

#replacing all titles with mr, mrs, miss, master

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

X_full['Title']=X_full.apply(replace_titles, axis=1)

X_test_full['Title']=X_test_full.apply(replace_titles, axis=1)
#Creating new family_size column

X_full['Family_Size']=X_full['SibSp']+X_full['Parch'] + 1

X_test_full['Family_Size']=X_test_full['SibSp']+X_test_full['Parch'] + 1



X_full['IsAlone']= [int(x) for x in (X_full['Family_Size'] > 1)]

X_test_full['IsAlone']=[int(x) for x in (X_test_full['Family_Size'] > 1)]



#Turning cabin number into Deck

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

X_full['Deck']=X_full['Cabin'].map(lambda x: substrings_in_string(str(x), cabin_list))

X_full['Age*Class']=X_full['Age']*X_full['Pclass']

X_full['Fare_Per_Person']=X_full['Fare']/(X_full['Family_Size']+1)



#Test

X_test_full['Deck']=X_test_full['Cabin'].map(lambda x: substrings_in_string(str(x), cabin_list))

X_test_full['Age*Class']=X_test_full['Age']*X_test_full['Pclass']

X_test_full['Fare_Per_Person']=X_test_full['Fare']/(X_test_full['Family_Size']+1)
# Shape of training data (num_rows, num_columns)

print(X_full.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (X_full.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# Shape of test data (num_rows, num_columns)

print(X_test_full.shape)



# Number of missing values in each column of training data

missing_val_count_by_columnTEST = (X_test_full.isnull().sum())

print(missing_val_count_by_columnTEST[missing_val_count_by_columnTEST > 0])
# Select numerical columns

numerical_cols = [cname for cname in X_full.columns if 

                X_full[cname].dtype in ['int64', 'float64']]



numerical_cols
X_train_numeric = X_full[numerical_cols]

X_test_numeric = X_test_full[numerical_cols]
cols_with_missing = (X_train_numeric.isnull().sum())

cols_with_missing = cols_with_missing[cols_with_missing > 0]

cols_with_missing = cols_with_missing.index

cols_with_missing
from sklearn.impute import SimpleImputer

# Make copy to avoid changing original data (when imputing)

X_train_plus = X_train_numeric.copy()

X_test_plus = X_test_numeric.copy()



# Make new columns indicating what will be imputed

for col in cols_with_missing:

    X_train_plus[col + '_was_missing'] = [int(x == True) for x in X_train_plus[col].isnull()]

    X_test_plus[col + '_was_missing'] = [int(x == True) for x in X_test_plus[col].isnull()]

    

X_train_plus["Fare" + '_was_missing'] = [int(x == True) for x in X_train_plus["Fare"].isnull()]

X_test_plus["Fare" + '_was_missing'] = [int(x == True) for x in X_test_plus["Fare"].isnull()]



X_train_plus["FarePerPerson" + '_was_missing'] = [int(x == True) for x in X_train_plus["Fare_Per_Person"].isnull()]

X_test_plus["FarePerPerson" + '_was_missing'] = [int(x == True) for x in X_test_plus["Fare_Per_Person"].isnull()]



# Imputation

my_imputer = SimpleImputer(strategy="mean")

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

imputed_X_test_plus = pd.DataFrame(my_imputer.transform(X_test_plus))



# Imputation removed column names; put them back

imputed_X_train_plus.columns = X_train_plus.columns

imputed_X_train_plus = imputed_X_train_plus.apply(pd.to_numeric, errors='ignore')



imputed_X_test_plus.columns = X_test_plus.columns

imputed_X_test_plus = imputed_X_test_plus.apply(pd.to_numeric, errors='ignore')
imputed_X_train_plus.head()
imputed_X_test_plus.head()
s = (X_full.dtypes == 'object')

object_cols = list(s[s].index)
X_train_categorical = X_full[object_cols]

X_test_categorical = X_test_full[object_cols]
cols_with_missing2 = (X_train_categorical.isnull().sum())

cols_with_missing2 = cols_with_missing2[cols_with_missing2 > 0]

cols_with_missing2 = cols_with_missing2.index

cols_with_missing2
# Make copy to avoid changing original data (when imputing)

X_train_plus2 = X_train_categorical.copy()

X_test_plus2 = X_test_categorical.copy()



# Make new columns indicating what will be imputed

for col in cols_with_missing2:

    X_train_plus2[col + '_was_missing'] = [int(x==True) for x in X_train_plus2[col].isnull()]

    X_test_plus2[col + '_was_missing'] = [int(x==True) for x in X_test_plus2[col].isnull()]

    

# X_train_plus2["IsMaster"] = [int(name.find("Master") >= 0) for name in X_train_plus2.Name]

# X_test_plus2["IsMaster"] = [int(name.find("Master") >= 0) for name in X_test_plus2.Name]



# Imputation

my_imputer2 = SimpleImputer(strategy="most_frequent")

imputed_X_train_plus2 = pd.DataFrame(my_imputer2.fit_transform(X_train_plus2))

imputed_X_test_plus2 = pd.DataFrame(my_imputer2.transform(X_test_plus2))



# Imputation removed column names; put them back

imputed_X_train_plus2.columns = X_train_plus2.columns

imputed_X_train_plus2 = imputed_X_train_plus2.apply(pd.to_numeric, errors='ignore')



imputed_X_test_plus2.columns = X_test_plus2.columns

imputed_X_test_plus2 = imputed_X_test_plus2.apply(pd.to_numeric, errors='ignore')
imputed_X_train_plus2.head()
imputed_X_test_plus2.head()
# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in imputed_X_train_plus2.columns if

                    imputed_X_train_plus2[cname].nunique() < 10 and 

                    imputed_X_train_plus2[cname].dtype == "object"]

categorical_cols
from sklearn.preprocessing import OneHotEncoder



# Use as many lines of code as you need!

# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(imputed_X_train_plus2[categorical_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(imputed_X_test_plus2[categorical_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = imputed_X_train_plus2.index

OH_cols_test.index = imputed_X_test_plus2.index
#get not one hot encoded columns

not_categorical_cols = [cname for cname in imputed_X_train_plus2.columns if

                    imputed_X_train_plus2[cname].nunique() >= 10 or 

                    (imputed_X_train_plus2[cname].dtype != "object")]

not_categorical_cols
processedCategoricalData = pd.concat([imputed_X_train_plus2[not_categorical_cols], OH_cols_train], axis=1)

processedCategoricalData.head()
processedCategoricalDataTest = pd.concat([imputed_X_test_plus2[not_categorical_cols], OH_cols_test], axis=1)

processedCategoricalDataTest.head()
fullProcessData = pd.concat([imputed_X_train_plus, processedCategoricalData], axis=1)
fullProcessDataTest = pd.concat([imputed_X_test_plus, processedCategoricalDataTest], axis=1)
# Select numeric columns only

numeric_cols = [cname for cname in fullProcessData.columns if fullProcessData[cname].dtype in ['int64', 'float64']]

X_processed = fullProcessData[numeric_cols].copy()



X_processedTest = fullProcessDataTest[numeric_cols].copy()
X_processed.head()
X_processedTest.head()
from sklearn.preprocessing import KBinsDiscretizer

enc = KBinsDiscretizer(n_bins=5, encode='onehot-dense')

X_FareBinned = enc.fit_transform(X_processed[["Fare_Per_Person"]])

X_Test_Binned = enc.transform(X_processedTest[["Fare_Per_Person"]])



X_processed2 = pd.concat([X_processed, pd.DataFrame(X_FareBinned, columns=["Fare1", "Fare2", "Fare3", "Fare4", "Fare5"])], axis=1)

X_processedTest2 = pd.concat([X_processedTest, pd.DataFrame(X_Test_Binned, columns=["Fare1", "Fare2", "Fare3", "Fare4", "Fare5"])], axis=1)
from sklearn.preprocessing import KBinsDiscretizer

enc = KBinsDiscretizer(n_bins=5, encode='onehot-dense')

X_AgeBinned = enc.fit_transform(X_processed[["Age"]])

X_Test_Binned2 = enc.transform(X_processedTest[["Age"]])



X_processed3 = pd.concat([X_processed2, pd.DataFrame(X_AgeBinned, columns=["Age1", "Age2", "Age3", "Age4", "Age5"])], axis=1)

X_processedTest3 = pd.concat([X_processedTest2, pd.DataFrame(X_Test_Binned2, columns=["Age1", "Age2", "Age3", "Age4", "Age5"])], axis=1)
X_processed3 = X_processed3.drop(["Age", "Fare"], axis=1)

X_processedTest3 = X_processedTest3.drop(["Age", "Fare"], axis=1)



X_processed3 = X_processed3.drop(["SibSp", "Parch"], axis=1)

X_processedTest3 = X_processedTest3.drop(["SibSp", "Parch"], axis=1)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



model = RandomForestClassifier(n_estimators=100, random_state=42)



# Multiply by -1 since sklearn calculates *negative* MAE

scores = cross_val_score(model, X_processed3, y,

                              cv=5,

                              scoring='accuracy')



print("Average Accuracy score:", scores.mean())
model.fit(X_processed3, y)

list(zip(X_processed3.columns,model.feature_importances_))
from xgboost import XGBClassifier



model_2 = XGBClassifier(n_estimators=300, learning_rate=0.01, max_depth=4, random_state=42) # Your code here



# Multiply by -1 since sklearn calculates *negative* MAE

scores2 = cross_val_score(model_2, X_processed3, y,

                              cv=5,

                              scoring='accuracy')



print("Average Accuracy score:", scores2.mean())
model_2.fit(X_processed3, y)

list(zip(X_processed3.columns,model_2.feature_importances_))
tempDF = pd.DataFrame()

tempDF["Features"] = X_processed3.columns

tempDF["Importance"] = model_2.feature_importances_



sortedImp = tempDF.sort_values(by="Importance", ascending=False)



superImp = sortedImp[sortedImp.Importance>0]

superImp.Features.values
X_processed4 = X_processed3[superImp.Features.values]



# Multiply by -1 since sklearn calculates *negative* MAE

scores3 = cross_val_score(model_2, X_processed4, y,

                              cv=5,

                              scoring='accuracy')



print("Average Accuracy score:", scores3.mean())
model_2.fit(X_processed3[superImp.Features.values], y)

list(zip(X_processed4,model_2.feature_importances_))
X_processed3[superImp.Features.values].corr()
#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

grid_n_estimator = [10, 50, 100, 300]

grid_learn = [.01, .03, .05, .1, .25]

grid_seed = [0]



grid_param = [    

            [{

            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html

            'learning_rate': grid_learn, #default: .3

            'max_depth': [1,2,4,6,8,10], #default 2

            'n_estimators': grid_n_estimator, 

            'seed': grid_seed  

             }]   

        ]
from sklearn.model_selection import GridSearchCV



model_3 = XGBClassifier(random_state=42) # Your code here

param = grid_param[0]



best_search = GridSearchCV(estimator = model_3, param_grid = param, cv = 5, scoring = 'accuracy')

best_search.fit(X_processed4, y)



best_param = best_search.best_params_

print('The best parameter for {} is {}'.format(model_3.__class__.__name__, best_param))

model_3.set_params(**best_param) 
model_3.fit(X_processed4, y)

list(zip(X_processed4,model_3.feature_importances_))
scores3 = cross_val_score(model_3, X_processed4, y,

                              cv=5,

                              scoring='accuracy')



print("Average Accuracy score:", scores3.mean())
preds_test = model_3.predict(X_processedTest3[superImp.Features.values])

preds_test[0:20]
sum(y)/len(y)
sum(preds_test)/len(preds_test)
#whoops lost the index

X_processedTest.index = X_test_full.index
# Save test predictions to file

output = pd.DataFrame({'PassengerId': X_processedTest.index,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)