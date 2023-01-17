# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print("Input Files:", os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# My notes:

# The current directory is called "working"

print("Working directory: ", os.listdir("../working"))
#Load the training and test data:

#

#Following this Kaggle tutorial: https://www.kaggle.com/dansbecker/basic-data-exploration

# save filepath to variable for easier access

training_data_file_path = '../input/train.csv'

# read the data and store data in DataFrame

training_data = pd.read_csv(training_data_file_path) 



# save filepath to variable for easier access

test_data_file_path = '../input/test.csv'

# read the data and store data in DataFrame

test_data = pd.read_csv(test_data_file_path) 



#Setting the output data

y = training_data.SalePrice
# Deal with categorical data:

#

# List categorical variables

category_columns = training_data.select_dtypes(include=['object']).columns



# Deal with "good" NA values:

#

# Unfortunately, the negative case for some categorical features was "NA" which translates to np.nan

# So, here, I replace the np.nan entries in those categorical features with "No"

# List of features giving "NA" as a usable category (such as 'NA = no fireplace'): 

valid_na_features = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

                     'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC',

                     'Fence','MiscFeature']

print('Number of features where NA is meaningful: ', len(valid_na_features))

print('List of features where NA is meaningful: ', valid_na_features)

# Replace the offending entries in train and test data

training_data[valid_na_features] = training_data[valid_na_features].replace(np.nan,'No')

test_data[valid_na_features] = test_data[valid_na_features].replace(np.nan,'No')





# Combine training and test data

df1 = training_data.drop(labels = ['SalePrice'], axis=1) 

df2 = test_data

frames = [df1,df2]

combined = pd.concat(frames)



# Turn object columns into category columns (better for get_dummies)

for column in combined.select_dtypes(include=[np.object]).columns:

    # get list of (combinded) entries in a categorical feature (excluding nulls)

    cleaned_cats = [x for x in combined[column].unique() if str(x) != 'nan']

    training_data[column] = training_data[column].astype('category', categories = cleaned_cats)

    test_data[column] = test_data[column].astype('category', categories = cleaned_cats)



# Deal with missing categorical data:

#

# Create a table of all features and the number of "NA" entries they have

missing_val_count_by_column = (combined[category_columns].isnull().sum()) # this is an indexed series (blurgh)

# Create a list of all features with "NA" entries

missing_val_columns = missing_val_count_by_column[missing_val_count_by_column > 0]

print('Number of categorical features with NA entries: ',missing_val_columns.size)

print('List of categorical features with NA entries: ')

print(missing_val_columns)

cat_vars_missing_data = missing_val_columns.index
# Start massaging X. (do all the same things to X_test) ***

X = training_data.drop(labels = ['Id', 'SalePrice'], axis=1)

#drop the categorical variables with bad data (see cell above)

X = X.drop(labels = cat_vars_missing_data, axis=1) 
#Handle missing values in numeric columns:

X_numeric = X._get_numeric_data()

#

#Following this answer (because imputing converts from DF to np array, sigh): 

#https://stackoverflow.com/questions/33660836/impute-entire-dataframe-all-columns-using-scikit-learn-sklearn-without-itera

from sklearn.preprocessing import Imputer

fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)

imputed_X = pd.DataFrame(fill_NaN.fit_transform(X_numeric))

imputed_X.columns = X_numeric.columns

imputed_X.index = X_numeric.index

X_numeric = imputed_X

X[X_numeric.columns] = X_numeric
# Turn categorical data into dummies

X = pd.get_dummies(X)

#Following the man page for Ridge: 

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

# from sklearn.linear_model import Ridge

#from sklearn.feature_selection import RFE

#ridge_model = Ridge(normalize=True)

#selector = RFE(ridge_model, 10, step=1)

#selector.fit(X, y) 

#print(selector.score(X, y)) 

#Xd = X.iloc[:,selector.support_]



#from sklearn.neural_network import MLPRegressor

#nn = MLPRegressor(hidden_layer_sizes=(50,50,50),max_iter=2000)

#nn.fit(Xd,y)

#nn.score(Xd,y)



from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=15)

clf = clf.fit(X, y)

clf.score(X,y)
selector = clf
# Start massaging X_test. ***

X_test = test_data.drop(labels = ['Id'], axis=1)

# Drop categorical features with actual missing data

X_test = X_test.drop(labels = cat_vars_missing_data, axis=1)



# Impute the missing data in the numeric fields

X_test_numeric = X_test._get_numeric_data()

imputed_X_test = pd.DataFrame(fill_NaN.fit_transform(X_test_numeric))

imputed_X_test.columns = X_test_numeric.columns

imputed_X_test.index = X_test_numeric.index

X_test_numeric = imputed_X_test

X_test[X_test_numeric.columns] = X_test_numeric



# Turn categorical data to dummies

X_test = pd.get_dummies(X_test)



X_pred = selector.predict(X_test)
# Following this tutorial: https://www.kaggle.com/dansbecker/submitting-from-a-kernel

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': X_pred})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
print(os.listdir("../working"))
# Some resources:

# Kaggle tutorial:https://www.kaggle.com/dansbecker/your-first-machine-learning-model

# Categorical encoding:

# https://pbpython.com/categorical-encoding.html

# Training with dummies: 

# https://markhneedham.com/blog/2017/07/05/pandasscikit-learn-get_dummies-testtrain-sets-valueerror-shapes-not-aligned/

# all_data = pd.concat((train,test))

# for column in all_data.select_dtypes(include=[np.object]).columns:

#     train[column] = train[column].astype('category', categories = all_data[column].unique())

#     test[column] = test[column].astype('category', categories = all_data[column].unique())



# Next step ideas:

# Next: try polynomial regression -- multiply columns up to, say, degree 6. 

# (this would cause too many features)

# Next: use sklearn.impute.SimpleImputer instead

# Note: 1MSSubClass is secretly a categorical feature

# Note: BsmtQual is secretly a numerical feature