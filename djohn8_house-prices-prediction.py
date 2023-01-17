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
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_train.head()
print('Size of training set is : {}\n\n'.format(df_train.shape))

display(df_train.info())

print('Size of test set is : {}\n\n'.format(df_test.shape))

display(df_test.info())
null = df_train.isnull().sum()

#display(null)

null_df = pd.DataFrame({'features': null.index, 'count': null.values})

display(null_df[null_df['count'] > 0])



print('Max entries per column in the dataset is: {}'.format(df_train.shape[0]))

print('Columns with more than 50% of the data missing in the training data is \n: {}'.format(null_df[null_df['count'] > df_train.shape[0]*0.5]))

#null_df[null_df['count'] > df_train.shape[0]*0.5]
def check_null(df):

    null = df.isnull().sum()

    #display(null)

    null_df = pd.DataFrame({'features': null.index, 'count': null.values})

    return null_df[null_df['count'] > 0]
def check_null_50(df):

    null = df.isnull().sum()

    #display(null)

    null_df = pd.DataFrame({'features': null.index, 'count': null.values})

    return null_df[null_df['count'] > df.shape[0]*0.5]
print('Columns with more than 50% of the data missing in the training data are')

check_null_50(df_train)
print('Columns with more than 50% of the data missing in the test data are')

check_null_50(df_test)
import seaborn as sns

sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False)
# features to remove, which have more than 50% missing data

remove_features = null_df[null_df['count'] > df_train.shape[0]*0.5].features.tolist()

remove_features
# remove the columns from the train and test sets

df_train.drop(remove_features, axis=1, inplace=True)

df_test.drop(remove_features, axis=1, inplace=True)
# remove the 'Id'column in both train and test

df_train.drop('Id', axis=1, inplace=True)

df_test.drop('Id', axis=1, inplace=True)

df_train.head()
display(df_train.shape)

display(df_test.shape)
#df_train.dtypes

# display(check_null(df_train))

# display(check_null(df_test))
#df_train[df_train.isnull().any(axis=1)]

# select the object data types

df_train_obj = df_train.select_dtypes(include=['object']).copy()



# check for missing in object type

#check_null(df_train_obj)

df_train_obj.empty
def fill_missing(df):

    df_obj = df.select_dtypes(include=['object']).copy()

    if df_obj.empty==False:

        for col in check_null(df_obj)['features'].tolist():

            df[col] = df[col].fillna(df[col].mode()[0])

    

    df_numeric = df.select_dtypes(include=['int64', 'float64']).copy()

    if df_numeric.empty==False:

        for col in check_null(df_numeric)['features'].tolist():

            df[col] = df[col].fillna(df[col].mean())
fill_missing(df_train)

fill_missing(df_test)
check_null(df_train)

check_null(df_test)
# for col in check_null(df_train_obj)['features'].tolist():

#     df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
#df_train['FireplaceQu'].mode()[0]

# check_null(df_train)
# df_train_numeric = df_train.select_dtypes(include=['int64', 'float64']).copy()

# check_null(df_train_numeric)
# for col in check_null(df_train_numeric)['features'].tolist():

#     df_train[col] = df_train[col].fillna(df_train[col].mean())



# # check for any missing values

# check_null(df_train)
# df_train.info()

# print('\n\nFinal training data shape is {}'.format(df_train.shape))
# df_train_obj['MSZoning'].value_counts()

#df_train_obj['MSZoning']
#pd.get_dummies(df_train_obj['MSZoning'], drop_first=True)
# Ref: https://www.datacamp.com/community/tutorials/joining-dataframes-pandas



combined_df = pd.concat([df_train, df_test], axis=0, sort=False, keys=['train', 'test']) # row wise

combined_df
# to get the train and test sets back, use the 'keys'

combined_df.loc['train']

combined_df.loc['test']
#for col in df_train_obj.columns.tolist():

# Ref: https://pbpython.com/categorical-encoding.html



# df_train_obj_ohe = pd.get_dummies(df_train_obj, columns = df_train_obj.columns.tolist())



# #df_train_obj_ohe

# # drop the columns for which we have encoded from df_train

# cat_cols_drop = df_train_obj.columns.tolist()

# df_train.drop(cat_cols_drop, axis=1, inplace=True)



# df_train_final = pd.concat([df_train, df_train_obj_ohe], axis=1)

# df_train_final
def cat_ohe_multcols(df):

    df_obj = df.select_dtypes(include=['object']).copy()

    columns = df_obj.columns.tolist()

    df_obj_ohe = pd.get_dummies(df, columns= columns, drop_first=True)

    return df_obj_ohe
onehot_df = cat_ohe_multcols(combined_df)

temp = pd.concat([combined_df,onehot_df], axis=1)



# remove the duplicated columns

final_df = temp.loc[:, ~temp.columns.duplicated()]
final_df['SalePrice']
final_df.shape
# check for duplicate columns

np.unique(final_df.columns.duplicated())
df_Train = final_df.loc['train']

df_Test = final_df.loc['test']
df_Train
# check if there any columns of obj type

display(df_Train.info())

display(df_Test.info())
#final_df.select_dtypes(include=['int64', 'float64'])
# drop 'SalesPrice' from df_Test, which we need to predict

df_Test.drop(['SalePrice'], axis=1, inplace=True)
# Train

X_train = df_Train.drop(['SalePrice'], axis=1)

y_train = df_Train['SalePrice']
import xgboost

xgbReg = xgboost.XGBRegressor()

xgbReg.fit(X_train, y_train)
#prediction

y_pred = xgbReg.predict(df_Test)
y_pred
# create a sample submission

submission_temp = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

pred_df = pd.DataFrame(y_pred)

submission = pd.concat([submission_temp['Id'], pred_df], axis=1)

submission.columns=['Id', 'SalePrice']

submission.to_csv('submission.csv', index=False)