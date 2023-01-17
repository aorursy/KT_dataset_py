# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

pd.options.display.max_columns = 1000

data = pd.read_csv('/kaggle/input/ameshousingdataset/AmesHousing.tsv', delimiter='\t')
data
# create initial function to transform features

def transform_features(df):

    return df



# create initial function to select features

def select_features(df):

    return df[['Gr Liv Area','SalePrice']]



# create function to train and test the model

def train_and_test(df):

    train = df[:1460]

    test = df[1460:]

    

    #select only the numerical columns

    numeric_train = train.select_dtypes(include=['integer','float'])

    numeric_test = test.select_dtypes(include=['integer','float'])

    

    numeric_train = train.select_dtypes(include=['integer', 'float'])

    numeric_test = test.select_dtypes(include=['integer', 'float'])

    

    # drop target column from training

    features = numeric_train.columns.drop('SalePrice')

    features = numeric_train.columns.drop("SalePrice")



    

    # train and test the model

    lr = LinearRegression()

    lr.fit(train[features], train["SalePrice"])



    pred = lr.predict(test[features])

    mse = mean_squared_error(test['SalePrice'], pred)

    rmse = mse**0.5

    return rmse



# test the functions and check the final output

transform_df = transform_features(data)

filtered_df = select_features(transform_df)

rmse = train_and_test(filtered_df)



rmse
# drop column with 5% or more missing values

missing = data.isnull().sum()

missing_col = (missing[missing > len(data)/20]).index

data = data.drop(missing_col, axis=1)
# drop text columns with 1 or more missing values

text_col = data.select_dtypes(include=['object']).columns

missing_text = data[text_col].isnull().sum()

ms_text_col = missing_text[missing_text >= 1].index

data = data.drop(ms_text_col, axis=1)
#fill missing value in numerical column with the most frequent value

num_col = data.select_dtypes(include=['integer','float']).columns

num_miss = data[num_col].isnull().sum()

num_miss_cols = num_miss[num_miss > 0].index

data[num_miss_cols] = data[num_miss_cols].fillna(data[num_miss_cols].mode().iloc[0])

data.isnull().sum().sort_values()
# create new feature to indicate the amount of years until the house is sold

data['years_sold'] = data['Yr Sold'] - data['Year Built']



# check for incorrect value

data['years_sold'][data['years_sold'] < 0] 
# create new feature to indicate the amount of years until it's being renovated from sale time

data['years_since_remod'] = data['Yr Sold'] - data['Year Remod/Add']



# check for incorrect value

data['years_since_remod'][data['years_since_remod'] < 0]
# drop the incorrect values from previous step

data = data.drop([2180, 1702, 2180, 2181], axis=0)



# remove the original column (not needed anymore)

data = data.drop(['Yr Sold', 'Year Remod/Add', 'Year Built'], axis=1)
# drop the columns that are not useful for the model

data = data.drop(['PID', 'Order'], axis=1)



# drop the columns that leak information about the sale

data = data.drop(['Mo Sold', 'Sale Type', 'Sale Condition'], axis=1)
def transform_features(df):

    # drop column with 5% or more missing values

    missing = df.isnull().sum()

    missing_col = (missing[missing > len(df)/20]).index

    df = df.drop(missing_col, axis=1)

    

    # drop text columns with 1 or more missing values

    text_col = df.select_dtypes(include=['object']).columns

    missing_text = df[text_col].isnull().sum()

    ms_text_col = missing_text[missing_text >= 1].index

    df = df.drop(ms_text_col, axis=1)

    

    #fill missing value in numerical column with the most frequent value

    num_col = df.select_dtypes(include=['integer','float']).columns

    num_miss = df[num_col].isnull().sum()

    num_miss_cols = num_miss[num_miss > 0].index

    df[num_miss_cols] = df[num_miss_cols].fillna(df[num_miss_cols].mode().iloc[0])

    

    # create new features

    df['years_sold'] = df['Yr Sold'] - df['Year Built']

    df['years_since_remod'] = df['Yr Sold'] - df['Year Remod/Add']

    

    df = df.drop([2180, 1702, 2180, 2181], axis=0)



    # drop not needed & leaking columns

    df = df.drop(['Yr Sold', 'Year Remod/Add', 'Year Built', 'PID', 'Order', 'Mo Sold',

                  'Sale Type', 'Sale Condition'], axis=1)



    return df



# test the function

df = pd.read_csv('/kaggle/input/ameshousingdataset/AmesHousing.tsv', delimiter='\t')

transformed = transform_features(df)

selected_features = select_features(transformed)

test = train_and_test(selected_features)

test
# check the numerical columns

num_df = transformed.select_dtypes(include = ['float', 'integer'])

num_df
# build the correlation with target column

corr = num_df.corr()['SalePrice'].abs().sort_values()

corr
# filter the columns with the correlation > 0.4

corr = corr[corr > 0.4] 

corr
# drop the low correlating columns

transformed = transformed.drop(corr[corr < 0.4].index, axis=1)
# Create a list of column names from documentation that should be categorical

nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 

                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 

                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 

                    "Misc Feature", "Sale Type", "Sale Condition"]



# list the columns that need to be transformed

trans_col = []

for i in nominal_features:

    if i in transformed.columns:

        trans_col.append(i)



# find out the unique values in each column

unique_counts = transformed[trans_col].apply(lambda x: len(x.value_counts())).sort_values()



# set the threshold for the amount of unique values. Here we will use column with <10 unique values.

nonunique_counts = unique_counts[unique_counts > 10]



# drop the columns with unique values >10

transformed = transformed.drop(nonunique_counts.index, axis=1) 
# select the remaining text columns and convert it to categorical data

text_cols = transformed.select_dtypes(include=['object'])



for i in text_cols:

    transformed[i] = transformed[i].astype('category')



# create dummy columns and drop the original columns

transformed = pd.concat([transformed,

                         pd.get_dummies(transformed.select_dtypes(include=['category']))

                        ], axis =1).drop(text_cols, axis=1)
def select_features(df, corrval=0.4, threshval=10):

    # check the numerical columns

    num_df = df.select_dtypes(include = ['float', 'int'])

    

    # build the correlation with target column

    corr = num_df.corr()['SalePrice'].abs().sort_values()

    

    # drop the low correlating columns

    df = df.drop(corr[corr < corrval].index, axis=1)

    

    # List the categorical columns

    nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 

                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 

                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 

                    "Misc Feature", "Sale Type", "Sale Condition"]

    

    # list the columns that need to be transformed

    trans_col = []

    for i in nominal_features:

        if i in df.columns:

            trans_col.append(i)

            

    # find out the unique values in each column

    unique_counts = df[trans_col].apply(lambda x: len(x.value_counts())).sort_values()



    # set the threshold for the amount of unique values

    nonunique_counts = unique_counts[unique_counts > threshval]



    # drop the columns with unique values exceeding threshold value

    df = df.drop(nonunique_counts.index, axis=1) 

    

    # select the remaining text columns and convert it to categorical data

    text_cols = df.select_dtypes(include=['object'])



    for i in text_cols:

        df[i] = df[i].astype('category')



    # create dummy columns and drop the original columns

    df = pd.concat([df,

                         pd.get_dummies(df.select_dtypes(include=['category']))

                        ], axis =1).drop(text_cols, axis=1)



    return df



# test the function

df = pd.read_csv('/kaggle/input/ameshousingdataset/AmesHousing.tsv', delimiter='\t')

transformed = transform_features(df)

selected_features = select_features(transformed)

test = train_and_test(selected_features)

test
from sklearn.model_selection import cross_val_score, KFold
# The function accepts k parameter, k=0 (default) for holdout validation, k=1 for cross validation,

# and k fold validation

def train_and_test(df, k=0):

    num_df = df.select_dtypes(include=['float', 'int'])

    features = df.columns.drop('SalePrice')

    lr = LinearRegression()



    if k==0:

        train = df[:1460]

        test = df[1460:]

    

        # train and test the model

        lr.fit(train[features], train["SalePrice"])

        pred = lr.predict(test[features])

        mse = mean_squared_error(test['SalePrice'], pred)

        rmse = np.sqrt(mse)

        

    elif k==1:

        # randomize order of rows

        np.random.seed(1)

        shuffled_index = np.random.permutation(df.index)

        df = df.reindex(shuffled_index)

        

        train = df[:1460]

        test = df[1460:]

    

        # train and test the model

        lr.fit(train[features], train["SalePrice"])

        pred1 = lr.predict(test[features])

        mse1 = mean_squared_error(test['SalePrice'], pred1)

        rmse1 = np.sqrt(mse1)

        

        lr.fit(test[features], test["SalePrice"])

        pred2 = lr.predict(train[features])

        mse2 = mean_squared_error(train['SalePrice'], pred2)

        rmse2 = np.sqrt(mse2)

        

        rmse = (rmse1 + rmse2) / 2

        

    else:

        kf = KFold(n_splits=k, shuffle=True, random_state=1)

        mses = cross_val_score(estimator=lr, X=df[features], y=df['SalePrice'], scoring='neg_mean_squared_error', cv=kf)

        rmse = np.mean(abs(mses)**0.5)



    return rmse



df = pd.read_csv('/kaggle/input/ameshousingdataset/AmesHousing.tsv', delimiter='\t')

transformed = transform_features(df)

selected_features = select_features(transformed)

test = train_and_test(selected_features, k=5)

test