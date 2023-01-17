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
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
# View training dataframe

print(train.info())
# View testing dataframe

print(test.info())
# Convert all object data columns to categorical/one-hot data dummy columns

import pandas as pd

from pandas import get_dummies



def all_categorical(data):

    object_columns = []

    for column in data:

        if data[column].dtype == object:

            object_columns.append(column)

    new_data = get_dummies(data, columns=object_columns, drop_first=True)

    return new_data



test = all_categorical(test)

train = all_categorical(train)
# Count number of missing values in dataframes

train.isna().sum()

test.isna().sum()
# Fill missing values of columns with either mean (if column feature is categorical) or median (if column feature is numerical)



def fill_missing(data):

    for column in data:

        if data[column].values.any():

            if data[column].max() == 1:

                column_median = data[column].median()

                updated_column = data[column].fillna(value=column_median)

                data.update(updated_column)

            else:

                column_mean = data[column].mean()

                updated_column = data[column].fillna(value=column_mean)

                data.update(updated_column)



fill_missing(train)

fill_missing(test)
# Count number of missing values in dataframes

train.isna().sum()

test.isna().sum()
# Find a list of features that are highly correlated with sales price and not correlated with one another and put 

# them in a modified dataframe.



import seaborn as sns

%matplotlib inline



cor_matrix = train.corr()



cor_columns = list(cor_matrix.columns)



def find_cor_features(matrix, target_column, threshold_cor, num_features):

    cor_features = []

    blacklist = []

    blacklist.append(target_column)

    while len(cor_features) < num_features:

        max_cor = 0

        max_feature = ''

        for column in matrix.columns:

            column_cor = matrix[target_column].corr(matrix[column])

            if column_cor > max_cor:

                if column not in blacklist:

                    if column not in cor_features:

                        max_cor = column_cor

                        max_feature = column

        feature_correlated = False

        for features in cor_features:

            if matrix[features].corr(matrix[max_feature]) >= threshold_cor:

                feature_correlated = True

                blacklist.append(max_feature)

        if not feature_correlated: 

            cor_features.append(max_feature)

    return cor_features

                

cor_features = find_cor_features(train, 'SalePrice', .6, 20)               



#train_mod = pd.DataFrame()

test_mod = pd.DataFrame()



train_mod['SalePrice'] = train['SalePrice']

for feature in cor_features:

#    train_mod[feature] = train[feature]

    test_mod[feature] = test[feature]
# Examine correlation between features in a heatmap



import seaborn as sns



train_mod_cor = train_mod.corr()



sns.heatmap(train_mod_cor, xticklabels=train_mod_cor.columns, yticklabels=train_mod_cor.columns)

# Save sale price as a seperate dataframe and drop it from the train dataframe



sale_price = train_mod['SalePrice']



train_mod = train_mod.drop('SalePrice', 1)

train = train.drop('SalePrice', 1)
# Linear Regression Model (Ordinary Least Squares)



from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



x_train, x_test, y_train, y_test = train_test_split(train_mod, sale_price, test_size=0.2, random_state=10)



Linear_Regression = LinearRegression()

Linear_Regression.fit(x_train, y_train)

y_predict = Linear_Regression.predict(x_test)

RMSE = sqrt(mean_squared_error(y_test, y_predict))

RMSE
# Random Forest Regression Model 



from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



x_train, x_test, y_train, y_test = train_test_split(train_mod, sale_price, test_size=0.2, random_state=10)



forest = RandomForestRegressor(n_estimators = 20)

forest.fit(x_train, y_train)

y_predict = forest.predict(x_test)

RMSE = sqrt(mean_squared_error(y_test, y_predict))

RMSE
# Support Vector Regression



from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR



x_train, x_test, y_train, y_test = train_test_split(train_mod, sale_price, test_size=0.2, random_state=10)



svr = SVR(gamma='scale')

svr.fit(x_train, y_train)

y_predict = forest.predict(x_test)

RMSE = sqrt(mean_squared_error(y_test, y_predict))

RMSE
# Feedforward Neural Network (with select features and all features)



from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import train_test_split

from keras.layers import Dense

from keras.models import Sequential 

from keras.optimizers import SGD



x_train, x_test, y_train, y_test = train_test_split(train_mod, sale_price, test_size=0.2, random_state=10)



model = Sequential()



model.add(Dense((64), activation='linear', input_dim=len(train_mod.columns)))

model.add(Dense((32), activation='relu'))

model.add(Dense((1), activation='linear'))



model.compile(loss='mean_squared_error', optimizer='adam')



model.fit(x_train, y_train, epochs=100, batch_size = 128, verbose=0)



y_predict = model.predict(x_test)

RMSE = sqrt(mean_squared_error(y_test, y_predict))

print(RMSE)
from sklearn.ensemble import RandomForestRegressor

from mlxtend.feature_selection import SequentialFeatureSelector as SFS



forest = RandomForestRegressor(n_estimators = 20)

x_train, x_test, y_train, y_test = train_test_split(train, sale_price, test_size=0.2, random_state=10)



sfs = SFS(forest, 

           k_features=20,

           forward=True, 

           floating=True, 

           verbose=2,

           scoring='neg_mean_squared_error',

           cv=0)



sfs = sfs.fit(x_train, y_train)



x_train_sfs = sfs.transform(x_train)

x_test_sfs = sfs.transform(x_test)



forest.fit(x_train_sfs, y_train)

y_predict = forest.predict(x_test_sfs)



RMSE = sqrt(mean_squared_error(y_test, y_predict))

RMSE
# Making CSV Prediction for Submission 



id_column = test['Id']



sp_column = forest.predict(test_mod)



submission = pd.DataFrame({ 'Id': id_column, 'SalePrice': sp_column})

submission.to_csv("my_submission.csv", index=False)