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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
# Loading the data

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



train.shape, test.shape
# having a look

train.head()
# Checking % of NAN values in the columns



features_with_na = [features for features in train.columns if train[features].isnull().sum()>0]



print("Features and its % of missing values: ")

print()

for feature in features_with_na:

    print(feature, np.round(train[feature].isnull().mean(), 4))
for feature in features_with_na:

    data = train.copy()

    

    # making a variable that indicates 1 if the observation was missing and 0 if not

    # in simple, converting missing values to 1 and others to 0

    data[feature] = np.where(data[feature].isnull(), 1, 0)

    

    # calculating the mean SalePrice where the info is missing or present

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.title(feature)

    plt.show()
# List of numerical features

num_features = [feature for feature in train.columns if train[feature].dtypes != 'O']

# 'O' means object, i.e. if not Object than obviously it's numerical

print("No. of numerical features: ", len(num_features))



train[num_features].head()
# List of varables that contain Year variables

year_features = [feature for feature in num_features if 'Yr' in feature or 'Year' in feature]

# we used the above logic because all the year variables have either 'Yr' or 'Year' in its name



print("Year Variables: ", year_features)
for i in year_features:

    print(i, train[i].unique())
# Analyzing the Temporal datetime variable

# Checking for relation between house sold year and sales price



train.groupby('YrSold')['SalePrice'].median().plot()

plt.xlabel("Year sold")

plt.ylabel('Median House Price')
for feature in year_features:

    if feature != 'YrSold':

        data = train.copy()

        data[feature] = data['YrSold'] - data[feature]

        

        plt.scatter(data[feature], data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalePrice')

        plt.show()
discrete_feature=[feature for feature in num_features if len(train[feature].unique())<25 

                  and feature not in year_features+['Id']]



print("Discrete Variables Count: {}".format(len(discrete_feature)))

discrete_feature
train[discrete_feature].head()
for feature in discrete_feature:

    data = train.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel("SalePrice")

    plt.title(feature)

    plt.show()
contin_feature=[feature for feature in num_features if feature not in discrete_feature+ year_features+['Id']]

print("No. of Continuous Features: {}".format(len(contin_feature)))
# Creating histograms because we are trying to find out the distribution of the continuous variables

for feature in contin_feature:

    data = train.copy()

    data[feature].hist(bins=25)

    plt.xlabel(feature)

    plt.ylabel("Count")

    plt.title(feature)

    plt.show()
for feature in contin_feature:

    data = train.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature] = np.log(data[feature])

        data['SalePrice']= np.log(data['SalePrice'])

        plt.scatter(data[feature], data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel("SalePrice")

        plt.title(feature)

        plt.show()
# Finding the outliers



for feature in contin_feature:

    data = train.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature] = np.log(data[feature])

        data.boxplot(column=feature)

        plt.ylabel(feature)

        plt.title(feature)

        plt.show()
categorical_feature=[feature for feature in train.columns if train[feature].dtype=='O']



#len(categorical_feature)

categorical_feature
train[categorical_feature].head()
for feature in categorical_feature:

    print('The feature {} has {} No. of categories'.format(feature, len(train[feature].unique())))
for feature in categorical_feature:

    data = train.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
target = train['SalePrice']

test_id = test['Id']



train_for_con = train.drop(['Id', 'SalePrice'], axis=1)

test_for_con = test.drop(['Id'], axis=1)
train_for_con.shape, test_for_con.shape
combined = train_for_con.append(test_for_con)

combined.shape
# checking for missing values

col_with_na = [x for x in combined.columns if combined[x].isnull().sum()>0]



print("Features and its % of missing values: ")

print()

for y in col_with_na:

    print(y, np.round(combined[y].isnull().mean(), 6))
# Missing categorical features



col_nan = [x for x in combined.columns if combined[x].isnull().sum()>0 

           and combined[x].dtypes=='O']

print("Categorical Features and its % of missing values: ")

print()

for y in col_nan:

    print(y, np.round(combined[y].isnull().mean(), 4))
def replace_cat(combined, col_nan):

    data = combined.copy()

    data[col_nan]=data[col_nan].fillna('Missing')

    return data

combined = replace_cat(combined, col_nan)

combined[col_nan].isnull().any().sum()
num_nan = [x for x in combined.columns if combined[x].isnull().sum()>0

          and combined[x].dtypes !='O']



print("Numerical Features and its % of missing values: ")

print()

for y in num_nan:

    print(y, np.round(combined[y].isnull().mean(), 4))
for i in num_nan:

    median_value = combined[i].median()

    

    # create a new feature to capture the NAN values

    # we are replacing the nan values by 1 and others by 0 in the newly created feature

    combined[i + '_nan'] = np.where(combined[i].isnull(), 1, 0)

    combined[i].fillna(median_value, inplace=True)



combined[num_nan].isnull().any().sum()
for x in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:

    combined[x] = combined['YrSold'] - combined[x]



combined.head()
# We can see the outcome of the above code as follows

combined[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
# Skewed Features

num_features = ['LotFrontage','LotArea', '1stFlrSF', 'GrLivArea']



for feature in num_features:

    combined[feature] = np.log(combined[feature])

    

combined.head()
# Also our target, 'SalePrice' is skewed, so...

target = np.log(target)

target.head()
## Handling rare categorical features

# all the categorical features in our dataset

categorical_features = [x for x in combined.columns if combined[x].dtype=='O']



for y in categorical_features:

    

    # finding the present % of the features

    temp = combined.groupby(y)['MSSubClass'].count()/len(combined)

    

    # taking the index of the features with more than 1% presence

    temp_df = temp[temp>0.01].index

    

    # converting the rare features to a new label 'rare_var'

    combined[y] = np.where(combined[y].isin(temp_df), combined[y], 'Rare_var')
combined.shape
combined = pd.get_dummies(combined, drop_first=True)

combined.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

combined = pd.DataFrame(scaler.fit_transform(combined), 

                        columns=combined.columns)

combined.head()
X_train = combined[:1460]

X_test = combined[1460:]

y_train = target



X_train.shape, X_test.shape, y_train.shape
from sklearn.ensemble import GradientBoostingRegressor
# Fitting the model

gradient_boost = GradientBoostingRegressor(learning_rate=0.1,

                                           n_estimators=50).fit(X_train, y_train)

# Predicting for the test data

grad_boost_predictions = gradient_boost.predict(X_test)

grad_boost_predictions = np.expm1(grad_boost_predictions)
from sklearn.model_selection import RandomizedSearchCV
gb_regressor = GradientBoostingRegressor()



# Hyperparameter optimization



n_estimators = [100, 500, 1000, 1200, 1400]

max_depth = [2, 3, 5, 10, 15]

learning_rate = [0.05, 0.1, 0.15, 0.20]

min_samples_split = [1, 2, 3, 4]

min_samples_leaf =[1, 2, 3]

min_weight_fraction_leaf = [0, 1, 2]

min_impurity_decrease = [0, 1, 2]



# Define the grid of parameters to search

parameter_grid = {

    'n_estimators': n_estimators,

    'max_depth': max_depth,

    'learning_rate': learning_rate,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf,

    'min_weight_fraction_leaf': min_weight_fraction_leaf,

    'min_impurity_decrease': min_impurity_decrease

}
%%time

random_cv = RandomizedSearchCV(estimator=gb_regressor,

                              param_distributions=parameter_grid,

                              cv =5, n_iter=50,

                              scoring= 'neg_mean_absolute_error',

                              n_jobs=4,

                              verbose=5,

                              return_train_score=True,

                              random_state=0)
%%time

random_cv.fit(X_train, y_train)
random_cv.best_estimator_
regressor = GradientBoostingRegressor(max_depth=2,

                                      min_impurity_decrease=0,

                                      min_samples_split=3, 

                                      min_weight_fraction_leaf=0,

                                      n_estimators=1000)
regressor.fit(X_train, y_train)
gd_predictions = regressor.predict(X_test)

gd_predictions = np.expm1(gd_predictions)
gd_predictions = pd.DataFrame(gd_predictions)

gd_predictions = pd.concat([test['Id'], gd_predictions], axis = 1)

gd_predictions.columns = ['Id', 'SalePrice']

gd_predictions.to_csv("gd_predictions.csv", index = False)