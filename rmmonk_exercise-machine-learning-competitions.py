# IMPORTS

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

home_data.describe()

# ANALYZE AND DEAL WITH NAN VALUES (Create a copy named home_data_nan)

home_data_nan = home_data.copy()



missing_val_count_by_column = (home_data_nan.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

# NOTE THAT Alley, FireplaceQu, PoolQC, Fence, MiscFeature have many NaN values out of total 1460 observations.

# We will drop these columns from the analysis vs. imputing them as the info is missing the majority of the values and it will be difficult to reliably impute a value

home_data_v1 = home_data_nan.drop(['Alley', 'PoolQC', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

home_data_v1.head()

# We will analyze the remaining data again further and decide on an imputation strategy



# View all the rows with NAN values in them

home_data_v1[home_data_v1.isnull().any(axis=1)].head()

# GARAGE features are NaN when garage area = 0. This is not NaN but should be changed to No Garage or NG as it could play an important role in the price.

# From the Null Count above, we can see that there is 81 NaN values for all the garage categories and after running this code we can see there is 81 rows where GarageArea is equal to 0, or NoGarage.



# GarageType        81

# GarageYrBlt       81

# GarageFinish      81

# GarageQual        81

# GarageCond        81



# Comment the describe or head code out to view a sample or statsdescribe of the dataset

# View all rows where GarageArea = 0 and the describe or stats of those columns

home_data_v1.loc[home_data_v1.GarageArea == 0].describe()

garage_features = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']



# Lets see how many NaN values we have for the 5 features above, when the GarageArea is equal to 0:

for feature in garage_features:

    print(home_data_v1[feature].loc[home_data_v1.GarageArea == 0].isnull().sum())

# MAKE COPY OF DATA

home_data_v2 = home_data_v1.copy()



for feature in garage_features:

    home_data_v2[feature].fillna(value='NoGar', inplace=True)



home_data_v2.describe()

# Total rows should be 1460 to ensure we have all rows
# Check missing value counts on v2, decide next data clean step

missing_val_count_by_column = (home_data_v2.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

# There appears to be two instances, one in BsmtExposure and one in BsmtFinType2 where there is an extra null value over the 37 in all other categories. 



# Steps for each one 

# 1.Check various categories in each column

# 2. Impute these one offs with most likely category.



# TO print the categories

basement_one_offs = ['BsmtExposure', 'BsmtFinType2']



for feature in basement_one_offs:

    print(home_data_v2[feature].value_counts())

# # ID #949

# home_data_v2[home_data_v2.BsmtExposure.isnull()]



# # ID # 333

# home_data_v2[home_data_v2.BsmtFinType2.isnull()]
# Basement features seem to have a similar number of NaN (37, with one 38)

# Just as there was no garage, we can check to see if in these cases there are no basements (apartment, condo, mobile home etc.)

# Check the NaN values when the 'TotalBsmtSF' is equal to 0

basement_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']



# Lets see how many NaN values we have for the 5 features above, when the TotalBsmtSF is equal to 0:

for feature in basement_features:

    print(home_data_v2[feature].loc[home_data_v2.TotalBsmtSF == 0].isnull().sum())

# So it would be clear that for 37 of the NaN entries, TotalBsmtSF is 0, which indicates no basement and not NaN.

# For those columns listed above I will add a new category of NoBase which indicates no basement.



home_data_v3 = home_data_v2.copy()



for feature in basement_features:

    home_data_v3[feature].fillna(value='NoBase', inplace=True)



home_data_v3.describe()

# Check missing value counts on v3



missing_val_count_by_column = (home_data_v3.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

# SCript to run value counts on various features at same time:



# for feature in ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']:

#     print(home_data_nan_v1[feature].value_counts())
# Create target object and call it y

y = home_data_v3.SalePrice



# Create X

# SELECT FEATURES (THE FACTORS WE HYPOTHESIZE WILL AFFECT OUR TARGET)



# LESSON FEATURES

lesson_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']





#Pearson_coefficient_features ('MasVnrArea', and 'GarageYrBlt', removed due to NaN)

pearson_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 

                    'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 

                    'GarageArea']



# CEATE OUR X DATAFRAME BY ISOLATING THE FEATURES WE WISH TO USE

X = home_data_v3[pearson_features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Various Models and Check Accuracy by comparing Mean Average Error against train/test data/predictions

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# SCRATCH PAD

missing_val_count_by_column = (X.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
## Use the below code to list the pearson coefficient of the various integer cols against sale price. 

## Select from list the coefficients that are highest/hypothesized to be good predictors of SalePrice.

def plot_dataframe(dataframe, y_label):

    """Generates a dataframe plot to be used by pandas in graph_correlations function."""

    color = 'coral'

    fig = plt.gcf()

    fig.set_size_inches(30, 12)

    plt.ylabel(y_label)



    ax = dataframe.correlation.plot(linewidth=3.3, color=color)

    ax.set_xticks(dataframe.index)

    ax.set_xticklabels(dataframe.attributes, rotation=75);  # REMOVE THE ; and see what happens

    plt.show()



def graph_correlations(dataset, features):

    """Calculate and graph all the Pearson correlations of a dataset"""

    """ Pearson's: -1 to +1 range, 0 is no relation, 0-1 is possible correlation, higher, 

        more likely, lower, less likely"""

    """Generate the list of features that we wish to graph"""



    cols = features



    """Creates a list of correlations to be plotted"""

    correlations = [dataset['SalePrice'].corr(dataset[f]) for f in cols]



    dataframe = pd.DataFrame({'attributes': cols, 'correlation': correlations})



    plot_dataframe(dataframe, "Data Factor Correlation to House Sale Price")



    print(len(cols), len(correlations))  # These should equal eachother to ensure our list is same length.





def print_correlations(dataset, features):

    cols = features



    for f in cols:

        pearson_coefficient = dataset['SalePrice'].corr(dataset[f])

        print("%s: %f" % (f, pearson_coefficient))



all_measurable_features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

                           'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

                           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

                           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 

                           'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

                           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']



print_correlations(home_data, all_measurable_features)



# Select final features to include based on correlation coefficients in int (new_)

final_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF',

       'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea']



# print_correlations(home_data, final_features)
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction. Deal with NaN data as well. 

# The list of columns is stored in a variable called features

# Will fill NaN values with zero as most likely explanation is basement/garage does not exist

test_X = test_data[pearson_features].fillna(0)



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
