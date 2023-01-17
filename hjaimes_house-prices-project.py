# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# LIBRARIES IMPORTING #

#######################

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn.ensemble

import xgboost as xgb

from scipy import stats

from math import sqrt

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder

from sklearn.decomposition import PCA

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer, explained_variance_score

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping





# EVALUATION CRITERIA #

#######################

# Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value

# and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive

# houses and cheap houses will affect the result equally.)





# GLOBAL VARIABLES #

####################

# Parameter to control the visualization of top frequency values (restricted to the N_MAX value)

# describe() method will be used

FACT = 0.25 # Spacing factor of the levels on encoding categorical variables

N_PCA = 12

T_SIZE = 0.15

UGLY_M = 9999.99

EPOCHS = 150

PATIENCE = 20



# Grouping columns of Set train/test (for data exploration)

object_columns_names = [] # Names of object columns _All



# Dealing with duplication

columns_names_to_evaluate_duplication_subset = [] # Refined variables to search for duplication - train



# Dealing with target column name

target_column_name = 'SalePrice'



# Important lists/DataFrames

# Lists

y_train = [] # Target training values

y_train_predicted = [] # Target predicted training values - It will be generated

y_test_predicted = [] # Target predicted testing values - It will be generated

y_predicted = []  # All predicted values - It will be generated

# DataFrames

X_train = pd.DataFrame() # Training features - numeric

X_train_norm = pd.DataFrame() # Training features normalized/standard scaled - numeric

X_train_scaled = pd.DataFrame() # Training features minmax scaled - numeric

X_train_pca = pd.DataFrame() # Training features PCA - numeric

X_train_re = pd.DataFrame() # Training features robust scaler - numeric

X_test = pd.DataFrame() # Testing features - numeric

X_test_norm = pd.DataFrame() # Testing features normalized/standard scaled - numeric

X_test_scaled = pd.DataFrame() # Testing features minmax scaled - numeric

X_test_pca = pd.DataFrame() # Training features PCA - numeric

X_test_re = pd.DataFrame() # Training features robust scaler - numeric

X = pd.DataFrame() # All numerical features (training + testing)

X_norm = pd.DataFrame() # All normalized features (training + testing)

X_scaled = pd.DataFrame() # All scaled features (training + testing)

X_pca = pd.DataFrame() # Training features PCA - numeric

X_re = pd.DataFrame() # Training features robust scaler - numeric



# For batch training

X_b_train = []

y_b_train = []

X_b_test = []

y_b_test = []





# FUNCTIONS #

#############

# Competition metric defined function

def competition_metric(y_values, y_predicted):

        if y_predicted.min() <= 0.0:

            return UGLY_M

        else:

            return sqrt(mean_squared_error(np.log(y_values), np.log(y_predicted)))



# Show nulls in dataset

def show_nulls_hist(dataset, string='default'):

    plt.title('Columns with nulls, {} set, descending order. (Max Id:{})'.format(string, dataset.index.max()))

    plt.ylabel('Count of nulls')

    dataset.isna().sum().sort_values(ascending=False).plot.bar()

    plt.tight_layout()

    plt.show()    

        

# Distribution plot + histogram - numerical columns

def dist_plot_norm(houses, column):

    plt.subplot(1,2,1)

    plt.title('Distribution plot against normality:')

    sns.distplot(houses[column], bins=50)

    plt.subplot(1,2,2)

    plt.title('Probability plot: Numerical column(s)')

    stats.probplot(houses[column], plot=plt)

    plt.tight_layout()

    plt.show()





# DATA IMPORTING #

##################

# Import data as DataFrames

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



# Target variable in as float64

y_train = train[target_column_name].astype('float64')



# Create a column for train and test

train['Set'] = 'train'

test[target_column_name] = np.nan # The order of assignation in code is important!

test['Set'] = 'test'



# Prepare the Sets for concatenation

if (len(train.columns) == len(test.columns)) and (train.columns.to_list()) == (test.columns.to_list()):

    print('\nNumber of columns and names in Training and Testing DataFrames names equal!\n')

else:

    print('\nCheck columns number and/or names of columns in Training and Testing DataFrames!\nTraining columns: {}\nTesting columns: {}\n Training col #: {}\n Testing col #: {}'.format(len(train.columns),

    len(test.columns), train.columns.to_list(), test.columns.to_list()))



# Default plot area

plt.rcParams['figure.figsize'] = (15, 7)        

    

# Show the null contribution of features - train

show_nulls_hist(train, string='train')

    

# Show the null contribution of feature - test 

show_nulls_hist(test, string='test')   



# Joint the data for treatment of nulls

houses = pd.concat([train, test], ignore_index=True) # From now on work with houses DataFrame





# DATA EXPLORATION AND CLEANING #

#################################

# According to "data description.txt" document, there are some categorical variables codified already, they will be spaced by xFACT to have more control on the model:

original_codified_categoricals = 'OverallQual OverallCond MSSubClass'.split()

# Same treatment in train as in test 

houses[original_codified_categoricals] = houses[original_codified_categoricals] * FACT



# Also in the same document declares that NA actually means something for certain categorical variables (i.e., 'Not have' as not having something of importance), it does not

# mean missing values! (excluding the ones that might imbalance the data)

original_NAmeaning_categoricals = 'BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2 GarageType GarageFinish GarageQual GarageCond MiscFeature FireplaceQu'.split()

houses.fillna(dict(zip(original_NAmeaning_categoricals, ['Not have' for i in range(len(original_NAmeaning_categoricals))])), inplace=True)



# Some variables are related to these above, just check easily for consistency (Garage = 'Not have', Basement = 'Not have', Fireplace = 'Not have', Miscellaneous = 'Not have'), should lead to zero

# on the following categorical to check selected: GarageQual, BsmtQual, FireplaceQu, MiscFeature

garage_NOTHAVE_numericals = 'GarageYrBlt GarageCars GarageArea'.split()

basement_NOTHAVE_numericals = 'BsmtFinSF1 BsmtFinSF2 BsmtUnfSF TotalBsmtSF BsmtFullBath BsmtHalfBath'.split()

fireplace_NOTHAVE_numericals = ['Fireplaces']

miscellaneous_NOTHAVE_numericals = ['MiscVal']



# Verify and fill all mulls with 0s accordingly

houses[houses['GarageQual'] == 'Not have'][garage_NOTHAVE_numericals].fillna(0.0, inplace=True)

houses[houses['BsmtQual'] == 'Not have'][basement_NOTHAVE_numericals].fillna(0.0, inplace=True)

houses[houses['FireplaceQu'] == 'Not have'][fireplace_NOTHAVE_numericals].fillna(0.0, inplace=True)

houses[houses['MiscFeature'] == 'Not have'][miscellaneous_NOTHAVE_numericals].fillna(0.0, inplace=True)



# Drop the 'Id' it won't be needed, and 'PoolQC', 'MiscFeature', 'Alley', 'Fence' and 'FireplaceQu'

# as they have too many NA values (600+, if maintained they will inbalance the data) - applies to train and test sets

houses.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)



# Show the null contribution of feature - full houses dataset, excluding target - Initial check

show_nulls_hist(houses.drop([target_column_name], axis=1), string='[Initial]full-no-target')



# Just in case

if (houses.drop([target_column_name], axis=1).isna().any().sum()) > 0: 

    print('Dataset still has nulls in it! (excluding target) - Initial check')

else:

    print('Dataset has not nulls in it! (excluding target) - Initial check')



# Use simple imputer to numeric variables

houses[houses.drop([target_column_name], axis=1).select_dtypes(['int64', 'float64']).columns] =  SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(houses[houses.drop([target_column_name],

                                                                                                    axis=1).select_dtypes(['int64', 'float64']).columns])



# Use simple imputer to categorical variables

# Do not process 'Set' from names

houses[houses.drop([target_column_name, 'Set'], axis=1).select_dtypes(['object']).columns] =  SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(houses[houses.drop([target_column_name, 'Set'],

                                                                                            axis=1).select_dtypes(['object']).columns])



# Show the null contribution of feature - full houses dataset, excluding target - Final check

show_nulls_hist(houses.drop([target_column_name], axis=1), string='[Final]full-no-target')



# Just in case, check again

if (houses.drop([target_column_name], axis=1).isna().any().sum()) > 0: 

    print('Dataset still has nulls in it! (excluding target) - Final check')

else:

    print('Dataset has not nulls in it! (excluding target) - Final check')



# Interesting columns that make sense to search for duplicates - just train dataset

columns_names_to_evaluate_duplication_subset = ['SalePrice', 'YearBuilt', 'YearRemodAdd', 'SaleType', 'LotFrontage', 'LotArea', 'HouseStyle', 'RoofStyle', '1stFlrSF', 'Neighborhood', 'GarageYrBlt']



# Search for duplicated values, if any and deal with them - just train dataset

duplicates = houses[houses['Set'] == 'train'].duplicated(subset = columns_names_to_evaluate_duplication_subset, keep=False)

if duplicates.sum() > 0:

    print('\nDuplicated records in Houses DataFrame found\n')

    print(houses[houses[houses['Set'] == 'train'].duplicated(subset = columns_names_to_evaluate_duplication_subset, keep=False)].sort_values(by = 'SalePrice'))

    print('\nDuplicated records in Houses DataFrame cleaned, first keep\n')

    

    # Filter duplicated and create a new clean DataFrame with inplace()

    houses[houses['Set'] == 'train'].drop_duplicates(subset = columns_names_to_evaluate_duplication_subset, keep='first', inplace=True)

else:    

    print('\nHouses DataFrame has not duplicated records in it\n')



    

# FEATURE ENGINEERING #

#######################        

# Treat object/category variables One-Hot-Encoding or Simple encoding to convert to numbers - All

object_column_names = houses.drop([target_column_name], axis=1).select_dtypes(['object']).columns.tolist()



# Drop 'Set' from list

object_column_names.remove('Set')



for column in object_columns_names:

    houses[column] = LabelEncoder().fit_transform(houses[column]) * FACT # Space the levels of encoding categoric variables in xFACT



# But in order to avoid multicollinearity the parameter drop_first=True, if we are going to use OHE

# houses = pd.get_dummies(houses, drop_first=True)  



# As all applicable columns already were encoded, drop the originals - All

houses.drop(object_column_names, axis=1, inplace=True)



# Distribution plot of the target variable

dist_plot_norm(houses[houses['Set'] == 'train'], target_column_name)



# Correlation between variables, special attention to target_column_name   

plt.title('Numeric relationships of variables')

sns.heatmap(houses[houses['Set'] == 'train'].corr(), cmap='coolwarm')

plt.tight_layout()

plt.show()



# Let's see .corr() in other way

plt.title('Numeric relationships of variables against target (target included)')

houses[houses['Set'] == 'train'].corr()[target_column_name].sort_values().plot(kind='bar')

plt.tight_layout()

plt.show()





# MODELING, ADJUSTMENT AND SELECTION ON METRICS#

################################################

# Extract the target column and prepare for modeling after treatment

# Do not process 'Set' from names

y_train = houses[houses['Set'] == 'train'][target_column_name]

X_train = houses[houses['Set'] == 'train'].drop([target_column_name, 'Set'], axis=1) # At this point all are numeric

X_test = houses[houses['Set'] == 'test'].drop([target_column_name, 'Set'], axis=1) # Testing has nulls as SalePrice



# All DataFrame

X = pd.concat([X_train, X_test], ignore_index=True)



# Fit a linear regression model for the data

model = LinearRegression()



# Fit it accordingly

model.fit(X_train, y_train)



# Obtain model predictions

y_train_predicted = model.predict(X_train)

y_test_predicted = model.predict(X_test)

y_predicted = model.predict(X)



# Some predicted values were negative when normalizing the data

# Just y_test_predicted data before submission

plt.title('Histogram of y_test_predicted')

sns.distplot(y_test_predicted, bins=50)

plt.tight_layout()

plt.show()



# Now y_train_predicted prediction histogram

plt.title('Histogram of y_train_predicted')

sns.distplot(y_train_predicted, bins=50)

plt.tight_layout()

plt.show()



# Whole data prediction histogram

plt.title('Histogram of y_predicted')

sns.distplot(y_predicted, bins=50)

plt.tight_layout()

plt.show()



# Draw an scatter plot on the predicted errors

sns.jointplot(x=y_train, y=y_train_predicted, kind='scatter', alpha=0.9)

plt.show()



# Draw an histogram of the errors to see if they are normal

plt.subplot(1,2,1)

plt.title('Histogram of errors (predicted - actual values) - first : Plain data')

sns.distplot((y_train-y_train_predicted), bins=50)

plt.subplot(1,2,2)

plt.title('Probability plot of errors:')

stats.probplot((y_train-y_train_predicted), plot=plt)

plt.tight_layout()

plt.show()



# Print the coefficients of the initial model, just to seize the feature importance

coefficient_names = X_train.columns.tolist()

coefficients = pd.DataFrame(model.coef_, coefficient_names)

coefficients.columns = ['Coefficient']

print(coefficients)

coefficients.sort_values('Coefficient').plot(kind='bar')

plt.tight_layout()

plt.show()



# Calculate the L-RMSE of the initial model

print('Logarithmic version of Root of Mean Squared Error L-RMSE (On Training data) - First Model: Plain data\n', competition_metric(y_train, y_train_predicted))



# Normalize features

# For X_train

norm = StandardScaler()

X_train_norm = norm.fit_transform(X_train)

# For X_test

X_test_norm = norm.transform(X_test)

# For X - all

X_scaled = norm.transform(X)



# Scale the features

# For X_train

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

# For X_test

X_test_scaled = scaler.transform(X_test)

# For X - all

X_scaled = scaler.transform(X)



# Make a Principal Component Analysis to improve the model

# For X_train

pca = PCA(n_components=N_PCA)

X_train_pca = pca.fit_transform(X_train_scaled)

# For X_test

X_test_pca = pca.transform(X_test_scaled)

# For X - all

X_pca = pca.transform(X_scaled)



# This data have outliers and regularization is needed, plus, the set is somewhat small let`s use RobustScaler()

# For X_train

re = RobustScaler()

X_train_re = re.fit_transform(X_train)

# For X_test

X_test_re = re.transform(X_test)

# For X - all

X_re = re.transform(X)



# Fit a linear regression model for the data

model_b = LinearRegression()



# Fit it accordingly

model_b.fit(X_train_re, y_train)



# Obtain model predictions

y_train_predicted = model_b.predict(X_train_re)

y_test_predicted = model_b.predict(X_test_re)

y_predicted = model_b.predict(X_re)



# Draw an scatter plot on the predicted errors

sns.jointplot(x=y_train, y=y_train_predicted, kind='scatter', alpha=0.9)

plt.show()



# Draw an histogram of the errors to see if they are normal

plt.subplot(1,2,1)

plt.title('Histogram of errors (predicted - actual values) - second: Robust Scaler')

sns.distplot((y_train-y_train_predicted), bins=50)

plt.subplot(1,2,2)

plt.title('Probability plot of errors:')

stats.probplot((y_train-y_train_predicted), plot=plt)

plt.tight_layout()

plt.show()



# Calculate the L-RMSE of the second model

print('Logarithmic version of Root of Mean Squared Error L-RMSE (On Training data) - Second Model:Robust Scaler\n', competition_metric(y_train, y_train_predicted))

 

#Let's try to do a TensorFlow magic here...

# Split the train data into train and test for PCA (as less memory consumption) or just scaled

X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_train_pca, y_train, test_size = T_SIZE, random_state = 7) # Trying with PCA



model_tf = Sequential()

model_tf.add(Dense(N_PCA,activation='relu'))

model_tf.add(Dense(7,activation='relu'))

model_tf.add(Dropout(0.3)) # To make this more reluctant to over fitting

model_tf.add(Dense(1, activation='linear'))



# It seems can be used 'mean_squared_logarithmic_error' directly, let's use the function defined for the competition

model_tf.compile(optimizer='Adam', loss='mean_squared_logarithmic_error')



# Adding an EarlyStopping()

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)    

      

# Fit the model

model_tf.fit(x=X_b_train, y=y_b_train.values, epochs=EPOCHS, validation_data=(X_b_test, y_b_test.values), callbacks=[early_stop])



# To see the evolution of the model's losses per epoch

pd.DataFrame(model_tf.history.history).plot()

plt.show()



# Now predict in base of the best model above the y_test_predicted

y_train_predicted = model_tf.predict(X_train_pca)

y_test_predicted = model_tf.predict(X_test_pca)

y_predicted = model_tf.predict(X_pca)



# Draw an histogram of the errors to see if they are normal

plt.title('Histogram of errors (predicted - actual values) - tf')

sns.distplot((y_train.values-y_train_predicted), bins=50)

plt.tight_layout()

plt.show()



# Calculate the L-RMSE of the TensorFlow model

print('Logarithmic version of Root of Mean Squared Error L-RMSE (On Training data) - TensorFlow Model:\n', competition_metric(y_train, y_train_predicted))

print('Explained variance score for this TensorFlow Model:\n', explained_variance_score(y_train, y_train_predicted))



#XGBoost

# Split the train data into train and test for PCA (as less memory consumption)

X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_train_pca, y_train, test_size = T_SIZE, random_state = 7) # Try with pca, re, etc



dtrain = xgb.DMatrix(X_b_train, label=y_b_train)

dtest = xgb.DMatrix(X_b_test, label=y_b_test)



# parameters to try

param_grid = {

    'colsample_bytree':[0.35,0.4,0.45],

    'gamma':[0.000],

    'min_child_weight':[1.5],

    'learning_rate':[0.1,0.15,0.2],

    'max_depth':[3],

    'n_estimators': [10000],

    'reg_alpha':[1],

    'reg_lambda':[0.05,0.1,0.15],

    'subsample':[0.6]  

}



# High computation resources demanding - handle with care!

#gs = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=param_grid, n_jobs=3, verbose=10, scoring=make_scorer(competition_metric, greater_is_better=False))

#gs.fit(X_b_train,y_b_train)

#print(gs.best_estimator_)



# After 30 min approx. I got the best parameters from the code above and gave this best estimator

# xgb.XGBRegressor(colsample_bytree=0.45, gamma=0.0, learning_rate=0.1, max_depth=3, min_child_weight=1.5, n_estimators=10000, reg_alpha=0.00001, reg_lambda=0.15, subsample=0.7)



# The I did a fine tuning of the grid to boost the performance of the model even more (around the values initially obtained, but decided duplicate the n_estimators)... and after 1 hour!!! (yes, you read it right)

# The best estimator was

best_xgb_model = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.2, gamma=0.0, gpu_id=-1, importance_type='gain',

                                  interaction_constraints=None, learning_rate=0.05, max_delta_step=0, max_depth=6, min_child_weight=1.5, monotone_constraints=None,

                                  n_estimators=10000, n_jobs=4, num_parallel_tree=1, objective='reg:squarederror', random_state=5, reg_alpha=0.9, reg_lambda=0.6, scale_pos_weight=1,

                                  seed=42)



# best_xgb_model = gs.best_estimator_ 

best_xgb_model.fit(X_train_pca, y_train) # Now estimate the full test values with the best one



# Now predict in base of the best model above the y_test_predicted

y_train_predicted = best_xgb_model.predict(X_train_pca)

y_test_predicted = best_xgb_model.predict(X_test_pca)

y_predicted = best_xgb_model.predict(X_pca)



# Calculate the L-RMSE of the Best XGB model

print('Logarithmic version of Root of Mean Squared Error L-RMSE (On Training data) - Best XGBoost Model: pca\n', competition_metric(y_train, y_train_predicted))



# Draw an histogram of the errors to see if they are normal

plt.subplot(1,2,1)

plt.title('Histogram of errors (predicted - actual values) - xgb: pca')

sns.distplot((y_train-y_train_predicted), bins=50)

plt.subplot(1,2,2)

plt.title('Probability plot of errors - xgb: pca')

stats.probplot((y_train-y_train_predicted), plot=plt)

plt.tight_layout()

plt.show()



# Draw an scatter plot on the predicted errors

sns.jointplot(x=y_train, y=y_train_predicted, kind='scatter', alpha=0.9)

plt.show()





# SUBMISSION #

##############

id_ = list(range(1461, 1461+len(y_test_predicted)))

data = pd.DataFrame(list(zip(id_, y_test_predicted)), columns=['Id', 'SalePrice'])



#Write the file

data.to_csv('submission.csv', index=False)
