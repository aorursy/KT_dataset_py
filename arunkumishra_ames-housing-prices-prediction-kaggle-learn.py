import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Path of the file to read

training_data_path = '../input/train.csv'

#read training data file using pandas

home_data = pd.read_csv(training_data_path,index_col='Id')



# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path,index_col='Id')
# Print training data size, shape and dimensions

print('Shape of the training data set: ' + str(home_data.shape))



# Print test data size, shape and dimensions

print('Shape of the test data set: ' + str(test_data.shape))
home_data.head()
mismatch_type_columns = []

for i in range(0,len(home_data.dtypes)-1):

    if home_data.dtypes[i] != test_data.dtypes[i]:

        print('Training and Test data type mismatch for column {} - {}!'.format(i+1,home_data.columns[i]))

        mismatch_type_columns.append(i)

    else:

        pass
convert_dict = {'BsmtFinSF1': float, 

                'BsmtFinSF2': float,

                'BsmtUnfSF': float,

                'TotalBsmtSF': float,

                'BsmtFullBath': float,

                'BsmtHalfBath': float,

                'GarageCars': float,

                'GarageArea': float,

               } 



home_data = home_data.astype(convert_dict) 
# Create target object and call it y

y = home_data.SalePrice



# Create training object and call it X

X = home_data.loc[:, home_data.columns != 'SalePrice']
# Check if any value in SalePrice is missing

home_data.SalePrice.isnull().any()
combined_data = pd.concat([X,test_data])

combined_data.shape
numeric_columns, categorical_columns = [],[]

for column in combined_data.columns:

    if combined_data[column].dtype == 'int64' or X[column].dtype == 'float64':

        numeric_columns.append(column)

    else:

        categorical_columns.append(column)

        

# numeric_columns

print('Numeric columns: '+ str(len(numeric_columns)))



# categorical_columns

print('Categorical columns: '+ str(len(categorical_columns)))
# Get correlation of numeric columns

corr_df = home_data.corr()

#Correlation with SalePruce

cor_target = abs(corr_df['SalePrice'])



#Selecting highly correlated features

relevant_features = cor_target[cor_target>=0.1]

#relevant_features
numeric_columns = [value for value in numeric_columns if value in relevant_features.index]
# Check for missing values in numeric columns

missing_numeric_cols = combined_data[numeric_columns].isnull().any()

null_numeric_columns=combined_data[numeric_columns].columns[missing_numeric_cols]

combined_data[null_numeric_columns].isnull().sum()
replacement_values = {'LotFrontage': combined_data['LotFrontage'].mean(), 'MasVnrArea': combined_data['MasVnrArea'].mean(), 'BsmtFinSF1': combined_data['BsmtFinSF1'].mean(), 'BsmtUnfSF': combined_data['BsmtUnfSF'].mean(), 'TotalBsmtSF': combined_data['TotalBsmtSF'].mean(), 'BsmtFullBath': combined_data['BsmtFullBath'].mean(), 'GarageYrBlt':0, 'GarageCars':combined_data['GarageCars'].mean(), 'GarageArea': combined_data['GarageArea'].mean()}



X = X.fillna(value=replacement_values)

test_data = test_data.fillna(value=replacement_values)
# Check numeric columns for any missing values

print('Training : '+str(X[numeric_columns].columns[X[numeric_columns].isnull().any()]))

print('Test : '+str(test_data[numeric_columns].columns[test_data[numeric_columns].isnull().any()]))
# Running boxplot on MSZoning

fig = plt.figure(figsize=(5,5))

sns.boxplot(X['MSZoning'],y)
# Running boxplots for 42 categories

fig = plt.figure(figsize=(15,30))





for num in range(1,43):

    ax = fig.add_subplot(14,3,num)

    sns.boxplot(X[categorical_columns[num]],y , ax=ax)



plt.tight_layout()

plt.show()
categorical_columns_keep = ['Alley','Street','Condition1', 'Condition2','BldgType','RoofMatl','MasVnrType','ExterQual','ExterCond','BsmtQual','BsmtCond','Heating','CentralAir','Electrical','KitchenQual','FireplaceQu','GarageType','GarageFinish','GarageCond','PoolQC','MiscFeature','SaleType','SaleCondition','MSZoning']
# find categorical columns with missing values

missing_categorical_cols = combined_data[categorical_columns_keep].isnull().any()

null_categorical_columns=combined_data[categorical_columns_keep].columns[missing_categorical_cols]

combined_data[null_categorical_columns].isnull().sum()
replacement_values = {'Alley': 'None', 'MasVnrType': 'None', 'BsmtQual': 'None', 'BsmtCond': 'None', 'Electrical': 'Sbrkr', 'KitchenQual': 'Ex', 'FireplaceQu':'None', 'GarageType':'None', 'GarageFinish': 'None', 'GarageCond': 'None', 'PoolQC': 'None', 'MiscFeature': 'None', 'SaleType': 'None', 'MSZoning': 'RL'}



X = X.fillna(value=replacement_values)

test_data = test_data.fillna(value=replacement_values)
# Check categorical columns for any missing values

print('Training : '+str(X[categorical_columns_keep].columns[X[categorical_columns_keep].isnull().any()]))

print('Test : '+str(test_data[categorical_columns_keep].columns[test_data[categorical_columns_keep].isnull().any()]))
# Make copy to avoid changing original data 

label_X = X.copy()

label_test_X = test_data.copy()

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()



# Source: https://www.kaggle.com/gowrishankarin/learn-ml-ml101-rank-4500-to-450-in-a-day

failed_features = []

for col in categorical_columns_keep:

    try:

        label_X[col] = label_encoder.fit_transform(label_X[col])

        label_test_X[col] = label_encoder.transform(label_test_X[col])

    except:

        failed_features.append(col)

        

print('Failed features: '+str(failed_features))

for col in failed_features:

    categorical_columns_keep.remove(col)
features = numeric_columns + categorical_columns_keep



label_X = label_X[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(label_X, y, random_state=1)




# Define the model. Set random_state to 1

estimators = np.arange(1,500,100)

maes = []

for est in estimators:

    rf_model = RandomForestRegressor(random_state=1,n_estimators=est)

    rf_model.fit(train_X, train_y)

    rf_val_predictions = rf_model.predict(val_X)

    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    maes.append(rf_val_mae)

    print('.', end =" ")

    #print("Validation MAE for Random Forest Model: {:,.0f} for # estimators: {}".format(rf_val_mae,est))



plt.plot(estimators,maes,'bo-')

plt.xlabel('#estimators')

plt.ylabel('Mean Absolute Error')

plt.title('MAE vs no. of estimators for the model')

plt.show()
model_xgb = XGBRegressor(n_estimators=111,random_state=1)

model_xgb.fit(train_X, train_y)

predictions = model_xgb.predict(val_X)



print("Mean Absolute Error: " + str(mean_absolute_error(predictions, val_y)))
label_test_X = label_test_X[features]



# make predictions which we will submit. We use the XGBoost model. 

test_preds = model_xgb.predict(label_test_X)
output = pd.DataFrame({'Id': test_data.index,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)