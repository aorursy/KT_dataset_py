# Linear algebra

import numpy as np



# Data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd



# Import XGB estimator

from xgboost.sklearn import XGBRegressor



# Metrics for root mean squared error

from sklearn.metrics import mean_squared_error

from math import sqrt
# Load training data

df_train = pd.read_csv('../input/train.csv')



# Load test data

df_test = pd.read_csv('../input/test.csv')
# Print data set info

df_train.info()

df_train.columns
# Check for missing values

# hasNull = np.sum(df_train.isnull()) > 0; print(hasNull.values)

# hasNull = np.sum(df_test.isnull()) > 0; print(hasNull.values)



#find missing values

for c in list(df_train.columns):

	total=0

	count=0

	found_flag=0

	for t in df_train[:][c].values:

		total+=1 

		if pd.isnull(t) : 

			if count==0:

				print("Found in : ",c, end="	");

				found_flag=1

			count+=1;

	if found_flag==1: print(count,"/",total,"\t","Percentage_missing= ",(count/total)*100)



# dropping those which are missing 70% of the time or more

drop_list=['Alley','PoolQC','Fence','MiscFeature']

for c in drop_list:

    df_train.drop(c, axis=1,inplace=True)

df_train.info()



#df_test

drop_list=['Alley','PoolQC','Fence','MiscFeature']

for c in drop_list:

    df_test.drop(c, axis=1,inplace=True)

df_test.info()
# Impute missing numerical values

# df_train.fillna(df_train.mean(), inplace=True);

# df_test.fillna(df_test.mean(), inplace=True);

import pandas as pd

import numpy as np

from sklearn.base import TransformerMixin



VERBOSE=True



class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 

        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],index=X.columns)

        #self.fill = pd.Series([ X[c].mean() if X[c].dtype != np.dtype('O') else np.nan for c in X],index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.fill)



df_train = DataFrameImputer().fit_transform(df_train)

df_train.info()



df_test = DataFrameImputer().fit_transform(df_test)

df_test.info()
# # Separate categorical data into different features

# df_train = pd.get_dummies(df_train);

# df_test = pd.get_dummies(df_test);



# # Find columns that are not present on the test data set

# # (since the test data set does not contain some values of the training data set)

# diff_columns = df_train.columns.difference(df_test.columns);



# # Add such columns to the test data set, with value 0

# df_test = df_test.join(pd.DataFrame(

#     0, 

#     index=df_test.index, 

#     columns=diff_columns));



# # Rearrange columns (or the features will be out of order in test and train sets)

# df_train = df_train.reindex_axis(sorted(df_train.columns), axis=1)

# df_test = df_test.reindex_axis(sorted(df_test.columns), axis=1)



import pandas as pd

import numpy as np

from sklearn.base import TransformerMixin



cat_cols=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Heating', 'HeatingQC', 'CentralAir', 'Electrical','KitchenQual','Functional','FireplaceQu', 'GarageType','GarageFinish','GarageQual', 'GarageCond', 'PavedDrive','SaleType','SaleCondition']

for c in cat_cols:

    df_train[c] = df_train[c].astype('category')

    df_test[c] = df_test[c].astype('category')

df_train[cat_cols] = df_train[cat_cols].apply(lambda x: x.cat.codes)

df_test[cat_cols] = df_test[cat_cols].apply(lambda x: x.cat.codes)
df_train.head()
#drop ids

train_ids=df_train[:]['Id']

df_train.drop('Id', axis=1,inplace=True)



test_ids=df_test[:]['Id']

df_test.drop('Id', axis=1,inplace=True)
#keeping values positive

df_train=(df_train - df_train.min()) #/ df_train.std()#(df_train.max() - df_train.min())

df_test=(df_test - df_test.min()) #/ df_test.std()#(df_test.max() - df_test.min())
# Select target

y_train = df_train['SalePrice'].values



# Drop target from training data set

df_train = df_train.drop('SalePrice', 1)



# Drop target from test data set (it was added in the previous step)

# df_test = df_test.drop('SalePrice', 1)



# Select all features (they are all numeric after the previous step)

x_train = df_train.values

x_test = df_test.values
# Initialize model

model = XGBRegressor()                  



# Fit the model on our data

model.fit(x_train, y_train)
# Predict training set

y_pred = model.predict(x_train)



# Print RMSE

print(sqrt(mean_squared_error(y_train, y_pred)))

# Print RMSE(log(y_train),log(y_pred))

#print(sqrt(mean_squared_error(np.log(y_train), np.log(y_pred))))





# Predict test set

y_pred = model.predict(x_test)
# Create empty submission dataframe

sub = pd.DataFrame()



# Insert ID and Predictions into dataframe

sub['Id'] = test_ids #df_test['Id']

sub['SalePrice'] = y_pred

print(sub.to_csv)

# Output submission file

sub.to_csv('submission.csv',index=False)