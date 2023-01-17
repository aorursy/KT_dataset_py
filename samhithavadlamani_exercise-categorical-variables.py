# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex3 import *

print("Setup Complete")
#RandomForest

import pandas as pd

pd.set_option('display.max_rows', None)

import numpy as np

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer





# READ THE DATA

X_train = pd.read_csv('../input/train.csv', index_col='Id') 

X_test = pd.read_csv('../input/test.csv', index_col='Id')







#EXTRACT LABELS, FULL=TRAIN+TEST

labels_train=X_train["SalePrice"]

X_train=X_train.drop(['SalePrice'], axis=1)

X_full=pd.concat([X_train,X_test])

#### X_train=1460, X_test=1459, X_full=2919



#HANDLING NULL VALUES: DROP AND IMPUTE

# drop columns with nan > 50% 

cols_with_missing = [col for col in X_full.columns if X_full[col].isnull().any()] 

null_col=X_full[cols_with_missing].isnull().sum()/np.shape(X_full)[0] #calculate proportion of null values in each nulls-containing column

dict(null_col) #converting it into dictionary

d=dict(filter(lambda i: i[1] >0.5, null_col.items())) #filtered dictionary with values > 0.5

cols_to_drop=list(d.keys()) #extracting corresponding column names

X_full.drop(cols_to_drop, axis=1, inplace=True) #drop the columns with null values > 50%



#impute the other numeric columns TRAIN

cols_to_impute=list(set(cols_with_missing)-set(cols_to_drop))

# extracting only numeric columns to impute

cols_to_impute_num = [cname for cname in cols_to_impute if X_full[cname].dtype in ['int64', 'float64']]

#cols_to_impute_num=[col for col in X_full[cols_to_impute].loc[:, X_full.dtype in ['int64', 'float64']].columns]

#imputing categoric columns

# extracting only categoric columns to impute

cols_to_impute_cat=[col for col in X_full[cols_to_impute].loc[:, X_full.dtypes == np.object].columns]



#split full data back to train and test

X_train=X_full.iloc[0:1460,:]

X_test=X_full.iloc[1460::,:]





#split training data into train and validation

x_train, x_valid, y_train, y_valid = train_test_split(X_train, labels_train,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(strategy='most_frequent')

imputed_x_train = pd.DataFrame(my_imputer.fit_transform(x_train[cols_to_impute_num+cols_to_impute_cat]))

imputed_x_valid = pd.DataFrame(my_imputer.transform(x_valid[cols_to_impute_num+cols_to_impute_cat]))

imputed_X_test = pd.DataFrame(my_imputer.transform(X_test[cols_to_impute_num+cols_to_impute_cat]))



imputed_x_train.index = x_train[cols_to_impute_num+cols_to_impute_cat].index

imputed_x_valid.index = x_valid[cols_to_impute_num+cols_to_impute_cat].index

imputed_X_test.index = X_test[cols_to_impute_num+cols_to_impute_cat].index





# Remove original columns and replace with imputed

x_train = x_train.drop(cols_to_impute_cat+cols_to_impute_num, axis=1)

x_valid = x_valid.drop(cols_to_impute_cat+cols_to_impute_num, axis=1)

X_test = X_test.drop(cols_to_impute_num+cols_to_impute_cat, axis=1)



x_train = pd.concat([x_train,imputed_x_train], axis=1)

x_valid = pd.concat([x_valid,imputed_x_valid], axis=1)

X_test = pd.concat([X_test,imputed_X_test], axis=1)



print(np.shape(x_train))



#CONVERTING CATEGORICAL TO NUMERIC - OHE

from sklearn.preprocessing import OneHotEncoder

# Columns that will be one-hot encoded: low-cardinal

object_cols = [col for col in x_train.columns if x_train[col].dtypes == "object"]

low_cardinality_cols = [col for col in x_train.columns if x_train[col].nunique() < 10]



#drop high-cardinal columns

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

x_train.drop(high_cardinality_cols, axis=1)

x_valid.drop(high_cardinality_cols, axis=1)

X_test.drop(high_cardinality_cols, axis=1)



# Produces 1,0 data columns corresponding to all the unique categorical entries in low-cardinal columns list

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[low_cardinality_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(x_valid[low_cardinality_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))





#OH encoding removes index in the data set. Putting the index back again

OH_cols_train.index = x_train.index

OH_cols_valid.index = x_valid.index

OH_cols_test.index = X_test.index





# Remove categorical columns (will replace with one-hot encoding)

num_X_train = x_train.drop(object_cols, axis=1)

num_X_valid = x_valid.drop(object_cols, axis=1)

num_X_test = X_test.drop(object_cols, axis=1)



# Add one-hot encoded columns to the original data

x_train = pd.concat([num_X_train, OH_cols_train], axis=1)

x_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

X_test = pd.concat([num_X_test, OH_cols_test], axis=1)





from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)



print("MAE:", score_dataset(x_train, x_valid, y_train, y_valid))



model = RandomForestRegressor(n_estimators=100, random_state=0)

#model.fit(pd.concat([x_train, x_valid], axis=0), pd.concat([y_train, y_valid], axis=0))

model.fit(x_train, y_train)



#feature importance

importance = model.feature_importances_ 

'''# summarize feature importance

for i,v in enumerate(importance):

    print("feature: %d, score: %0.5f" %(i,v))

pyplot.bar([x for x in range(len(importance))], importance)

pyplot.show()'''

index=np.argsort(importance)[::-1]

imp=np.sort(importance)[::-1]

for i in range(0, len(imp)):

    print("%d, feature: %d, score:%0.5f" %(i+1, index[i],imp[i]))

preds = model.predict(X_test)

subm_df = pd.read_csv('../input/sample_submission.csv')

subm_df['SalePrice'] = preds

subm_df.to_csv('Bakaito_Submission.csv', index=False)

print('Completed')




