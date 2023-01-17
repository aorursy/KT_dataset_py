# Define libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Read the data then convert to dataframe

X = pd.read_csv('../input/train.csv', index_col='Id') 



# Working on the training dataset

# We need to remove rows with null values to make sure 

# that all training targets are present

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)



# To keep things simple, we will drop the columns with null values 

# on the training features

cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 

X.drop(cols_with_missing, axis=1, inplace=True)



# Split the training data set to training and test data

# ratio would be training = 80% and test = 20%

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)



# We need a function that we could reuse 

# each time we need to return the MAE

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# Checking the first 5 rows of features for both training and validation

X_train.head()
X_valid.head()
X_train.shape
X_valid.shape
# Will drop all columns with "categorical" type of data

drop_X_train = X_train.select_dtypes(exclude="object")

drop_X_valid = X_valid.select_dtypes(exclude="object")
drop_X_train.shape
drop_X_valid.shape
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
X_train.shape
X_valid.shape
# Basically when we say label encoding, just imagine a 

# table in a database where valid values are listed on a list (consider enum)

# we will assign a unique digit for each



from sklearn.preprocessing import LabelEncoder



label_X_train_tmp = X_train.copy()

label_X_valid_tmp = X_valid.copy()

encoder = LabelEncoder()



# We need to make sure that all the list of accepted values

# are both the same on both train and valid data or else we will encounter an error

# therefore we need to know which features(labels) are unique on train data 

# and on valid data then we remove those features from the equation 



object_cols        = [col for col in label_X_train_tmp.columns if label_X_train_tmp[col].dtype == "object"]

accepted_cols      = [col for col in object_cols if set(label_X_train_tmp[col]) == set(label_X_valid_tmp[col])]

to_be_removed_cols = list(set(object_cols)-set(accepted_cols))

print(object_cols)

print(accepted_cols)

print(to_be_removed_cols)
label_X_train_tmp.shape
label_X_valid_tmp.shape
# Drop all to be removed columns

label_X_train = label_X_train_tmp.drop(to_be_removed_cols, axis=1)

label_X_valid = label_X_valid_tmp.drop(to_be_removed_cols, axis=1)
label_X_train.shape
label_X_valid.shape
# Proceed with label encoding for each, then measure MAE

# Each non numerical feature will be mapped into a unique digit

for col in object_cols:

    label_X_train[col] = encoder.fit_transform(X_train[col])

    label_X_valid[col] = encoder.fit_transform(X_valid[col])
label_X_train.shape
label_X_valid.shape
label_X_train.head()
label_X_valid.head()
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
# Just a short note, keep in mind that when we say "cardinality"

# it is a simple as the total number of unique accepted values 

# a feature has. For example, if field1 values could be yes or no, 

# then it has a cardinality of 2



# Also do some research on lambda function and map function, it is really really useful
# Cardinality IS an issue since features with very huge number of cardinality 

# could bloat your data, thus making it harder to be processed. 

# I will only one-hot encode features less than 10 cardinality



oh_X_train_tmp = X_train.copy()

oh_X_valid_tmp = X_valid.copy()



# Get all object columns 

object_cols = [col for col in oh_X_train_tmp.columns if oh_X_train_tmp[col].dtype == "object"]



# Get all features with less than 10 cardinality so that we know which features are we going to drop

low_cardinality_cols = [col for col in object_cols if oh_X_train_tmp[col].nunique() < 10]

to_be_removed_cols   = list(set(object_cols)-set(low_cardinality_cols))

print(to_be_removed_cols)
# We now know which features have more than 10 cardinality

# will drop it from the dataset

oh_X_train = oh_X_train_tmp.drop(to_be_removed_cols, axis=1)

oh_X_valid = oh_X_valid_tmp.drop(to_be_removed_cols, axis=1)
oh_X_train.shape
oh_X_valid.shape
# Will now do one-hot encoding then check MAE score

from sklearn.preprocessing import OneHotEncoder

oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)



# one-hot encode each object cols, on both training data and validation data

# need to update object_cols since we removed 3 features earlier

object_cols   = [col for col in oh_X_train.columns if oh_X_train[col].dtype == "object"]

oh_cols_train = pd.DataFrame(oh_encoder.fit_transform(oh_X_train[object_cols]))

oh_cols_valid = pd.DataFrame(oh_encoder.transform(oh_X_valid[object_cols]))
oh_cols_train.head()
oh_cols_valid.head()
# Putting back the index

oh_cols_train.index = X_train.index

oh_cols_valid.index = X_valid.index
# Now, as you see on top, we have one-hot encoded features with less than 10 cardinality

# therefore, we can now remove all object columns from the data set then replace them

# with the one-hot encoded version

oh_X_train_tmp = oh_X_train.drop(object_cols, axis=1) 

oh_X_valid_tmp = oh_X_valid.drop(object_cols, axis=1)

oh_X_train = pd.concat([oh_X_train_tmp, oh_cols_train], axis=1)

oh_X_valid = pd.concat([oh_X_valid_tmp, oh_cols_valid], axis=1)
oh_X_train.shape
oh_X_valid.shape
# measure MAE for one-hot encoding

print(score_dataset(oh_X_train, oh_X_valid, y_train, y_valid))