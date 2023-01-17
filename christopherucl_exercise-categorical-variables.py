# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex3 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X = pd.read_csv('../input/train.csv', index_col='Id') 

X_test = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)



# To keep things simple, we'll drop columns with missing values

cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 

X.drop(cols_with_missing, axis=1, inplace=True)

X_test.drop(cols_with_missing, axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)
X_train.shape # all col with na removed. we have both numerical and categorical data
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
X_train.columns

#X_train.shape

X_train.shape
for i in X_train.columns:

    print(i)
X_train.Condition2.head()
X_train.dtypes
X_train['SaleCondition'].dtype == 'object'
categorical_col = [i for i in X_train.columns if (X_train[i].dtype == 'object')]
categorical_col
len(categorical_col)
X_train.dtypes
s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
print(len(object_cols))
X_train_copy = X_train.copy()

X_valid_copy = X_valid.copy()
X_train_copy.drop(categorical_col, axis=1, inplace=True) # drop categorica variable
X_train_copy.shape


# Fill in the lines below: drop columns in training and validation data

drop_X_train = X_train_copy

drop_X_valid = X_valid.select_dtypes(exclude=['object'])



# Check your answers

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
print("MAE from Approach 1 (Drop categorical variables):")

print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())

print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())
#step_2.a.hint()
step_2.a.solution()
# All categorical columns

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train[col]) == set(X_valid[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))

        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
label_X_valid1 = X_valid.drop(bad_label_cols, axis=1)

label_X_valid1.dtypes

label_X_valid1.shape
label_X_train1 = X_train.drop(bad_label_cols, axis=1)

label_X_train1.dtypes

label_X_train1.shape
label_X_train1.columns == label_X_valid1.columns
from sklearn.preprocessing import LabelEncoder



# Drop categorical columns that will not be encoded

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)



# Apply label encoder 

s = (label_X_train.dtypes == 'object')

object_cols = list(s[s].index)

label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col]) # Your code here

    

# Check your answer

step_2.b.check()
# Lines below will give you a hint or solution code

#step_2.b.hint()

step_2.b.solution()
print("MAE from Approach 2 (Label Encoding):") 

print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
object_nunique
object_cols
zip(object_cols, object_nunique)
d
aaa = list(zip(object_cols, object_nunique))

sorted(aaa[:], key=lambda x: x[1])
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order. sorted() is not inplace

sorted(d.items(), key=lambda x: x[1]) #lambda func tells the sorted to do it by the first index x[1]
X_train.columns
                        #FINDING THE CARDINALITY OF NEIGBOURHOOD 

#Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), bad_label_cols))

d = dict(zip(bad_label_cols, object_nunique))



# Print number of unique entries by column, in ascending order. sorted() is not inplace

sorted(d.items(), key=lambda x: x[1]) #lambda func tells the sorted to do it by the first index x[1]
# Fill in the line below: How many categorical variables in the training data

# have cardinality greater than 10?

high_cardinality_numcols = 3



# Fill in the line below: How many columns are needed to one-hot encode the 

# 'Neighborhood' variable in the training data?

num_cols_neighborhood = 25



# Check your answers

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution()
# Fill in the line below: How many entries are added to the dataset by 

# replacing the column with a one-hot encoding?

OH_entries_added = 990000



# Fill in the line below: How many entries are added to the dataset by

# replacing the column with a label encoding?

label_entries_added = 0



# Check your answers

step_3.b.check()
# Lines below will give you a hint or solution code

#step_3.b.hint()

#step_3.b.solution()
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
from sklearn.preprocessing import OneHotEncoder



# Use as many lines of code as you need!

OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)



OH_cols_train = pd.DataFrame(OHE.fit_transform(X_train[low_cardinality_cols]))

OH_cols_valid = pd.DataFrame(OHE.transform(X_valid[low_cardinality_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index



# Remove categorical columns leaving behind only numerical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features left behind in previous line

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)





# Check your answer

step_4.check()
OH_X_valid.shape
# Lines below will give you a hint or solution code

#step_4.hint()

#step_4.solution()
print("MAE from Approach 3 (One-Hot Encoding):") 

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(OH_X_train, y_train)

preds = model.predict(OH_X_valid)

#preds
# (Optional) Your code here

# Save test predictions to file

output = pd.DataFrame({'Id': OH_X_valid.index,

                       'SalePrice': preds})

output.to_csv('submission.csv', index=False)