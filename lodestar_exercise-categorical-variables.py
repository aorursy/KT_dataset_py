# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex3 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 100)



# Read the data

X = pd.read_csv('../input/train.csv', index_col='Id') 

X_test = pd.read_csv('../input/test.csv', index_col='Id')



print("Right after files input: Shape of X", X.shape, "Shape of X_test", X_test.shape)



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice



print("Shape of X after dropping rows with missing target :", X.shape)

X.drop(['SalePrice'], axis=1, inplace=True)

print("Shape of X after dropping SalePrice column :", X.shape)



#LodeStar: Before we remove any missing data, let's get assessment for how many rows & columns are missing data 

#How many rows missing data in X ? NOTE: Code more complicated here than for columns !! 

rows_missing_X = [row for row in range(len(X.index)) if X.iloc[row].isnull().any()] 

print("No. of rows with missing data in X :", len(rows_missing_X))



#How many rows missing data in X_test ? NOTE: Code more complicated here than for columns !! 

rows_missing_Xtest = [row for row in range(len(X_test.index)) if X_test.iloc[row].isnull().any()] 

print("No. of rows with missing data in X_test :", len(rows_missing_Xtest))



# How many columns missing data in X ? 

cols_missing_X = [col for col in X.columns if X[col].isnull().any()] 

print("No. of columns with missing data in X :", len(cols_missing_X))

print(cols_missing_X) 



#LodeStar: I'm doing this to find more about X_test, will help in use of X_test  

cols_missing_Xtest = [col for col in X_test.columns if X_test[col].isnull().any()] 

print("No. of columns with missing data in X_test :", len(cols_missing_Xtest))

print(cols_missing_Xtest) 



print("Columns missing in X but not X_test :", set(cols_missing_X) - set(cols_missing_Xtest) ) 



print("Columns missing in X_test but not X :", set(cols_missing_Xtest) - set(cols_missing_X)) 

 

# To keep things simple, we'll drop columns with missing values in X, for both X and X_test (IMPORTANT: keep track) 

X.drop(cols_missing_X, axis=1, inplace=True)

X_test.drop(cols_missing_X, axis=1, inplace=True) 



#Now rows missing data in X ? NOTE: Code more complicated here than for columns !! 

rows_missing_X = [row for row in range(len(X.index)) if X.iloc[row].isnull().any()] 

print("No. of rows with missing data in X, after removing missing columns :", len(rows_missing_X))



#Now rows missing data in X_test ? NOTE: Code more complicated here than for columns !! 

rows_missing_Xtest = [row for row in range(len(X_test.index)) if X_test.iloc[row].isnull().any()] 

print("No. of rows with missing data in X_test, after removing missing columns :", len(rows_missing_Xtest)) 



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)



print("Shape of X", X.shape, "Shape of X_test", X_test.shape)

print("Shape of X_train", X_train.shape, "Shape of X_valid", X_valid.shape, "Shape of y_train", y_train.shape, "Shape of y_valid", y_valid.shape)
X_train.head()
X_valid.head()
X_test.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# Fill in the lines below: drop columns with Categorical Data in training and validation data

drop_X_train = X_train.select_dtypes(exclude=['object']) 

drop_X_valid = X_valid.select_dtypes(exclude=['object']) 



# Check your answers

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
print("MAE from Approach 1 (Drop categorical variables):")

print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique(), X_train['Condition2'].nunique())

print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique(), X_valid['Condition2'].nunique())
#step_2.a.hint()
#step_2.a.solution()
# All columns with Categorical data 

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"] 



# LodeStar: Details about the columns with Categorical data 

print("Total no. of columns with Categorical data in X_train :", len(object_cols))

#print(object_cols) 



#LodeStar: Just want to see what the sets of values are and how they compare. 

#for col in object_cols:

    #if set(X_train[col]) == set(X_valid[col]): 

        #print(set(X_train[col]), "AND", set(X_valid[col])) 



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train[col]) == set(X_valid[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))



#LodeStar: good_label_cols + bad_label_cols = object_cols 

print("good_label_cols :", len(good_label_cols), "bad_label_cols :", len(bad_label_cols))

        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)



from sklearn.preprocessing import LabelEncoder



#LodeStar: First make copy to avoid changing original data 

X_train_copy = X_train.copy() 

X_valid_copy = X_valid.copy() 



# Drop categorical columns that will not be encoded

label_X_train = X_train_copy.drop(bad_label_cols, axis=1)

label_X_valid = X_valid_copy.drop(bad_label_cols, axis=1)



# Apply label encoder. Your code goes here

label_encoder = LabelEncoder() 

for col in good_label_cols: 

    label_X_train[col] = label_encoder.fit_transform(X_train_copy[col]) 

    label_X_valid[col] = label_encoder.transform(X_valid_copy[col]) 

    

    

# Check your answer

step_2.b.check()
# Lines below will give you a hint or solution code

#step_2.b.hint()

#step_2.b.solution()
print("MAE from Approach 2 (Label Encoding):") 

print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
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

OH_entries_added = 990000 #(i.e. 10000 * 100; 100 columns for each entry of the categorical column, repeated for 10000 rows !) 

                          # but the original column will disappear, hence 1000000 - 10000 = 990000 ! 



# Fill in the line below: How many entries are added to the dataset by

# replacing the column with a label encoding?

label_entries_added = 0 #(the categorical column is replaced by another column which has numbers instead of text data !)



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
from sklearn.preprocessing import OneHotEncoder



# Use as many lines of code as you need!



#Making copies of the original datasets before changing them 

X_train_plus = X_train.copy() 

X_valid_plus = X_valid.copy() 



#X_test_plus = X_test.copy() 



#LodeStar: Checking shape of training and validation datasets before applying One-Hot Encoding   

print("Shape of X_train :", X_train_plus.shape, "Shape of X_valid :", X_valid_plus.shape)



#Applying some settings to the OneHotEncoder 

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)



cols_missing_Xtrain = [col for col in X_train.columns 

                         if X_train[col].isnull().any()]



print("Columns missing data in X_train, right before fit_transform:", len(cols_missing_Xtrain)) 



#for col in low_cardinality_cols: 

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_plus[low_cardinality_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid_plus[low_cardinality_cols])) 



print("After OH-Encoding :: Shape of OH_cols_train :", OH_cols_train.shape, "Shape of OH_cols_valid :", OH_cols_valid.shape)



#OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test_plus[low_cardinality_cols])) 



#One-Hot Encoding removes index so putting it back 

OH_cols_train.index = X_train_plus.index 

OH_cols_valid.index = X_valid_plus.index 



#Remove ALL Categorical columns, they will be replaced with One-Hot Encoding 

num_X_train = X_train_plus.drop(object_cols, axis=1) 

num_X_valid = X_valid_plus.drop(object_cols, axis=1) 



print("After removing ALL categorical columns : Shape of X_train :", num_X_train.shape, "Shape of X_valid :", num_X_valid.shape)

#print((OH_cols_train.shape), (num_X_train.shape))



#Now add the One-Hot Encoded columns to the rest of the DataFrame (that has been stripped off its categorical data and contains numerical features only) 

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1) 



print("After concatenating the One-Hot columns to stripped dataset : Shape of X_train :", OH_X_train.shape, "Shape of X_valid :", OH_X_valid.shape) 



# Check your answer

step_4.check() 



#OH_X_train.head()
# Lines below will give you a hint or solution code

#step_4.hint()

#step_4.solution()
print("MAE from Approach 3 (One-Hot Encoding):") 

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
X_test.head()
#LodeStar: Making a copy before changing the structure 

X_test_plus = X_test.copy() 



#LodeStar: Checking X_test dataset 

print("Before One-Hot Encoding acts on X_test, its shape :", X_test_plus.shape) 





#LodeStar: Finding total number of columns with Categorical data in X_test 

object_cols = [col for col in X_test_plus.columns if X_test_plus[col].dtype == "object"] 



print("Total no. of columns with Categorical data in X_test NOW ", len(object_cols)) 
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_test_plus[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_test_plus[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)



# (Optional) PREPROCESS THE TEST DATA THEN USE MODEL TO GENERATE PREDICTIONS 



from sklearn.impute import SimpleImputer 



#LodeStar: Finding columns with missing data in X_test again. Remember only columns relevant to X were removed earlier ! 

#LodeStar: Column "Electrical" was not missing in X_test, but was removed bcoz of X, hence 19 (missing in X & Xtest) + 15 (missing only in Xtest) - 1 = 33 (total missing in Xtest before removal)

cols_missing_Xtest = [col for col in X_test_plus.columns 

                        if X_test_plus[col].isnull().any()] 



print("Columns missing data in X_test BEFORE major preprocessing :", len(cols_missing_Xtest)) 

#print(cols_missing_Xtest)



#How many rows missing data in X_test ? NOTE: Code more complicated here than for columns !! 

rows_missing_Xtest = [row for row in range(len(X_test_plus.index)) if X_test_plus.iloc[row].isnull().any()] 

print("Rows missing data in X_test BEFORE major preprocessing :", len(rows_missing_Xtest))

#print(rows_missing_Xtest) 



#LodeStar: Remove ALL Categorical columns from X_test_plus. They will be replaced after numerical Imputation  

#LodeStar: Store ALL the Categorical columns in another DataFrame (WILL BE USEFUL LATER !) 

num_X_test = X_test_plus.drop(object_cols, axis=1) 

categ_X_test = X_test_plus.select_dtypes(include=['object']) 



print("After removing ALL categorical columns : Shape of num_X_test :", num_X_test.shape)



#LodeStar: Now using the "Median" Imputation Strategy for filling in missing NUMERICAL data in columns 

app3_imputer = SimpleImputer(strategy='median') 

imputed_X_test = pd.DataFrame(app3_imputer.fit_transform(num_X_test)) 



#LodeStar: Imputation removes column names. So putting them back

imputed_X_test.columns = num_X_test.columns 

imputed_X_test.index = num_X_test.index 

#imputed_X_test.columns = X_test_plus.columns 



#LodeStar: Now checking for missing data in columns for imputed_X_test (NUMERICAL part)  

cols_missing_Xtest = [col for col in imputed_X_test.columns 

                         if imputed_X_test[col].isnull().any()]



print("Columns missing numerical data in imputed_X_test :", len(cols_missing_Xtest))



#LodeStar: Now using the "most_frequent" Imputation Strategy for filling in missing CATEGORICAL data in columns 

app4_imputer = SimpleImputer(strategy='most_frequent') 

imputedCateg_X_test = pd.DataFrame(app4_imputer.fit_transform(categ_X_test)) 



#LodeStar: Imputation removes column names. So putting them back 

imputedCateg_X_test.columns = categ_X_test.columns 

imputedCateg_X_test.index = categ_X_test.index 



#LodeStar: Now checking for missing data in columns for imputedCateg_X_test (CATEGORICAL part)

cols_missing_Xtest = [col for col in imputedCateg_X_test.columns 

                         if imputedCateg_X_test[col].isnull().any()]



print("Columns missing categorical data in imputedCateg_X_test :", len(cols_missing_Xtest)) 



#LodeStar: Concatenating the numerical and Categorical parts for the full dataset ! 

joined_X_test = pd.concat([imputed_X_test, imputedCateg_X_test], axis=1) 



print("Verifying Concatenation:: Imputed Shape ", (imputed_X_test.shape), " + Categ Shape ", (imputedCateg_X_test.shape), " = Joined Shape :", joined_X_test.shape)



cols_missing_joinedXtest = [col for col in joined_X_test.columns 

                          if joined_X_test[col].isnull().any()]



print("Columns missing data in joined_X_test :", cols_missing_joinedXtest)



#LodeStar: Finding out how many missing values are there. If there are a few (< 1%) we can drop those specific rows

#cols_missing_sum = (joined_X_test.isnull().sum()) 

#print(cols_missing_sum[cols_missing_sum > 0]) 



#nan_rows = joined_X_test[joined_X_test['MSZoning'].isnull()]

#print((nan_rows))



#Now rows missing data in X_test ? NOTE: Code more complicated here than for columns !! 

rows_missing_joinedXtest = [row for row in range(len(joined_X_test.index)) if joined_X_test.iloc[row].isnull().any()] 

print("Rows missing data in joined_X_test :", rows_missing_joinedXtest) 



#LodeStar: Find total missing data in X_test 

total_missing_joinedXtest = joined_X_test.isnull().sum().sum()

print("Total no. of missing data in joined_X_test :", total_missing_joinedXtest) 



#LodeStar: Tried the below dropping of few rows but the submission requires all rows to be present. Hence abandoned it. 

#LodeStar: Dropping the few rows that have missing data (9 rows of missing data = 9/1459 < 1%) 

#joined_X_test.dropna(axis=0, inplace=True)

#print("Shape of joined_X_test after dropping the NA/NAN rows", joined_X_test.shape) 



#LodeStar: Now try One-Hot Encoding. Applying some settings to the OneHotEncoder 

# >>>>> DO NOT INSTANTIATE OHE AGAIN ! >>>>> OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) 

# The OH_encoder instance has already been declared and fitted with the training dataset. Use that fit for the test data too !   



#LodeStar: Applying OH Encoding to the model X_test for fitting and transforming it 

# >>>>> NEVER USE fit_transform on test dataset, it should use the fitting from the training (model)! >>>>> 

OH_cols_test = pd.DataFrame(OH_encoder.transform(joined_X_test[low_cardinality_cols]))



#One-Hot Encoding removes index so putting it back 

OH_cols_test.index = joined_X_test.index



print("After OH_Encoding :: Shape of OH_cols_test :", OH_cols_test.shape)



#Remove ALL Categorical columns, they will be replaced with One-Hot Encoding 

num_joined_X_test = joined_X_test.drop(object_cols, axis=1) 



print("After removing ALL categorical columns : Shape of joined_X_test :", num_joined_X_test.shape)



#Now add the One-Hot Encoded columns to the rest of the DataFrame (that has been stripped off its categorical data and contains numerical features only) 

OH_X_test = pd.concat([num_joined_X_test, OH_cols_test], axis=1)



print("After concatenating the One-Hot columns to stripped dataset : Shape of joined_X_test :", OH_X_test.shape) 
# Define and fit model

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(OH_X_train, y_train)



# Get validation predictions and MAE

preds_test = model.predict(OH_X_test)

print("Shape of preds_test :", preds_test.shape)

print("MAE (Your approach): One-Hot Encoding to Test data")

#print(mean_absolute_error(y_valid, preds_valid))
joined_X_test.tail()
imputed_X_test.head()
categ_X_test.head()
X_test_plus.head()
preds_test.view()
OH_X_test.tail()
# Save test predictions to file

#LodeStar: This cell DOES need change in the assignment of "Id". Id should be assigned according to the original X_test file, NOT according to final_X_test.

output = pd.DataFrame({'Id': OH_X_test.index, 

                      'SalePrice': preds_test}) 

output.to_csv('submission.csv', index=False)