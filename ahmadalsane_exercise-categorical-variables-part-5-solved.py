cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)
X_test_cols_with_missing = [col for col in X_test.columns if X_test[col].isnull().any()]
print(len(X_test_cols_with_missing))
print(X_test_cols_with_missing)
X_test_obj_cols_with_missing = [col for col in X_test_cols_with_missing if X_test[col].dtype == "object"]
X_test_num_cols_with_missing = [col for col in X_test_cols_with_missing if X_test[col].dtype != "object"]
print('number of object cols with missing values in X_test: ',len(X_test_obj_cols_with_missing))
print('number of numerical cols with missing values in X_test: ',len(X_test_num_cols_with_missing))
print('object cols with missing values in X_test: ',X_test_obj_cols_with_missing)
print('numerical cols with missing values in X_test: ',X_test_num_cols_with_missing)
# find the sum fo missing values for each col
X_test_missing_val_count_by_column = (X_test.isnull().sum())
print(X_test_missing_val_count_by_column[X_test_missing_val_count_by_column > 0])

#find the count of each value in each col using value_counts() which will show the count of each value occured in specific column
X_test['MSZoning'].value_counts()
X_test['MSZoning'].fillna(value='RL', inplace=True)
X_test['MSZoning'].fillna(value='RL', inplace=True)
X_test['Utilities'].fillna(value='AllPub', inplace=True)
X_test['Exterior1st'].fillna(value='VinylSd', inplace=True)
X_test['Exterior2nd'].fillna(value='VinylSd', inplace=True)
X_test['BsmtFinSF1'].fillna(value=0.0, inplace=True)
X_test['BsmtFinSF2'].fillna(value=0.0, inplace=True)
X_test['BsmtUnfSF'].fillna(value=0.0, inplace=True)
X_test['TotalBsmtSF'].fillna(value=0.0, inplace=True)
X_test['BsmtFullBath'].fillna(value=0.0, inplace=True)
X_test['BsmtHalfBath'].fillna(value=0.0, inplace=True)
X_test['KitchenQual'].fillna(value='TA', inplace=True)
X_test['Functional'].fillna(value='Typ', inplace=True)
X_test['GarageCars'].fillna(value=2.0, inplace=True)
X_test['GarageArea'].fillna(value=0.0, inplace=True)
X_test['SaleType'].fillna(value='WD', inplace=True)
object_cols = [col for col in X_test.columns if X_test[col].dtype == "object"]

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_test[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(OH_X_train, y_train)
preds_test = model.predict(OH_X_test)
#and finally create the output

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)