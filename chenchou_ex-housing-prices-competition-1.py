import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from learntools.core import *



X = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

y = X.SalePrice

X_test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')
features = ['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt',

            'YearRemodAdd','1stFlrSF','2ndFlrSF','GrLivArea','FullBath',

            'HalfBath','BedroomAbvGr','TotRmsAbvGrd','Fireplaces']

X_features = X[features].copy()



# Split training/validation sets with certain features

X_tra, X_val, y_tra, y_val = train_test_split(X_features, y, random_state=1)
# Various models

md1 = DecisionTreeRegressor(random_state=1)

md2 = DecisionTreeRegressor(random_state=1, max_leaf_nodes=100)

md3 = RandomForestRegressor(random_state=1)

md4 = RandomForestRegressor(random_state=0, n_estimators=100, criterion='mae')

md5 = RandomForestRegressor(random_state=0, n_estimators=100)



models = [md1, md2, md3, md4, md5]



def score_model(model, X_t, X_v, y_t, y_v):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i], X_tra, X_val, y_tra, y_val)

    print("Model %d: %d" % (i+1, mae))
X_no_missing_y = X.dropna(axis=0, subset=['SalePrice'])

X.drop(['SalePrice'], axis=1, inplace=True)



# Numerical only

X_full_numerical = X.select_dtypes(exclude=['object'])

#X_test_numerical = X_test.select_dtypes(exclude=['object'])



X_trainN, X_validN, y_trainN, y_validN = train_test_split(X_full_numerical, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)



# Drop columns with missing values

cols_with_missing = [col for col in X_trainN.columns

                     if X_trainN[col].isnull().any()]



reduced_X_train = X_trainN.drop(cols_with_missing, axis=1)

reduced_X_valid = X_validN.drop(cols_with_missing, axis=1)

print("Dropping columns with missing values:", score_model(md5, reduced_X_train, reduced_X_valid, y_trainN, y_validN))



# Imputation with 'median' strategy

final_imputer = SimpleImputer(strategy='median')

final_X_train = pd.DataFrame(final_imputer.fit_transform(X_trainN))

final_X_valid = pd.DataFrame(final_imputer.transform(X_validN))

final_X_train.columns = X_trainN.columns

final_X_valid.columns = X_validN.columns



print("Imputation with 'median':", score_model(md5, final_X_train, final_X_valid, y_trainN, y_validN))



#X_test_numerical = X_test.select_dtypes(exclude=['object'])

#final_X_test = pd.DataFrame(final_imputer.transform(X_test_numerical))

#test_preds = md5.predict(final_X_test)
# Drop columns with missing values

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]

X_no_na = X.drop(cols_with_missing, axis=1)

X_test.drop(cols_with_missing, axis=1, inplace=True)



X_train, X_valid, y_train, y_valid = train_test_split(X_no_na, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)



# Categorical only

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



# Columns can be label encoded

good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]

bad_label_cols = list(set(object_cols)-set(good_label_cols))



# Drop categorical columns that will not be encoded

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)



# Apply label encoder 

label_encoder = LabelEncoder()

for col in good_label_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])

    

print("Label Encoding:", score_model(md5, label_X_train, label_X_valid, y_train, y_valid))
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]



# Columns that will be dropped from the dataset

#high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))



OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)



OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)



print("One-hot Encoding:", score_model(md5, OH_X_train, OH_X_valid, y_train, y_valid))
# Predict One-hot Encoding 1 with 'ffill' method

object_cols = [col for col in X_test.columns if X_test[col].dtype == "object"]

low_cardinality_cols = [col for col in object_cols if X_test[col].nunique() < 10]



X_test = X_test.fillna(method='ffill')

# Alternative of imputation

#my_imputer = SimpleImputer(strategy='most_frequent')

#imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))

#imputed_X_test.columns = X_test.columns



OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))

OH_cols_test.index = X_test.index

num_X_test = X_test.drop(object_cols, axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)



model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(OH_X_train, y_train)



preds_test = model.predict(OH_X_test)
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)