# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def write_submissions(file_name, test_df, predictions):

    test_df.Id = test_df.Id.astype('int32')



    output = pd.DataFrame({

        'Id': test_df.Id, 'SalePrice': predictions

    })

    output.to_csv(file_name, index=False)

    

def get_categorical_columns(data_df):

    return list(data_df.select_dtypes(include=['category', 'object']))





def get_numeric_columns(data_df):

    return list(data_df.select_dtypes(exclude=['category', 'object']))





def read_train_test_data():

    train_df = pd.read_csv('../input/train.csv')

    test_df = pd.read_csv('../input/test.csv')

    

    print("Shape of Train Data: " + str(train_df.shape))

    print("Shape of Test Data: " + str(test_df.shape))

    

    categorical_columns = get_categorical_columns(train_df)

    print("No of Categorical Columns: " + str(len(categorical_columns)))

    numeric_columns = get_numeric_columns(train_df)

    print("No of Numeric Columns: " + str(len(numeric_columns)))



    return train_df, test_df
train_df, test_df = read_train_test_data()
numeric_columns = get_numeric_columns(train_df)

train_df[numeric_columns].describe()
categorical_columns = get_categorical_columns(train_df)

train_df[categorical_columns].describe()
from sklearn.tree import DecisionTreeRegressor

    

def simple_model(X, y, X_test):

    model = DecisionTreeRegressor(random_state=1)

    predictions = None

    try:

        model.fit(X, y)

        predictions = model.predict(X_test)

    except Exception as exception:

        print(exception)

        pass



    return predictions
### Simple Model 

X = train_df[numeric_columns].copy()

X.drop(columns=["SalePrice"], axis=1, inplace=True)

y = train_df.SalePrice

print("Shape of Modified Train Data: " + str(X.shape))

simple_model(X, y, test_df[X.columns])
X = X.dropna(axis=1)

print(X.shape)

simple_model(X, y, test_df[X.columns])
cols_with_missing_values = [col for col in X.columns if X[col].isnull().any()]

X = X.drop(cols_with_missing_values, axis=1)

print(X.shape)

simple_model(X, y, test_df[X.columns])
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

imputed_numeric_X = imputer.fit_transform(X)



imputed_numeric_test = imputer.transform(test_df[X.columns])
predictions = simple_model(imputed_numeric_X, y, imputed_numeric_test)

del imputed_numeric_X, imputed_numeric_test
print("No. of Numeric Features in Source: {0}, in Model: {1}".format(len(numeric_columns), len(X.columns)))

faulty_columns = set(numeric_columns) - set(X.columns)

print(faulty_columns)
for a_faulty_column in X.columns:

    if(a_faulty_column !=  'SalePrice'):

        train_df[a_faulty_column].loc[train_df[a_faulty_column].isnull()] = train_df[a_faulty_column].mean()

        train_df[a_faulty_column].loc[~np.isfinite(train_df[a_faulty_column])] = train_df[a_faulty_column].mean()

        

        test_df[a_faulty_column].loc[test_df[a_faulty_column].isnull()] = test_df[a_faulty_column].mean()

        test_df[a_faulty_column].loc[~np.isfinite(test_df[a_faulty_column])] = test_df[a_faulty_column].mean()

        

        train_df[a_faulty_column].astype('float32', inplace=True)

        test_df[a_faulty_column].astype('float32', inplace=True)
train_df, test_df = read_train_test_data()

X = train_df[categorical_columns]

y = train_df.SalePrice

print(X.shape)

test_X = test_df[categorical_columns]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



failed_features = []

for aFeature in categorical_columns:

    try:

        X[aFeature] = le.fit_transform(X[aFeature])

        test_X[aFeature] = le.transform(test_X[aFeature])

    except:

        failed_features.append(aFeature)

        

X.drop(columns=failed_features, inplace=True)

test_X.drop(columns=failed_features, inplace=True)



print(failed_features)
imputer = SimpleImputer(strategy='median')

imputed_X = imputer.fit_transform(X)



imputed_test = imputer.transform(test_X)
predictions = simple_model(imputed_X, y, imputed_test)

del imputed_X, imputed_test
write_submissions("simple_model_s2.csv", test_df, predictions)
train_df, test_df = read_train_test_data()
le = LabelEncoder()



failed_features = [] # Placeholder to store the failed features.

for aFeature in categorical_columns:

    try:

        train_df[aFeature] = le.fit_transform(train_df[aFeature])

        test_df[aFeature] = le.transform(test_df[aFeature])

    except:

        failed_features.append(aFeature)

        

train_df.drop(columns=failed_features, inplace=True)

test_df.drop(columns=failed_features, inplace=True)
imputer = SimpleImputer(strategy='median')

imputer.fit(train_df)

imputed_data = imputer.transform(train_df)

train_df = pd.DataFrame(imputed_data, columns=train_df.columns)
imputer = SimpleImputer(strategy='median')

imputer.fit(test_df)

imputed_data = imputer.transform(test_df)

test_df = pd.DataFrame(imputed_data, columns=test_df.columns)
features = test_df.columns

X = train_df[features]

y = train_df.SalePrice

predictions = simple_model(X, y, test_df)
write_submissions("simple_model_s3.csv", test_df, predictions)
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X, y)

forest_predictions = forest_model.predict(test_df)
write_submissions("simple_model_s4.csv", test_df, forest_predictions)
from xgboost import XGBRegressor

model_xgb = XGBRegressor()

model_xgb.fit(X, y, verbose=False)

xgb_predictions = model_xgb.predict(test_df)
write_submissions("simple_model_s5.csv", test_df, xgb_predictions)
print("Shape of Train Data: {0}".format(X.shape))

print("Shape of Test Data: {0}".format(test_df.shape))

print("No. of Failed Categorical Features: {0}".format(len(failed_features)))

print("No. of Failed Numeric Features: {0}".format(len(faulty_columns)))
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



from sklearn.metrics import mean_absolute_error



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Write loop to find the ideal tree size from max_leaf_nodes

maes = [[get_mae(a, train_X, val_X, train_y, val_y), a] for a in max_leaf_nodes]

print(maes)
best_tree_size = min(maes)[1]

print(best_tree_size)
optimal_tree_size_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)



# fit the final model and uncomment the next two lines

optimal_tree_size_model.fit(train_X, train_y)

optimal_tree_predictions = optimal_tree_size_model.predict(test_df)

# step_2.check()
xgb_model = XGBRegressor(n_estimators=1000, learning_Rate=0.05)

xgb_model.fit(train_X, train_y, early_stopping_rounds=15, eval_set=[(val_X, val_y)], verbose=True)
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.ensemble import GradientBoostingRegressor



gbr_model = GradientBoostingRegressor()

gbr_model.fit(train_X, train_y)

plots = plot_partial_dependence(gbr_model, features=[1, 2, 8], X=train_X, feature_names=train_X.columns, grid_resolution=200)
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(SimpleImputer(), XGBRegressor())

pipeline.fit(X, y)

validation_predictions = pipeline.predict(val_X)



print("MAE " + str(mean_absolute_error(validation_predictions, val_y)))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X,y, scoring='neg_mean_absolute_error')

print(scores)