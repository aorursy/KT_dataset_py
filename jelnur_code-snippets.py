import pandas as pd



# Read csv file into DataFrame

main_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

data = pd.read_csv(main_file_path)
# Describe data

data.describe()
# Get columns list

data.columns
# print first 5 rows

data.head()
# Selecting single column

price_data = data.Price

price_data.head()
# the head command returns the top few lines of data.

price_data.head()
# Selecting multiple columns

columns_of_interest = ['Address', 'Rooms']

two_columns_of_data = data[columns_of_interest]

two_columns_of_data.head()
from sklearn.datasets import load_iris

from sklearn import tree

iris = load_iris()

clf = tree.DecisionTreeClassifier()

clf = clf.fit(iris.data, iris.target)



import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("iris")



dot_data = tree.export_graphviz(clf, out_file=None,

                                feature_names=iris.feature_names,  

                                class_names=iris.target_names,  

                                filled=True, rounded=True,  

                                special_characters=True) 

graph = graphviz.Source(dot_data)  

graph
# split data into training and test sets

from sklearn.model_selection import train_test_split



melbourne_predictors = ['Rooms']

X = data[melbourne_predictors]

y = data.Price



train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)
from sklearn.tree import DecisionTreeRegressor



melbourne_model = DecisionTreeRegressor(max_depth=30)

melbourne_model.fit(train_X, train_y)
from sklearn.metrics import mean_absolute_error



predicted_home_prices = melbourne_model.predict(test_X)

print("Decision tree MAE: ", mean_absolute_error(test_y, predicted_home_prices))
from sklearn.ensemble import RandomForestRegressor



# Define model

melbourne_model = RandomForestRegressor()



# Fit model

melbourne_model.fit(train_X, train_y)

y_pred = melbourne_model.predict(test_X)

print("Random Forest MAE: ", mean_absolute_error(test_y, y_pred))
# Submission

my_submission = pd.DataFrame({'Rooms': test_X.Rooms, 'SalePrice': predicted_home_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
# check column has any null values

data.Landsize.isnull().any()
# check which columns are null

data.isnull().any()
# count nulls

data.isnull().sum()
# RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor()

forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(test_X)

print(mean_absolute_error(test_y, melb_preds))
# Drop the columns where all elements are nan:

data.dropna(axis=1, how='all')
# drop columns with any null data

cols_with_missing = [col for col in original_data.columns if original_data[col].isnull().any()]

redused_original_data = original_data.drop(cols_with_missing, axis=1)

reduced_test_data = test_data.drop(cols_with_missing, axis=1)
# Imputation

import numpy as np

from sklearn.preprocessing import Imputer

#my_imputer = Imputer()

#data_with_imputed_values = my_imputer.fit_transform(data)



imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

imp.fit([[1, 2], [np.nan, 3], [7, 10]])



X = [[np.nan, 2], [6, np.nan], [7, 6]]

print(imp.transform(X))    
# selecting columns

melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].nunique() < 10 and

                                candidate_train_predictors[cname].dtype == "object"]

numeric_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]

test_predictors = candidate_test_predictors[my_cols]
# getting sample of data

train_predictors.sample(10)
# convert objects to one-hot encodings

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)



# align test and training predictos

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)

# import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

data.dropna(axis=0, subset=['Price'], inplace=True)

y = data.Price

X = data.drop(['Price'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25)



my_imputer = Imputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], verbose=False)



# make predictions

predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
