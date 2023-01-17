import pandas as pd

from learntools.core import *



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



# Drop houses where the target is missing

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)



target = train_data.SalePrice



X_train = train_data.drop(['SalePrice'], axis=1)
from sklearn.impute import SimpleImputer



imputed_X_train_plus = X_train.copy()

imputed_X_test_plus = test_data.copy()



cols_with_missing = (col for col in X_train.columns 

                                 if X_train[col].isnull().any())

for col in cols_with_missing:

    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()

    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()



# Imputation

my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(imputed_X_train_plus.select_dtypes(exclude=['object'])))

imputed_X_test_plus = pd.DataFrame(my_imputer.transform(imputed_X_test_plus.select_dtypes(exclude=['object'])))

# "cardinality" means the number of unique values in a column.

low_cardinality_cols = [cname for cname in imputed_X_train_plus.columns if 

                                imputed_X_train_plus[cname].nunique() < 10 and

                                imputed_X_train_plus[cname].dtype == "object"]

numeric_cols = [cname for cname in imputed_X_train_plus.columns if 

                                imputed_X_train_plus[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

train_predictors = imputed_X_train_plus[my_cols]

test_predictors = imputed_X_test_plus[my_cols]



one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence



cols_to_use = ['YearBuilt', 'MSSubClass', 'LotArea']



def get_some_data():

    data = pd.read_csv('../input/train.csv')

    y = data.SalePrice

    X = data[cols_to_use]

    my_imputer = SimpleImputer()

    imputed_X = my_imputer.fit_transform(X)

    return imputed_X, y

    



X, y = get_some_data()

my_model = GradientBoostingRegressor()

my_model.fit(X, y)

my_plots = plot_partial_dependence(my_model, 

                                   features=[0,2], 

                                   X=X, 

                                   feature_names=cols_to_use, 

                                   grid_resolution=10)

from sklearn.ensemble import RandomForestRegressor



# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(final_train, target)

from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=1000)

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(final_train, target, verbose=False)
# make predictions with XGBoost

predictions = my_model.predict(final_test)

output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': predictions})

print(output)
# make predictions with Random Forest

test_preds = rf_model_on_full_data.predict(final_test)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

print(output)

# output.to_csv('submission.csv', index=False)
from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer



# make predictions with Random Forest using pipeline

my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())



my_pipeline.fit(final_train, target)

predictions = my_pipeline.predict(final_test)



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': predictions})

print(output)

output.to_csv('submission.csv', index=False)
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



scores = cross_val_score(my_pipeline, final_train, target, scoring='neg_mean_absolute_error')

print(scores)

print('Mean Absolute Error %2f' %(-1 * scores.mean()))