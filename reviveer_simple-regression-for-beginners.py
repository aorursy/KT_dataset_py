# Data Manipulation

import pandas as pd

import numpy as np



# sklearn helper functions

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import ShuffleSplit, cross_validate, GridSearchCV, cross_val_score



# sklearn ML algorithms

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor



# Visualization

import matplotlib.pyplot as plt



# Notebook customization

pd.options.display.max_rows = 500

pd.options.display.max_columns = 100
# read data from csv files

df_train_raw = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test_raw = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
# create a copy for exploration purposes

df1 = df_train_raw.copy()



# display some basic information about the data

print(df_train_raw.info())

df_train_raw.sample(10)
# check null value count in train and test sets

print("Null values in training set:\n", df1.isnull().sum())

print("-"*50)



print("Null values in test set:\n", df_test_raw.isnull().sum())

print("-"*50)



# summary statistics for train set

df1.describe(include='all')
# Delete attributes with high null count in training set

drop_columns_list = ['Id','Alley', 'PoolQC', 'Fence', 'MiscFeature']

df1.drop(columns=drop_columns_list, inplace=True)
plt.scatter(df1['GrLivArea'], df1['SalePrice'], c = "pink", marker = "s")

plt.title("Looking for outliers")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.grid(True)

plt.show()
df1 = df1[df1['GrLivArea'] < 4000]
# Taking natural log of 'SalePrice'

df1['SalePrice'] = np.log1p(df1['SalePrice'])
# Remove target variable from training set and keep a copy of it

train_labels = df1['SalePrice'].copy()

df1.drop('SalePrice', axis=1, inplace=True)
# selecting features for different preprocessing methods

preprocess_features1 = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageYrBlt', 'GarageCars', 'GarageArea']

preprocess_features2 = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'SaleType']

preprocess_features3 = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

preprocess_features4 = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

preprocess_features5 = ['KitchenQual']

preprocess_features6 = ['Functional']

preprocess_features7 = ['FireplaceQu']



remove_list = preprocess_features1 + preprocess_features2 + preprocess_features3 + preprocess_features4 + preprocess_features5 + preprocess_features6 + preprocess_features7



all_numeric_columns = list(df1.select_dtypes(include=[np.number]).columns.values)

all_object_columns = list(df1.select_dtypes(include=['object']).columns.values)



preprocess_features8 = [i for i in all_numeric_columns if i not in remove_list]

preprocess_features9 = [i for i in all_object_columns if i not in remove_list]



# creating transformers

transformer1 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])



transformer2 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder())

])



transformer3 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value="No Basement")),

    ('onehot', OneHotEncoder())

])



transformer4 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value="No Garage")),

    ('onehot', OneHotEncoder())

])



transformer5 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value="TA")),

    ('onehot', OneHotEncoder())

])



transformer6 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value="Typ")),

    ('onehot', OneHotEncoder())

])



transformer7 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value="No Fireplace")),

    ('onehot', OneHotEncoder())

])



transformer8 = Pipeline(steps=[

    ('scaler', StandardScaler())

])



transformer9 = Pipeline(steps=[

    ('onehot', OneHotEncoder())

])





# final transformer: ColumnTransformer will automatically apply transforms to specific columns

preprocessor = ColumnTransformer(

    transformers=[

        ('t1', transformer1, preprocess_features1),

        ('t2', transformer2, preprocess_features2),

        ('t3', transformer3, preprocess_features3),

        ('t4', transformer4, preprocess_features4),

        ('t5', transformer5, preprocess_features5),

        ('t6', transformer6, preprocess_features6),

        ('t7', transformer7, preprocess_features7),

        ('t8', transformer8, preprocess_features8),

        ('t9', transformer9, preprocess_features9)])
# apply full preprocessing pipeline on train data

df1 = preprocessor.fit_transform(df1)
# list of ML algorithms for cross-validation

MLA = [

    RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, min_samples_leaf=4, n_jobs=-1),

    

    ExtraTreesRegressor(n_estimators=500, max_leaf_nodes=16, min_samples_leaf=4, n_jobs=-1),

    

    AdaBoostRegressor(DecisionTreeRegressor(max_depth=1), 

                       n_estimators=200, 

                       loss='square',

                       learning_rate=0.5),

    

    XGBRegressor(learning_rate=0.05, max_depth=4, n_estimators=50, reg_alpha=0.01, reg_lambda=0.3, seed=0),

    

    BaggingRegressor(base_estimator=DecisionTreeRegressor(), 

                      n_estimators=500,

                      max_samples=50,

                      bootstrap=True,

                      n_jobs=-1),    # Bagging

    

    BaggingRegressor(base_estimator=DecisionTreeRegressor(), 

                      n_estimators=500,

                      max_samples=50,

                      bootstrap=False,

                      n_jobs=-1),    # Pasting

    

    BaggingRegressor(base_estimator=DecisionTreeRegressor(), 

                      n_estimators=500,

                      max_samples=100,

                      bootstrap=True,

                      max_features=5,

                      bootstrap_features=True,

                      n_jobs=-1),    # Random Patches Method

    

    BaggingRegressor(base_estimator=DecisionTreeRegressor(), 

                      n_estimators=500,

                      bootstrap=False,

                      max_features=5,

                      bootstrap_features=True,

                      n_jobs=-1),    # Random Subspaces Method

    

    LinearRegression(),

    

    Ridge(alpha=1, solver='auto'),

    

    Lasso(alpha=0.1),

    

    ElasticNet(alpha=0.1, l1_ratio=1),

    

    SGDRegressor(early_stopping=True, n_iter_no_change=10, learning_rate='constant', eta0=0.002),

    

    SVR(kernel='linear', epsilon=0.1),

    

    SVR(kernel='poly', degree=2, C=100, epsilon=0.1),

    

    DecisionTreeRegressor(min_samples_leaf=10)

]
cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)



MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Score Mean', 'MLA Test Score Mean' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



MLA_predict = train_labels.copy()



row_index = 0

for alg in MLA:



    # set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    # cross validation

    cv_results = cross_validate(alg, df1, train_labels, cv=cv_split, scoring='neg_root_mean_squared_error', return_train_score=True)

    

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Train Score Mean'] = -(cv_results['train_score'].mean())

    MLA_compare.loc[row_index, 'MLA Test Score Mean'] = -(cv_results['test_score'].mean())



    # save MLA predictions

    alg.fit(df1, train_labels)

    MLA_predict[MLA_name] = alg.predict(df1)

    

    row_index+=1

    

# print and sort table

MLA_compare.sort_values(by = ['MLA Test Score Mean'], ascending = True, inplace = True)

MLA_compare
shortlisted_model = Ridge()



param_grid = {

    'alpha': [1, 2, 3, 10, 20, 30, 50, 100, 300, 400]

}



grid_search = GridSearchCV(shortlisted_model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=cv_split, return_train_score=True)

grid_search.fit(df1, train_labels)



print('AFTER GridSearch Parameters: ', grid_search.best_params_)

print("AFTER GridSearch Training RMSE mean: {:.2f}". format(-max(grid_search.cv_results_['mean_train_score'])))

print("AFTER GridSearch Test RMSE mean: {:.2f}". format(-max(grid_search.cv_results_['mean_test_score'])))

print('-'*10)
cvres = grid_search.cv_results_

print("Train Score Mean\t", "Test Score Mean\t", "Parameters")

print("-"*70)

for mean_train_score, mean_test_score, params in zip(cvres['mean_train_score'], cvres['mean_test_score'], cvres['params']):

    print(-mean_train_score,"\t", -mean_test_score,"\t", params)
# predictors for voting regressor

predictor1 = Ridge(alpha=30, solver='auto')

    

predictor2 = Lasso(alpha=0.1)

    

predictor3 = ElasticNet(alpha=0.1, l1_ratio=1)

    

predictor4 = SVR(kernel='linear', epsilon=0.1)

    

predictor5 = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)

    

predictor6 = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, min_samples_leaf=4, n_jobs=-1)
voting_reg = VotingRegressor(

    estimators=[('pred1', predictor1), ('pred2', predictor2), ('pred3', predictor3), ('pred4', predictor4), ('pred5', predictor5), ('pred6', predictor6)]

)



-(cross_val_score(voting_reg, df1, train_labels, cv=cv_split, scoring='neg_root_mean_squared_error').mean())
# copy test set 'Id'

submission_Ids = df_test_raw['Id'].copy()



# remove features with high null value count

df_test_raw.drop(columns=drop_columns_list, inplace=True)



# apply preprocessing pipeline to test set

df_test_prepared = preprocessor.transform(df_test_raw)
# FINAL MODEL

final_model = Ridge(alpha=30)

final_model.fit(df1, train_labels)



# submission predictions

final_pred = final_model.predict(df_test_prepared)

final_pred = np.expm1(final_pred)
# create submission file

my_submission = pd.DataFrame({'Id': submission_Ids, 'SalePrice': final_pred})

my_submission.to_csv("submission.csv", index=False)