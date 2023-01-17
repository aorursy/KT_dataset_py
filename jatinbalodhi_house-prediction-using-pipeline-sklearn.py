import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import SGDRegressor, Lasso

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.feature_selection import SelectKBest, f_classif, chi2



import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline





pd.set_option('display.max_columns', 81)



class ModifiedLabelEncoder(LabelEncoder):



    def fit_transform(self, y, *args, **kwargs):

        return super().fit_transform(y)

#     .reshape(-1, 1)



    def transform(self, y, *args, **kwargs):

        return super().transform(y)

#     .reshape(-1, 1)



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_df.shape
train_df.describe()

X = train_df.copy()



y = X['SalePrice']

X.drop(columns=['Id', 'SalePrice'], inplace=True)

test_df.drop(columns='Id', inplace=True)
non_numerical_cols =  list(X.select_dtypes('object').columns)

numerical_cols = list(set(X.columns) - set(non_numerical_cols))



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

#     ('label_encoder', ModifiedLabelEncoder())

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, non_numerical_cols)

    ])



random_state = np.random

model = XGBRegressor()

# model = RandomForestRegressor()
pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('k_best', SelectKBest()),

    ('model', model)

 ])
# pipeline.fit(X_train, y_train)

parameters = [

    {

        'k_best__score_func': [f_classif, chi2],

#         'k_best__k': [20, 40, 60, 80, 100, 120],

        'k_best__k': [80, 90, 100, 120, 140, 200],

        'model__n_estimators': [0, 100, 200, 300, 400, 500],

        'model__learning_rates': [0.01]

    }

]

gscv = GridSearchCV(pipeline, param_grid=parameters, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=15)

gscv.fit(X_train, y_train)
gscv.best_params_, 1* - gscv.best_score_
y_pred = gscv.predict(X_val)

(mean_squared_error(y_val, y_pred), 1* - gscv.best_score_)

# y_val[0:5], y_pred[0:5]
submission_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission_df.shape

# test_predict.head()

submission_df['SalePrice'] = gscv.predict(test_df)

submission_df.to_csv('submission.csv', index=False)
lasso= Lasso()



pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('k_best', SelectKBest()),

    ('model', lasso)

 ])



parameters= {

    'model__alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100],

    'k_best__k': [50, 60, 70, 80, 90, 120, 140]

}



lasso = Lasso()

lasso_reg = GridSearchCV(pipeline, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)

lasso_reg.fit(X_train,y_train)



print('The best value of Alpha is: ',lasso_reg.best_params_)

y_pred = lasso_reg.predict(X_val)

(mean_squared_error(y_val, y_pred), 1* - lasso_reg.best_score_)
submission_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission_df.shape

# test_predict.head()

submission_df['SalePrice'] = lasso_reg.predict(test_df)

submission_df.to_csv('submission_lasso.csv', index=False)