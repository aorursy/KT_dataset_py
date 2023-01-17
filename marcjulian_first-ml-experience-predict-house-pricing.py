# Import all modules



import numpy as np # linear algebra

import pandas as pd # data processing



# ML model: RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



#Cross Validation

from sklearn.model_selection import cross_val_score



# Models

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



# Preprocessing

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



# Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV



# Visualization

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



print('Import completed')
# Load csv files

train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv')

# First look at the data

print(train_data.shape)

train_data.head()
# Set target which will be predicted later on

target = train_data['SalePrice']



# Splitting data in numerical and categorical subsets

num_attr = train_data.select_dtypes(exclude='object').drop('SalePrice', axis=1).copy()

cat_attr = train_data.select_dtypes(include='object').copy()
# Finding outliers by graphing numerical attributes to SalePrice

plots = plt.figure(figsize=(12,20))



print('Loading 35 plots ...')

for i in range(len(num_attr.columns)-1):

    plots.add_subplot(9, 4, i+1)

    sns.regplot(num_attr.iloc[:,i], target)

    

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
cat_attr.columns
sns.countplot(x='SaleCondition', data=cat_attr)
# Missing values for numerical attributes

num_attr.isna().sum().sort_values(ascending=False).head()
# Missing values for categorical attributes

cat_attr.isna().sum().sort_values(ascending=False).head(16)
# Copy the data to prevent changes to original data

data_copy = train_data.copy()



data_copy.MasVnrArea = data_copy.MasVnrArea.fillna(0)



# Columns which can be filled with 'None'

cat_cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',

                     'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',

                     'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',

                     'MasVnrType']

for cat in cat_cols_fill_none:

    data_copy[cat] = data_copy[cat].fillna("None")

    

data_copy.isna().sum().sort_values(ascending=False).head()
# Dropping outliers found when visualizing the numerical subset of our dataset

data_copy = data_copy.drop(data_copy['LotFrontage'][data_copy['LotFrontage']>200].index)

data_copy = data_copy.drop(data_copy['LotArea'][data_copy['LotArea']>100000].index)

data_copy = data_copy.drop(data_copy['BsmtFinSF1'][data_copy['BsmtFinSF1']>4000].index)

data_copy = data_copy.drop(data_copy['TotalBsmtSF'][data_copy['TotalBsmtSF']>6000].index)

data_copy = data_copy.drop(data_copy['1stFlrSF'][data_copy['1stFlrSF']>4000].index)

data_copy = data_copy.drop(data_copy.GrLivArea[(data_copy['GrLivArea']>4000) & (target<300000)].index)

data_copy = data_copy.drop(data_copy.LowQualFinSF[data_copy['LowQualFinSF']>550].index)



X = data_copy.drop('SalePrice', axis=1)



y = data_copy.SalePrice





numerical_transformer = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('std_scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, num_attr.columns),

        ('cat', categorical_transformer, cat_attr.columns)

    ])
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05)



xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', xgb_model)

                             ])





param_grid = [

    {'model__n_estimators': [50, 100, 150, 200, 250, 300, 500], 'model__learning_rate': [0.01, 0.02, 0.05, 0.1, 0.3]}

]



grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)





grid_search.fit(X, y)





best_model = grid_search.best_estimator_
# Applying the same data cleaning we used for the training data to the test data



test_X = test_data.copy()

test_X.MasVnrArea = test_X.MasVnrArea.fillna(0)

test_X = test_X.drop('Id', axis=1)

test_preds = best_model.predict(test_X)

test_preds
output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

print('Submitted')