import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



X = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')

X_test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')
X.info()
numeric_cols_with_missing = [cname for cname in X.columns 

                if X[cname].dtype in ['int64', 'float64']

                and (X[cname].isnull().any())]



missing_val_count_by_column = X[numeric_cols_with_missing].isnull().sum()

print(missing_val_count_by_column[missing_val_count_by_column > 0])
sns.kdeplot(data=X['LotFrontage'],shade=True)

print(X['LotFrontage'].min())
sns.stripplot(data=X,x=X['MasVnrType'],y=X['MasVnrArea'])

print(X['MasVnrType'].value_counts())
print(X['GarageCars'].value_counts())

print(X['GarageArea'].value_counts())
sns.regplot(x=X['GarageYrBlt'],y=X['SalePrice'])
categorical_cols_with_missing = [cname for cname in X.columns 

                if X[cname].dtype == "object"

                and (X[cname].isnull().any())]



missing_val_count_by_column = X[categorical_cols_with_missing].isnull().sum()

print(missing_val_count_by_column[missing_val_count_by_column > 0])
from sklearn.impute import SimpleImputer



none_imp = SimpleImputer(strategy='constant',fill_value='None')

most_freq_imp = SimpleImputer(strategy='most_frequent')

zero_imp = SimpleImputer(strategy='constant',fill_value=0.0)

median_imp = SimpleImputer(strategy='median')



def impute(data,imput_type,column_name):

    if(imput_type == 'zero'):

        data[[column_name]] = zero_imp.fit_transform(data[[column_name]])

    elif(imput_type == 'none'):

        data[[column_name]] = none_imp.fit_transform(data[[column_name]])

    elif(imput_type == 'most_freq'):

        data[[column_name]] = most_freq_imp.fit_transform(data[[column_name]])

    elif(imput_type == 'median'):

        data[[column_name]] = median_imp.fit_transform(data[[column_name]])
fill_with_none = ('GarageType','GarageFinish','GarageQual','GarageCond','MasVnrType','Alley',

                 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','PoolQC','Fence','MiscFeature')

fill_with_most_freq = ('Electrical',)

fill_with_zero = ('MasVnrArea',)

fill_with_median = ('LotFrontage','GarageYrBlt')

for x in (X,X_test):

    for col in (fill_with_none):

        impute(x,'none',col)

    for col in (fill_with_zero):

        impute(x,'zero',col)

    for col in (fill_with_most_freq):

        impute(x,'most_freq',col)

    for col in (fill_with_median):

        impute(x,'median',col)
missing_val_count_by_column = X_test.isnull().sum()

print(missing_val_count_by_column[missing_val_count_by_column > 0])
for col in ('BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea'):

    impute(X_test,'zero',col)

for col in ('SaleType','Exterior1st','Exterior2nd','MSZoning','Utilities','KitchenQual','Functional'):

    impute(X_test,'most_freq',col)
import category_encoders as ce

cat_features = [cname for cname in X.columns

                   if X[cname].dtype == 'object']

    

count_enc = ce.CountEncoder(cols=cat_features)

count_enc.fit(X[cat_features])



X = X.join(count_enc.transform(X[cat_features]).add_suffix('_count'))

X_test = X_test.join(count_enc.transform(X_test[cat_features]).add_suffix('_count'))



target_enc = ce.TargetEncoder(cols= cat_features)

target_enc.fit(X=X[cat_features],y=X['SalePrice'])



X = X.join(target_enc.transform(X[cat_features]).add_suffix('_target'))

X_test = X_test.join(target_enc.transform(X_test[cat_features]).add_suffix('_target'))

from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)

OH_cols_X = pd.DataFrame(OH_encoder.fit_transform(X[cat_features]))

OH_cols_X_test = pd.DataFrame(OH_encoder.transform(X_test[cat_features]))



# One-hot encoding removed index; put it back

OH_cols_X.index = X.index

OH_cols_X_test.index = X_test.index



# Remove categorical columns (will replace with one-hot encoding)

X = X.drop(cat_features, axis=1)

X_test = X_test.drop(cat_features, axis=1)



# Add one-hot encoded columns to numerical features

X = pd.concat([X, OH_cols_X], axis=1)

X_test = pd.concat([X_test, OH_cols_X_test], axis=1)
y =X["SalePrice"]

X = X.drop("SalePrice",axis = 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = pd.DataFrame(scaler.fit_transform(X),index=X.index,columns=X.columns)

X_test = pd.DataFrame(scaler.fit_transform(X_test),index=X_test.index,columns=X_test.columns)
import xgboost as xgb

from xgboost import plot_importance

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

from numpy import sort



regr = xgb.XGBRegressor(n_estimators = 1000,learning_rate=0.05)



xgbSelector = xgb.XGBRegressor(n_estimators = 1000,learning_rate=0.05).fit(X,y)

model = SelectFromModel(xgbSelector,prefit=True,threshold='median')

X_new = model.transform(X)

    

selected_features = pd.DataFrame(model.inverse_transform(X_new), 

                                        index=X.index,

                                        columns=X.columns)

    

cols_to_keep = selected_features.columns[selected_features.var() != 0]

X_med = X[cols_to_keep]

X_test_med = X_test[cols_to_keep]

scores = -1 * cross_val_score(regr,X_med,y,cv=5,scoring='neg_mean_absolute_error')

print(scores)
xgbSelector2 = xgb.XGBRegressor(n_estimators = 1000,learning_rate=0.05).fit(X,y)

model = SelectFromModel(xgbSelector,prefit=True,threshold='mean')

X_new = model.transform(X)

    

selected_features = pd.DataFrame(model.inverse_transform(X_new), 

                                        index=X.index,

                                        columns=X.columns)

    

cols_to_keep = selected_features.columns[selected_features.var() != 0]

X_mean = X[cols_to_keep]

X_test_mean = X_test[cols_to_keep]

scores = -1 * cross_val_score(regr,X_mean,y,cv=5,scoring='neg_mean_absolute_error')

print(scores)



print(np.median(sort(xgbSelector.feature_importances_)))

print(np.mean(sort(xgbSelector.feature_importances_))) 
X = X_med

X_test = X_test_med
print(X.columns)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal



param_distribs = {

        'n_estimators': [600,700,800,900,1000],

        'max_depth': [2,3,4,5,6],

        'learning_rate': reciprocal(0.01,1),

    }



regressor = xgb.XGBRegressor()



rnd_search = RandomizedSearchCV(regressor, param_distributions=param_distribs,

                                n_iter=50, cv=5,scoring='neg_mean_absolute_error',

                                verbose=2,random_state=42,n_jobs=-1)

rnd_search.fit(X,y)
from sklearn.model_selection import GridSearchCV

param_grid = [ {'learning_rate': [0.03,0.04,0.05,0.06],

              'n_estimators': [800,900,1000],'max_depth': [3,4]}

]

grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_absolute_error', return_train_score=True, n_jobs =-1)

grid_search.fit(X,y)
print(rnd_search.best_params_)

print(rnd_search.best_score_)

print(grid_search.best_params_)

print(grid_search.best_score_)



#Let's pick the parameters found with the RandomSearchCV



# best_parameters I found myself (learning_rate= 0.054,n_estimators=1000,max_depth= 3)

final_regressor = xgb.XGBRegressor(learning_rate= 0.06,n_estimators=800,max_depth= 3)
scores = -1 * cross_val_score(final_regressor,X,y,cv=6,scoring='neg_mean_absolute_error')

print(scores)

print("Average MAE score (across experiments):")

print(scores.mean())
final_regressor.fit(X,y)

predictions = final_regressor.predict(X_test)



# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                    'SalePrice': predictions})

output.to_csv('submission.csv', index=False)



print('Output produced')