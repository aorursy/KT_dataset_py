# Core

import pandas as pd

import numpy as np



# Data Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Path of the file to read.

filepath = '../input/house-prices-advanced-regression-techniques/train.csv'



#Note: 1st column is ID

housing = pd.read_csv(filepath, index_col=0)
housing.shape
housing.describe().T
housing.info()
housing.select_dtypes(include=['object']).columns



housing.select_dtypes(include=['object']).describe()

housing.select_dtypes(exclude='object').columns



housing.select_dtypes(exclude='object').describe().T

housing.select_dtypes(exclude='object').info()
num_attributes = housing.select_dtypes(exclude='object').drop('SalePrice',axis=1).copy()
fig = plt.figure(figsize=(12,18))

for i in range(len(num_attributes.columns)):

    fig.add_subplot(9,4,i+1)

    sns.distplot(num_attributes.iloc[:,i].dropna(),kde_kws={'bw':0.10})

    plt.xlabel(num_attributes.columns[i])



plt.tight_layout()

plt.show()
plt.figure(figsize = (9,6))

sns.distplot(housing['SalePrice'])

plt.title('Dsitribution of SalesPrice')

plt.show()
housing['SalePrice'] = np.log(housing['SalePrice'])
plt.figure(figsize = (9,6))

sns.distplot(housing['SalePrice'])

plt.title('Dsitribution of SalesPrice')

plt.show()
target = housing.SalePrice
fig = plt.figure(figsize=(12, 18))



for i in range(len(num_attributes.columns)):

    fig.add_subplot(9, 4, i+1)

    sns.boxplot(y=num_attributes.iloc[:,i])



plt.tight_layout()

plt.show()
f = plt.figure(figsize=(12,20))



for i in range(len(num_attributes.columns)):

    f.add_subplot(9, 4, i+1)

    sns.scatterplot(num_attributes.iloc[:,i], target)

    

plt.tight_layout()

plt.show()
corr_matrix = housing.corr()

corr_matrix["SalePrice"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ["SalePrice", "OverallQual", "GrLivArea",

"GarageCars"]

scatter_matrix(housing[attributes], figsize=(12, 8))
correlation = housing.corr()



f, ax = plt.subplots(figsize=(14,12))

plt.title('Correlation of numerical attributes', size=16)

sns.heatmap(correlation)

plt.show()

num_correlation = housing.select_dtypes(exclude='object').corr()

plt.figure(figsize=(20,20))

plt.title('High Correlation')

sns.heatmap(num_correlation > 0.8, annot=True, square=True)
corr = num_correlation.corr()

print(corr['SalePrice'].sort_values(ascending=False))

# Show columns with most null values:

num_attributes.isna().sum().sort_values(ascending=False).head()
cat_columns = housing.select_dtypes(include='object').columns

print(cat_columns)

var = housing['KitchenQual']

f, ax = plt.subplots(figsize=(10,6))

sns.boxplot(y=housing.SalePrice, x=var)

plt.show()

f, ax = plt.subplots(figsize=(12,8))

sns.boxplot(y=housing.SalePrice, x=housing.Neighborhood)

plt.xticks(rotation=40)

plt.show()
## Count of categories within Neighborhood attribute

fig = plt.figure(figsize=(12.5,4))

sns.countplot(x='Neighborhood', data=housing)

plt.xticks(rotation=90)

plt.ylabel('Frequency')

plt.show()

housing[cat_columns].isna().sum().sort_values(ascending=False).head(17)

# Create copy of dataset  ====================================

housing_data_copy = housing.copy()



# Dealing with missing/null values ===========================

# Numerical columns:

housing_data_copy.MasVnrArea = housing_data_copy.MasVnrArea.fillna(0)





# Categorical columns:

cat_cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',

                     'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',

                     'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',

                     'MasVnrType','Electrical']

for cat in cat_cols_fill_none:

    housing_data_copy[cat] = housing_data_copy[cat].fillna("None")

# Check for outstanding missing/null values

# Scikit-learn's Imputer will be used to address these

housing_data_copy.isna().sum().sort_values(ascending=False).head()
# Remove outliers based on observations on scatter plots against SalePrice:

housing_data_copy = housing_data_copy.drop(housing_data_copy['LotFrontage'][housing_data_copy['LotFrontage']>200].index)

housing_data_copy = housing_data_copy.drop(housing_data_copy['LotArea'][housing_data_copy['LotArea']>100000].index)

housing_data_copy = housing_data_copy.drop(housing_data_copy['BsmtFinSF1']

                                     [housing_data_copy['BsmtFinSF1']>4000].index)

housing_data_copy = housing_data_copy.drop(housing_data_copy['TotalBsmtSF']

                                     [housing_data_copy['TotalBsmtSF']>6000].index)

housing_data_copy = housing_data_copy.drop(housing_data_copy['1stFlrSF']

                                     [housing_data_copy['1stFlrSF']>4000].index)

housing_data_copy = housing_data_copy.drop(housing_data_copy.GrLivArea

                                     [(housing_data_copy['GrLivArea']>4000) & 

                                      (target<300000)].index)

housing_data_copy = housing_data_copy.drop(housing_data_copy.LowQualFinSF

                                     [housing_data_copy['LowQualFinSF']>550].index)
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split



# Remove attributes that were identified for excluding when viewing scatter plots & corr values

attributes_drop = ['SalePrice', 'MiscVal', 'MSSubClass', 'MoSold', 'YrSold', 

                   'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd'] # high corr with other attributes



X = housing_data_copy.drop(attributes_drop, axis=1)

cat_columns = X.select_dtypes(include='object').columns 

num_attributes = X.select_dtypes(exclude='object').columns 

# Create target object and call it y

y = housing_data_copy.SalePrice

from sklearn import metrics

from sklearn.model_selection import cross_val_score





def crossval(model):

    scores = cross_val_score(model, X, y, cv=10)

    return scores.mean()



def print_scores(original, predicted):  

    r2square = metrics.r2_score(original, predicted)

    mae = metrics.mean_absolute_error(original, predicted)

    mse = metrics.mean_squared_error(original, predicted)

    rmse = np.sqrt(metrics.mean_squared_error(original, predicted))

    print('MAE:', mae)

    print('MSE:', mse)

    print('RMSE:', rmse)

    print('R2 Square', r2square)
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder



X = pd.get_dummies(X)





# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



imputer=SimpleImputer()



# Impute the data

train_X=imputer.fit_transform(train_X)

val_X=imputer.transform(val_X)





# Normalisation 

normaliser = StandardScaler()

train_X = normaliser.fit_transform(train_X)

val_X = normaliser.transform(val_X)


# Random Forest

rf_model = RandomForestRegressor(n_estimators=100,random_state=2)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)



print_scores(val_y, rf_val_predictions)

print("Training set score : ", rf_model.score(train_X, train_y))

print("Validation set score : ", rf_model.score(val_X, val_y))

rf_model.feature_importances_



# I tried to show the column name next to importance score but looks some issue, 

# it seems to be misalligned



#for name, score in zip(X.columns, rf_model.feature_importances_):

#    print(name, score)
# Ridge Model



from sklearn.linear_model import Ridge



rg_model = Ridge()

rg_model.fit(train_X, train_y)

pred = rg_model.predict(val_X)



print_scores(val_y, pred)
print("Training set score : ", rg_model.score(train_X, train_y))

print("Validation set score : ", rg_model.score(val_X, val_y))

# Random Forest

from sklearn.model_selection import cross_val_score



# Lets take full data again

housing_full_data = imputer.fit_transform(X)

housing_full_data_tr=pd.DataFrame(housing_full_data, columns=X.columns)

housing_full_data_tr[num_attributes] = normaliser.fit_transform(housing_full_data_tr[num_attributes])



scores = cross_val_score(rf_model, housing_full_data_tr, y, cv=10)

print('For Random Forest model:')

print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.2f}".format(scores.mean()))

# Ridge model



scores = cross_val_score(rg_model, housing_full_data_tr, y, cv=5)

print('For Ridge model:')

print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.2f}".format(scores.mean()))

from sklearn.model_selection import GridSearchCV



# Tuning Ridge

param_grid = [{'alpha': [0,0.01,0.5,1,5,10,100],'solver':['auto','svd','cholesky']}]



model = Ridge()



# -------------------------------------------------------

grid_search = GridSearchCV(model, param_grid, cv=5)



grid_search.fit(housing_full_data_tr, y)



print("grid_search score: {:.2f}\n".format(grid_search.score(housing_full_data_tr, y)))



print(grid_search.best_params_)



print(grid_search.best_estimator_)
# Build the final model

final_model = grid_search.best_estimator_
# path to file you will use for predictions

test_data_path =  '../input/house-prices-advanced-regression-techniques/test.csv'



test_data = pd.read_csv(test_data_path)



test_X = test_data.copy()

# Repeat process for missing/null values

# Numerical columns:

test_X.MasVnrArea = test_X.MasVnrArea.fillna(0)



# Categorical columns:

for cat in cat_cols_fill_none:

    test_X[cat] = test_X[cat].fillna("None")



attributes_drop = [ 'MiscVal', 'MSSubClass', 'MoSold', 'YrSold', 

                   'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd'] # high corr with other attributes



test_X = test_X.drop(attributes_drop, axis=1)



test_X=pd.get_dummies(test_X)



#Align the test columns same train set columns

final_train, final_test = X.align(test_X, join='left', axis=1)



housing_test_prepared=imputer.transform(final_test)

housing_test_prepared_tr=pd.DataFrame(housing_test_prepared, columns=X.columns)

housing_test_prepared_tr[num_attributes] = normaliser.transform(housing_test_prepared_tr[num_attributes])

test_pred = rg_model.predict(housing_test_prepared_tr)



def inv_y(transformed_y):

    return np.exp(transformed_y)



inv_y(test_pred)
output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': inv_y(test_pred)})



output.to_csv('submission.csv', index=False)