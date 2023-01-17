import numpy as np # linear algebra
import pandas as pd # data processing
from matplotlib import pyplot as plt # data visualization
import seaborn as sns # statistics visualization
from sklearn.compose import ColumnTransformer # feature engineering 
from scipy.stats import skew # skew features
from sklearn.pipeline import Pipeline, make_pipeline # creates "bundle" of functions
from sklearn.impute import SimpleImputer # missing values
from sklearn.preprocessing import OneHotEncoder # encode categorical features
from sklearn.preprocessing import RobustScaler # scale features against outliers
from sklearn.model_selection import train_test_split # split dataset in sub-training and -validation sets
from sklearn.model_selection import GridSearchCV # hyperparameters tuning
from sklearn.metrics import mean_absolute_error # error computation
from sklearn.metrics import mean_squared_error # error computation
from sklearn.metrics import mean_squared_log_error # error computation
from sklearn.model_selection import cross_val_score, KFold # model validation
from sklearn.svm import SVR # machine learning model
from sklearn.ensemble import RandomForestRegressor # machine learning model
from sklearn.ensemble import GradientBoostingRegressor # machine learning model
import xgboost as xgb # machine learning model
from lightgbm import LGBMRegressor # machine learning model
from sklearn.kernel_ridge import KernelRidge # machine learning model
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
#display preview of train dataset
train_df
#display preview of test dataset
test_df
train_df.describe()
train_df.dropna(axis=0, subset=['SalePrice'], inplace=True)
# Copy the target column in a new dataset and drop it from the train dataset to only have features remaining 
y = train_df.SalePrice
train_df.drop(['SalePrice'], axis=1, inplace=True)
# Shape of training data (num_rows, num_columns)
print(train_df.shape)
# Number of missing values in each column of test data
total_train = train_df.isnull().sum().sort_values(ascending=False)
percent_train = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data_train = pd.concat([total_train, percent_train],
                         axis=1, keys=['Total', 'Percent'])
missing_data_train.head(20)
# Shape of test data (num_rows, num_columns)
print(test_df.shape)
# Number of missing values in each column of test data
total_test = (test_df.isnull().sum().sort_values(ascending=False))
percent_test = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total_test, percent_test], 
                         axis=1, keys=['Total', 'Percent'])
missing_data_test.head(20)
#train_df = train_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
#test_df = test_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
dataframe = pd.concat((train_df, test_df), sort=False).reset_index(drop=True)
n_train = train_df.shape[0]
train_df_wPrice = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

#correlation matrix
correlation_matrix = train_df_wPrice.corr(method='spearman')
f, ax = plt.subplots(figsize=(18, 9))

#hiding the upper part of the plot
mask = np.zeros_like(correlation_matrix)
mask[np.triu_indices_from(mask)] = True

#drawing the map
sns.heatmap(correlation_matrix, 
            square=True, mask=mask);
dataframe['TotalArea'] = dataframe['TotalBsmtSF'] + dataframe['1stFlrSF'] + dataframe['2ndFlrSF'] + dataframe['GrLivArea'] + dataframe['GarageArea']
dataframe['Bathrooms'] = dataframe['FullBath'] + dataframe['HalfBath']*0.5 
dataframe['Year average']= (dataframe['YearRemodAdd']+dataframe['YearBuilt'])/2
dataframe['PoolQC'] = dataframe['PoolQC'].fillna("None")
dataframe['MiscFeature'] = dataframe['MiscFeature'].fillna("None")
dataframe['Alley'] = dataframe['Alley'].fillna("None")
dataframe['Fence'] = dataframe['Fence'].fillna("None")
dataframe['FireplaceQu'] = dataframe['FireplaceQu'].fillna("None")
dataframe.head()
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    dataframe[col] = dataframe[col].fillna("None")
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    dataframe[col] = dataframe[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    dataframe[col] = dataframe[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    dataframe[col] = dataframe[col].fillna("None")
# "Cardinality" means the number of unique values in a column
# Selecting categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [col_name for col_name in dataframe.columns if
                    dataframe[col_name].nunique() < 10 and 
                    dataframe[col_name].dtype == "object"]

# Selecting numerical columns
numerical_cols = [col_name for col_name in dataframe.columns if 
                dataframe[col_name].dtype in ['int64', 'float64']]
#Checking the graphic
fig, ax = plt.subplots()
ax.scatter(x = train_df['GrLivArea'], y = y)
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#Deleting outliers
remove = (train_df[(train_df['GrLivArea']>4000) & (y<300000)].index)
dataframe = dataframe.drop(remove)
train_df = train_df.drop(remove)
y = y.drop(remove)

#Checking the graphic again
fig, ax = plt.subplots()
ax.scatter(x = train_df['GrLivArea'], y = y)
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
n_train = n_train-2
#Checking the graphic
fig, ax = plt.subplots()
ax.scatter(x = train_df['OverallQual'], y = y)
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('OverallQual', fontsize=13)
plt.show()
# Check the skew of all numerical features
skewed_feats = dataframe[numerical_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(15)
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #X_full[feat]  += 1
    dataframe[feat] = boxcox1p(dataframe[feat], lam)
train = dataframe[:n_train]
test = dataframe[n_train:]

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(train, y, train_size=0.8, test_size=0.2, random_state=0)

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = test[my_cols].copy()
X_full = train[my_cols].copy()
X_train.head(10)
# Processing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Processing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
kfolds = KFold(n_splits=3, shuffle=True, random_state=1)

def cv_rmse(model, X=X_train):
    rmse = np.sqrt(np.log1p(-cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=kfolds)))
    return (rmse)
# Define model
model = RandomForestRegressor()

# Bundle preprocessing and modeling code in a pipeline
classif = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
classif.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds0 = classif.predict(X_valid)
t_preds0 = classif.predict(X_train)

score0 = mean_absolute_error(y_valid, preds0)
score_log = mean_squared_log_error(y_valid, preds0)
cv_rmse0=cv_rmse(classif)


print('MAE 1:', score0)
print('MSE 1 :', cv_rmse0.mean())
print('RMSLE:', np.sqrt(score_log))

plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_train,t_preds0)
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_valid,preds0);
# Define models
model1 = LGBMRegressor()
model2 = RandomForestRegressor()
model3 = xgb.XGBRegressor()
model4 = KernelRidge()
model5 = GradientBoostingRegressor()
# Bundle preprocessing and modeling code in a pipeline
pipeline1 = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model1)
                             ])
pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model2)
                             ])
pipeline3 = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model3)
                             ])
pipeline4 = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model4)
                             ])
pipeline5 = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model5)
                             ])

# Preprocessing of training data, fit model 
pipeline1.fit(X_train, y_train)
pipeline2.fit(X_train, y_train)
pipeline3.fit(X_train, y_train)
pipeline4.fit(X_train, y_train)
pipeline5.fit(X_train, y_train)


# Preprocessing of validation data, and getting predictions
preds1 = pipeline1.predict(X_valid)
t_preds1 = pipeline1.predict(X_train)

preds2 = pipeline2.predict(X_valid)
t_preds2 = pipeline2.predict(X_train)

preds3 = pipeline3.predict(X_valid)
t_preds3 = pipeline3.predict(X_train)

preds4 = pipeline4.predict(X_valid)
t_preds4 = pipeline4.predict(X_train)

preds5 = pipeline5.predict(X_valid)
t_preds5 = pipeline5.predict(X_train)
score1 = mean_absolute_error(y_valid, preds1)
score_log = mean_squared_log_error(y_valid, preds1)
cv_rmse1=cv_rmse(pipeline1)


print('MAE 1:', score1)
print('MSE 1:', cv_rmse1.mean())
print('RMSLE:', np.sqrt(score_log))


plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_train,t_preds1, label='train set')
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_valid,preds1, label='validation set')
plt.legend();
score2 = mean_absolute_error(y_valid, preds2)
score_log = mean_squared_log_error(y_valid, preds2)
cv_rmse2=cv_rmse(pipeline2)

print('MAE 2:', score2)
print('MSE 2:', cv_rmse2.mean())
print('RMSLE:', np.sqrt(score_log))

plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_train,t_preds2)
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_valid,preds2);
score3 = mean_absolute_error(y_valid, preds3)
score_log = mean_squared_log_error(y_valid, preds3)
cv_rmse3=cv_rmse(pipeline3)

print('MAE 3:', score3)
print('MSE 3:', cv_rmse3.mean())
print('RMSLE:', np.sqrt(score_log))

plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_train,t_preds3)
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_valid,preds3);
score4 = mean_absolute_error(y_valid, preds4)
score_log = mean_squared_log_error(y_valid, abs(preds4))
cv_rmse4=cv_rmse(pipeline4)

print('MAE 4:', score4)
print('MSE 4:', cv_rmse4.mean())
print('RMSLE:', np.sqrt(score_log))

plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_train,t_preds4)
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_valid,preds4);
score5 = mean_absolute_error(y_valid, preds5)
score_log = mean_squared_log_error(y_valid, preds5)
cv_rmse5 = cv_rmse(pipeline5)

print('MAE 5:', score5)
print('MSE 5:', cv_rmse5.mean())
print('RMSLE:', np.sqrt(score_log))

plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_train,t_preds5)
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_valid,preds5);
# Printing the actual parameters in use (defaults)
print('Parameters currently in use:\n')
print(model2.get_params())

# Creating a grid which contains the values the hyperparameters can take
param_grid2 = {'min_samples_split' : [2,4,7,10], 'n_estimators' : [50,75,100]}

# Using GridSearchCV with the former grid
CV_rfr = GridSearchCV(model2, param_grid2)
pipeline2cv = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', CV_rfr)
                           ])

# Fitting the model to all the training data to get better results for test 
pipeline2cv.fit(X_train, y_train)
print(CV_rfr.best_params_)

preds2cv = pipeline2cv.predict(X_valid)
t_preds2cv = pipeline2cv.predict(X_train)
score2cv = mean_absolute_error(y_valid, preds2cv)
score_log = mean_squared_log_error(y_valid, preds2cv)
cv_rmse2cv=cv_rmse(pipeline2cv)

print('MAE 2cv:', score2cv)
print('MSE 2cv:', cv_rmse2cv.mean())
print('RMSLE:', np.sqrt(score_log))

plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_train,t_preds2cv)
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_valid,preds2cv);
print('Parameters currently in use:\n')
print(model3.get_params())

param_grid3 = {'learning_rate':[0.005,0.01,0.05,0.1], 
              'n_estimators':[50,75,100],}

CV_xgbr = GridSearchCV(model3, param_grid3)
pipeline3cv = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', CV_xgbr)
                             ])
pipeline3cv.fit(X_train, y_train)
print(CV_xgbr.best_params_)

preds3cv = pipeline3cv.predict(X_valid)
t_preds3cv = pipeline3cv.predict(X_train)
score3cv = mean_absolute_error(y_valid, preds3cv)
score_log = mean_squared_log_error(y_valid, preds3cv)
cv_rmse3cv = cv_rmse(pipeline3cv)

print('MAE 3cv:', score3cv)
print('MSE 3cv:', cv_rmse3cv.mean())
print('RMSLE:', np.sqrt(score_log))

plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_train,t_preds3cv)
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_valid,preds3cv);
print('Parameters currently in use:\n')
print(model4.get_params())

param_grid = {'alpha': [0.5, 0.75, 1], 'kernel': ['linear','polynomial']}

CV_krr = GridSearchCV(model4, param_grid)
pipeline4cv = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', CV_krr)
                            ])
pipeline4cv.fit(X_train, y_train)
print(CV_krr.best_params_)

preds4cv = pipeline4cv.predict(X_valid)
t_preds4cv = pipeline4cv.predict(X_train)
score4cv = mean_absolute_error(y_valid, preds4cv)
score_log = mean_squared_log_error(y_valid, abs(preds4cv))
cv_rmse4cv = cv_rmse(pipeline4cv)

print('MAE 4cv:', score4cv)
print('MSE 4cv:', cv_rmse4cv.mean())
print('RMSLE:', np.sqrt(score_log))

plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_train,t_preds4cv)
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_valid,preds4cv);
print('Parameters currently in use:\n')
print(model5.get_params())

param_grid = {'learning_rate': [0.05, 0.1, 0.5], 'loss': ['ls','huber'], 
              'n_estimators': [50, 75, 100]}

CV_gbr = GridSearchCV(model5, param_grid)
pipeline5cv = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', CV_gbr)
                            ])
pipeline5cv.fit(X_train, y_train)
print(CV_gbr.best_params_)

preds5cv = pipeline5cv.predict(X_valid)
t_preds5cv = pipeline5cv.predict(X_train)
score5cv = mean_absolute_error(y_valid, preds5cv)
score_log = mean_squared_log_error(y_valid, preds5cv)
cv_rmse5cv = cv_rmse(pipeline5cv)

print('MAE 5cv:', score5cv)
print('MSE 5cv:', cv_rmse5cv.mean())
print('RMSLE:', np.sqrt(score_log))

plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_train,t_preds5cv)
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(y_valid,preds5cv);
# Preprocessing of test data, fitting model on all the train data to get even more accurate model
pipeline3cv.fit(X_full,y)
preds_test = (pipeline3cv.predict(X_test))
# Save test predictions to file
output = pd.DataFrame({'Id': test_df.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
output.head(20)
output.describe()