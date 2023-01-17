import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# Read the data
X_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

X_full.head()
X_full.shape
X_full.columns
#missing values
missing = X_full.isnull().sum()
missing = missing[missing>0]
missing.sort_values(inplace=True)
missing.plot.bar()
numerical_features = X_full.select_dtypes(exclude=['object']).drop(['SalePrice'], axis=1).copy()
print(numerical_features.columns)
print(numerical_features.shape)
categorical_features = X_full.select_dtypes(include=['object']).copy()
print(categorical_features.columns)
print(categorical_features.shape)
fig = plt.figure(figsize=(12,18))
for i in range(len(numerical_features.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.scatterplot(numerical_features.iloc[:, i],X_full['SalePrice'])
plt.tight_layout()
plt.show()
figure, axs = plt.subplots(8, 2)
figure.set_size_inches(20,30)
_ = sns.regplot(X_full['LotFrontage'], X_full['SalePrice'], ax=axs[0,0])
_ = sns.regplot(X_full['LotArea'], X_full['SalePrice'], ax=axs[0,1])
_ = sns.regplot(X_full['MasVnrArea'], X_full['SalePrice'], ax=axs[1,0])
_ = sns.regplot(X_full['BsmtFinSF1'], X_full['SalePrice'], ax=axs[1,1])
_ = sns.regplot(X_full['BsmtFinSF2'], X_full['SalePrice'], ax=axs[2,0])
_ = sns.regplot(X_full['TotalBsmtSF'], X_full['SalePrice'], ax=axs[2,1])
_ = sns.regplot(X_full['1stFlrSF'], X_full['SalePrice'], ax=axs[3,0])
_ = sns.regplot(X_full['2ndFlrSF'], X_full['SalePrice'], ax=axs[3,1])
_ = sns.regplot(X_full['LowQualFinSF'], X_full['SalePrice'], ax=axs[4,0])
_ = sns.regplot(X_full['GrLivArea'], X_full['SalePrice'], ax=axs[4,1])
_ = sns.regplot(X_full['WoodDeckSF'], X_full['SalePrice'], ax=axs[5,0])
_ = sns.regplot(X_full['OpenPorchSF'], X_full['SalePrice'], ax=axs[5,1])
_ = sns.regplot(X_full['EnclosedPorch'], X_full['SalePrice'], ax=axs[6,0])
_ = sns.regplot(X_full['3SsnPorch'], X_full['SalePrice'], ax=axs[6,1])
_ = sns.regplot(X_full['MiscVal'], X_full['SalePrice'], ax=axs[7,0])

X_full = X_full.drop(X_full[X_full['LotFrontage']>200].index)
X_full = X_full.drop(X_full[X_full['LotArea']>100000].index)
X_full = X_full.drop(X_full[X_full['MasVnrArea']>1200].index)
X_full = X_full.drop(X_full[X_full['BsmtFinSF1']>4000].index)
X_full = X_full.drop(X_full[X_full['TotalBsmtSF']>4000].index)
X_full = X_full.drop(X_full[(X_full['LowQualFinSF']>600) & (X_full['SalePrice']>400000)].index)
X_full = X_full.drop(X_full[(X_full['GrLivArea']>4000) & (X_full['SalePrice']<300000)].index)
X_full = X_full.drop(X_full[X_full['EnclosedPorch']>500].index)
X_full = X_full.drop(X_full[X_full['MiscVal']>5000].index)

X_full.shape

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)
# X_full.drop(columns=['Alley','MiscFeature','PoolQC'], axis=1, inplace=True)
# X_test_full.drop(columns=['Alley','MiscFeature','PoolQC'], axis=1, inplace=True)
# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
X_train.head()
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
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

# Define model
model_1 = RandomForestRegressor(n_estimators=100, random_state=1)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model_1)
                             ])

my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)

mae, rmsle = mean_absolute_error(y_valid, preds), np.sqrt(mean_squared_log_error( y_valid, preds ))
print("Model 1 MAE: %d, RMSLE: %f" % (mae, rmsle))
# from sklearn.model_selection import cross_val_score

# def get_score(n_estimators):
#     my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
#                               ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
#                              ])
#     scores = -1 * cross_val_score(my_pipeline, X_full, y,
#                               cv=5,
#                               scoring='neg_mean_absolute_error')
#     return scores.mean()

# results = {}
# for i in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
#     results[i] = get_score(i)

# print(results)
# model_3 = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# model_3.fit(X_train, y_train)

# predictions = model_3.predict(X_valid)
# print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
# # Preprocessing of test data, fit model
# preds_test = my_pipeline.predict(X_test)

# # Save test predictions to file
# output = pd.DataFrame({'Id': X_test.index,
#                        'SalePrice': preds_test})
# output.to_csv('submission.csv', index=False)