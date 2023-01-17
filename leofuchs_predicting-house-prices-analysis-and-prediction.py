# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import scipy.stats as st
import seaborn as sns
import warnings

%matplotlib inline

pd.options.display.max_columns = None
warnings.filterwarnings('ignore')
color = sns.color_palette()
sns.set_style('darkgrid')

# Importing the train and test datasets in pandas dataframes
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# Drop the 'Id' column from the train dataframe
train_data.drop(columns='Id', inplace=True)

y_train = train_data['SalePrice']
# The shape of the data
train_data.shape, test_data.shape, y_train.shape
# Display the first five rows of the training dataset.
train_data.head()
# The description of the train dataset
train_data.describe()
# Looking the type of the columns in the dataset
train_data.info()
# Showing the numerical varibales with the highest correlation with 'SalePrice', sorted from highest to lowest
correlation = train_data.select_dtypes(include=[np.number]).corr()

print(correlation['SalePrice'].sort_values(ascending=False))
# Heatmap of correlation of numeric features
fig, ax = plt.subplots(figsize = (14,14))

plt.title('Correlation Between Numeric Features', size=15)
sns.heatmap(correlation, square=True, vmax=0.8, cmap='coolwarm', linewidths=0.01);
# Zoomed HeatMap of the most Correlayed variables
zoomed_correlation = correlation.loc[['SalePrice','GrLivArea','TotalBsmtSF','OverallQual','FullBath','TotRmsAbvGrd','YearBuilt', 'YearRemodAdd', '1stFlrSF','GarageYrBlt','GarageCars','GarageArea'],
                                     ['SalePrice','GrLivArea','TotalBsmtSF','OverallQual','FullBath','TotRmsAbvGrd','YearBuilt', 'YearRemodAdd', '1stFlrSF','GarageYrBlt','GarageCars','GarageArea']]

fig , ax = plt.subplots(figsize = (14,14))
plt.title('Zoomed Correlation Between Numeric Features', size=15)
sns.heatmap(zoomed_correlation, square=True, vmax=0.8, annot=True, cmap='coolwarm', linewidths=0.01);
# Pair plot
cols = ['SalePrice','GrLivArea','TotalBsmtSF','OverallQual','FullBath','TotRmsAbvGrd','YearBuilt', 'YearRemodAdd', '1stFlrSF','GarageYrBlt','GarageCars','GarageArea']

sns.set()
sns.pairplot(train_data[cols], size=2, kind='scatter', diag_kind='kde');
plt.figure(figsize=(25,5))

ax1 = plt.subplot(1, 3, 1)
plt.scatter(x=train_data.TotalBsmtSF, y=train_data.SalePrice)
plt.title('TotalBsmtSF x SalePrice', size=15)

ax2 = plt.subplot(1, 3, 2)
plt.scatter(x=train_data['1stFlrSF'], y=train_data.SalePrice)
plt.title('1stFlrSF x SalePrice', size=15)

ax3 = plt.subplot(1, 3, 3)
plt.scatter(x = train_data.GrLivArea, y=train_data.SalePrice)
plt.title('GrLivArea x SalePrice', size=15)

plt.show()
print(train_data.shape)

# Removing the four outliers found 
train_data.drop(train_data[train_data['TotalBsmtSF'] > 5000].index, inplace=True)
train_data.drop(train_data[train_data['1stFlrSF'] > 4000].index,inplace=True)
train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000)].index, inplace = True)

print(train_data.shape)
# Visualising missing values of numeric features
msno.matrix(train_data.select_dtypes(include=[np.number]));
# Visualising percentage of missing values of the top 5 numeric variables
total = train_data.select_dtypes(include=[np.number]).isnull().sum().sort_values(ascending=False)
percent = (train_data.select_dtypes(include=[np.number]).isnull().sum() / train_data.select_dtypes(include=[np.number]).isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, join='outer', keys=['Missing Count', 'Missing Percentage'])
missing_data.index.name=' Numeric Feature'
missing_data.head(5)
# Visualising missing values of categorical features
msno.matrix(train_data.select_dtypes(include=[np.object]));
# Visualising percentage of missing values of the top 10 categorical variables
total = train_data.select_dtypes(include=[np.object]).isnull().sum().sort_values(ascending=False)
percent = (train_data.select_dtypes(include=[np.object]).isnull().sum() / train_data.select_dtypes(include=[np.object]).isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Missing Count', 'Missing Percentage'])
missing_data.index.name =' Object Feature'
missing_data.head(20)
# Concatenate the training and test datasets into a single dataframe
data_full = pd.concat([train_data,test_data], ignore_index=True)
data_full.drop('Id', axis=1, inplace=True)

data_full.shape
# Sum of missing values by numeric features
sum_missing_values = data_full.select_dtypes(include=[np.number]).isnull().sum()

sum_missing_values[sum_missing_values > 0].sort_values(ascending=False)
# Numeric features with small number of NaNs: replace with 0
for col in ['BsmtHalfBath', 'BsmtFullBath', 'GarageArea', 'GarageCars', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']:
    data_full[col].fillna(0, inplace=True)

# Check if missing values are imputed successfully
sum_missing_values = data_full.select_dtypes(include=[np.number]).isnull().sum()
sum_missing_values[sum_missing_values > 0].sort_values(ascending=False)
# Numeric features with medium number of NaNs: replace with the mean
data_full['MasVnrArea'].fillna(data_full['MasVnrArea'].mean(), inplace=True)

# Check if missing values are imputed successfully
sum_missing_values = data_full.select_dtypes(include=[np.number]).isnull().sum()
sum_missing_values[sum_missing_values > 0].sort_values(ascending=False)
# Cut 'YearBuilt' into 10 parts
data_full['YearBuiltCut'] = pd.qcut(data_full['YearBuilt'], 10)

# Impute the missing values of 'GarageYrBlt' based on the median of 'YearBuilt' 
data_full['GarageYrBlt'] = data_full.groupby(['YearBuiltCut'])['GarageYrBlt'].transform(lambda x : x.fillna(x.median()))

# Convert the values to integers
data_full['GarageYrBlt'] = data_full['GarageYrBlt'].astype(int)

# Drop 'YearBuiltCut' column
data_full.drop('YearBuiltCut', axis=1, inplace=True)

# Check if missing values are imputed successfully
sum_missing_values = data_full.select_dtypes(include=[np.number]).isnull().sum()
sum_missing_values[sum_missing_values > 0].sort_values(ascending=False)
# Cut 'LotArea' into 10 parts
data_full['LotAreaCut'] = pd.qcut(data_full['LotArea'], 10)

# Impute the missing values of 'LotFrontage' based on the median of 'LotArea' and 'Neighborhood'
data_full['LotFrontage'] = data_full.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x : x.fillna(x.median()))
data_full['LotFrontage'] = data_full.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x : x.fillna(x.median()))

# Drop 'LotAreaCut' column
data_full.drop('LotAreaCut',axis=1,inplace=True)

# Check if missing values are imputed successfully
sum_missing_values = data_full.select_dtypes(include=[np.number]).isnull().sum()
sum_missing_values[sum_missing_values > 0].sort_values(ascending=False)
# Sum of missing values by feature (object)
sum_missing_values = data_full.select_dtypes(include=[np.object]).isnull().sum()
sum_missing_values[sum_missing_values > 0].sort_values(ascending=False)
# Categorical features with less than 5 missing values: replace with the mode (most frequently occured value)
for col in ['MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'SaleType', 'Exterior2nd', 'KitchenQual', 'Electrical']:
    data_full[col].fillna(data_full[col].mode()[0], inplace=True)

# Check if missing values are imputed successfully
sum_missing_values = data_full.select_dtypes(include=[np.object]).isnull().sum()
sum_missing_values[sum_missing_values > 0].sort_values(ascending=False)
# Categorical features with more than 5 missing values: replace with 'None'
for col in ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageQual','GarageCond','GarageFinish','GarageType','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','MasVnrType']:
    data_full[col].fillna('None', inplace=True)

# Check if missing values are imputed successfully
sum_missing_values = data_full.select_dtypes(include=[np.object]).isnull().sum()
sum_missing_values[sum_missing_values > 0].sort_values(ascending=False)
data_full.select_dtypes(include=[np.number]).columns
# Converting numeric features to categorical features
str_cols = ['YrSold','YearRemodAdd','YearBuilt','MoSold','MSSubClass','GarageYrBlt']

for col in str_cols:
    data_full[col] = data_full[col].astype(str)
data_full.select_dtypes(include=[np.object]).columns
data_full['GarageCond'].unique()
# ExterQual = Evaluates the quality of the material on the exterior: Ex(Excellent), Gd(Good), TA(Typical), Fa(Fair), Po(Poor)
data_full["oExterQual"] = data_full['ExterQual'].map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})

# ExterCond = Evaluates the present condition of the material on the exterior: Ex(Excellent), Gd(Good), TA(Typical), Fa(Fair), Po(Poor)
data_full["oExterCond"] = data_full['ExterCond'].map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})

# BsmtQual = Evaluates the height of the basement: Ex(Excellent), Gd(Good), TA(Typical), Fa(Fair), Po(Poor), NA(No Basement)
data_full["oBsmtQual"] = data_full['BsmtQual'].map({'None':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})

# BsmtExposure = Refers to walkout or garden level walls: Gd(Good), Av(Average), Mn(Minimum), No(No Exposure), NA(No Basement)
data_full["oBsmtExposure"] = data_full['BsmtExposure'].map({'None':1, 'No':2, 'Av':3, 'Mn':3, 'Gd':4})

# BsmtCond = Evaluates the general condition of the basement: Ex(Excellent), Gd(Good), TA(Typical), Fa(Fair), Po(Poor), NA(No Basement)
data_full["oBsmtCond"] = data_full['BsmtCond'].map({'None':1, 'Po':2, 'Fa':3, 'TA':3, 'Gd':4})

# HeatingQC = Heating quality and condition: Ex(Excellent), Gd(Good), TA(Average), Fa(Fair), Po(Poor)
data_full["oHeatingQC"] = data_full['HeatingQC'].map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})

# KitchenQual: Kitchen quality: Ex(Excellent), Gd(Good), TA(Typical), Fa(Fair), Po(Poor)
data_full["oKitchenQual"] = data_full['KitchenQual'].map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})

# FireplaceQu: Fireplace quality: Ex(Excellent), Gd(Good), TA(Average), Fa(Fair), Po(Poor), NA(No Fireplace)
data_full["oFireplaceQu"] = data_full['FireplaceQu'].map({'None':1, 'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})

# GarageFinish: Interior finish of the garage: Fin(Finished), RFn(Rough Finished), Unf(Unfinished), NA(No Garage)
data_full["oGarageFinish"] = data_full['GarageFinish'].map({'None':1, 'Unf':2, 'RFn':3, 'Fin':4})

# GarageQual: Garage quality: Ex(Excellent), Gd(Good), TA(Typical), Fa(Fair), Po(Poor), NA(No Garage)
data_full["oGarageQual"] = data_full['GarageQual'].map({'None':1, 'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})

# GarageCond: Garage condition: Ex(Excellent), Gd(Good), TA(Typical), Fa(Fair), Po(Poor), NA(No Garage)
data_full["oGarageCond"] = data_full['GarageCond'].map({'None':1, 'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})

# PavedDrive: Paved driveway: Y(Padev), P(Partial Pavement), N(Dirt)
data_full["oPavedDrive"] = data_full['PavedDrive'].map({'N':1, 'P':2, 'Y':3})
data_full.select_dtypes(include=[np.number]).columns
# House square feet = First floor square feet + Second floor square feet + Total square feet of basement area
data_full['HouseSF'] = data_full['1stFlrSF'] + data_full['2ndFlrSF'] + data_full['TotalBsmtSF']

# Porch square feet = Three season porch area in square feet + Enclosed porch area in square feet + Screen porch area in square feet
data_full['PorchSF'] = data_full['3SsnPorch'] + data_full['EnclosedPorch'] + data_full['OpenPorchSF'] + data_full['ScreenPorch']

# Total square feet = House square feet + Porch square feet + Garage area
data_full['TotalSF'] = data_full['HouseSF'] + data_full['PorchSF'] + data_full['GarageArea']
# Estimate Skewness of the data
train_data.skew()
# Estimate Kurtosis of the data
train_data.kurt()
# Plot the Skewness and Kurtosis of the data
plt.figure(figsize=(15,5))

ax1 = plt.subplot(1, 2, 1)
sns.distplot(train_data.skew(), axlabel ='Skewness')

ax2 = plt.subplot(1, 2, 2)
sns.distplot(train_data.kurt(), axlabel ='Kurtosis')

plt.show()
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew

# Label encoding class
class labenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        label = LabelEncoder()
        
        X['YrSold'] = label.fit_transform(X['YrSold'])
        X['YearRemodAdd'] = label.fit_transform(X['YearRemodAdd'])
        X['YearBuilt'] = label.fit_transform(X['YearBuilt'])
        X['MoSold'] = label.fit_transform(X['MoSold'])
        X['GarageYrBlt'] = label.fit_transform(X['GarageYrBlt'])
        
        return X
    
# Skewness transform class
class skewness(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        skewness = X.select_dtypes(include=[np.number]).apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= 1].index
        
        X[skewness_features] = np.log1p(X[skewness_features])
        
        return X

# One hot encoding class
class onehotenc(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.get_dummies(X)
        
        return X
# Creating a copy of the full dataset
data_full_copy = data_full.copy()

# Creating a new data with the applied transformations using a Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([('labenc', labenc()), ('skewness', skewness()), ('onehotenc', onehotenc())])

data_pipeline = pipeline.fit_transform(data_full_copy)

data_full.shape, data_pipeline.shape
data_full.head()
data_pipeline.head()
X_train = data_pipeline[:train_data.shape[0]]
y_train = X_train['SalePrice']
X_train.drop(columns='SalePrice', inplace=True)

X_test = data_pipeline[train_data.shape[0]:]
X_test.drop(columns='SalePrice', inplace=True)

X_train.shape, y_train.shape, X_test.shape
plt.figure(figsize=(25,5))

ax1 = plt.subplot(1, 3, 1)
sns.distplot(y_train, kde=False, fit=st.norm)
plt.title('Normal', size = 15)

ax2 = plt.subplot(1, 3, 2)
sns.distplot(y_train, kde=False, fit=st.lognorm)
plt.title('Log Normal', size = 15)

ax3 = plt.subplot(1, 3, 3)
sns.distplot(y_train, kde=False, fit=st.johnsonsu)
plt.title('Johnson SU', size = 15)

plt.show()
# Transforming 'SalePrice' into normal distribution
y_train_transformed = np.log(y_train)

y_train_transformed.skew(), y_train_transformed.kurt()
# Plotting 'SalePrice' before and after the transformation
plt.figure(figsize=(15,5))

ax1 = plt.subplot(1, 2, 1)
sns.distplot(y_train)
plt.title('Before Transformation', size=15)

ax2 = plt.subplot(1, 2, 2)
sns.distplot(y_train_transformed)
plt.title('After Transformation', size=15)

plt.show()
# Using RobustScaler to transform X_train and X_test
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()

X_train_scaled = robust_scaler.fit(X_train).transform(X_train)
X_test_scaled = robust_scaler.transform(X_test)
# Shape of final data we will be working on
X_train_scaled.shape, y_train_transformed.shape, X_test_scaled.shape
# Display features by their importance (lasso regression coefficient)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.001)

lasso.fit(X_train_scaled, y_train_transformed)

y_pred_lasso = lasso.predict(X_test_scaled)

lasso_coeff = pd.DataFrame({'Feature Importance':lasso.coef_}, index=data_pipeline.drop(columns='SalePrice').columns)
lasso_coeff.sort_values('Feature Importance', ascending=False)
# Plot features by importance (feature coefficient in the model)
lasso_coeff[lasso_coeff['Feature Importance'] != 0].sort_values('Feature Importance').plot(kind='barh',figsize=(20,20))
from sklearn.decomposition import PCA

# Concatenate the training and test datasets into a single dataframe
data_full_2 = np.concatenate([X_train_scaled, X_test_scaled])

# Choose the number of principle components such that 95% of the variance is retained
pca = PCA(0.95)
data_full_2 = pca.fit_transform(data_full_2)

var_PCA = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

# Principal Component Analysis of data
print(var_PCA)
# Principal Component Analysis plot of the data
plt.figure(figsize=(15,5))

plt.bar(x=range(1, len(var_PCA) + 1), height=var_PCA)
plt.ylabel("Explained Variance (%)", size=15)
plt.xlabel("Principle Components", size=15)
plt.title("Principle Component Analysis Plot : Training Data", size=15)
plt.show()
# Shape of final data we will be working on
X_train_scaled = data_full_2[:train_data.shape[0]]

X_test_scaled = data_full_2[train_data.shape[0]:]

X_train_scaled.shape, y_train_transformed.shape, X_test_scaled.shape
# Importing the models
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet, Lasso, SGDRegressor, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR, SVR

# kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


#alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
#alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
#e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
#e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


# Adicionar RidgeCV(alpha=alphas_alt, cv=kfolds)
# Adicionar LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds)
# Adicionar ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio)
# Adicionar SVR(C= 20, epsilon= 0.008, gamma=0.0003,)
# Adicionar GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)
# Adicionar LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200, bagging_fraction=0.75, bagging_freq=5, bagging_seed=7, feature_fraction=0.2,
# feature_fraction_seed=7, verbose=-1,)

# Adicionar XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                    # max_depth=3, min_child_weight=0,
                                    # gamma=0, subsample=0.7,
                                    # colsample_bytree=0.7,
                                    # objective='reg:linear', nthread=-1,
                                    # scale_pos_weight=1, seed=27,
                                    # reg_alpha=0.00006)
                        
# StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                               # meta_regressor=xgboost,
                               # use_features_in_secondary=True)
                        
# Creating the models
models = [LinearRegression(), 
          SVR(),
          SGDRegressor(),
          SGDRegressor(max_iter=1000, tol=1e-3),
          GradientBoostingRegressor(),
          RandomForestRegressor(),
          Lasso(),
          Lasso(alpha=0.01, max_iter=10000),
          Ridge(),
          BayesianRidge(),
          KernelRidge(),
          KernelRidge(alpha=0.6, kernel='polynomial',degree=2, coef0=2.5),
          ElasticNet(),
          ElasticNet(alpha=0.001, max_iter=10000), ExtraTreesRegressor()
         ]

names = ['Linear Regression',
         'Support Vector Regression',
         'Stochastic Gradient Descent',
         'Stochastic Gradient Descent 2',
         'Gradient Boosting Tree',
         'Random Forest',
         'Lasso Regression',
         'Lasso Regression 2',
         'Ridge Regression',
         'Bayesian Ridge Regression',
         'Kernel Ridge Regression',
         'Kernel Ridge Regression 2',
         'Elastic Net Regularization',
         'Elastic Net Regularization 2',
         'Extra Trees Regression'
        ]
# Define a root mean square error function
def rmse(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
from sklearn.model_selection import KFold, cross_val_score
warnings.filterwarnings('ignore')

# Perform 5-folds cross-validation to evaluate the models 
for model, name in zip(models, names):
    # Root mean square error
    score = rmse(model, X_train_scaled, y_train_transformed)
    print("- {}: Mean: {:.6f}, Std: {:4f}".format(name, score.mean(), score.std()))
from sklearn.model_selection import GridSearchCV

class gridSearch():
    def __init__(self, model):
        self.model = model
    def grid_get(self, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train_transformed)
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        
        #print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
        print('Best Parameters: {}, \nBest Score: {}'.format(grid_search.best_params_, np.sqrt(-grid_search.best_score_)))
gridSearch(KernelRidge()).grid_get({'alpha':[3.5, 4, 4.5, 5, 5.5, 6, 6.5], 'kernel':["polynomial"], 'degree':[3], 'coef0':[1, 1.5, 2, 2.5, 3, 3.5]})
gridSearch(ElasticNet()).grid_get({'alpha':[0.006, 0.0065, 0.007, 0.0075, 0.008], 'l1_ratio':[0.070, 0.075, 0.080, 0.085, 0.09, 0.095], 'max_iter':[10000]})
gridSearch(Ridge()).grid_get({'alpha':[10, 20, 25, 30, 35, 40, 45, 50, 55, 57, 60, 65, 70, 75, 80, 100], 'max_iter':[10000]})
gridSearch(SVR()).grid_get({'C':[13, 15, 17, 19, 21], 'kernel':["rbf"], "gamma":[0.0005, 0.001, 0.002, 0.01], "epsilon":[0.01, 0.02, 0.03, 0.1]})
gridSearch(Lasso()).grid_get({'alpha':[0.01, 0.001, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009], 'max_iter':[10000]})
ker = KernelRidge(alpha=6.5, coef0=2.5, degree=3, kernel='polynomial')
ela = ElasticNet(alpha=0.007, l1_ratio=0.07, max_iter=10000)
ridge = Ridge(alpha=35, max_iter= 10000)
svr = SVR(C=13, epsilon=0.03, gamma=0.001, kernel='rbf')
lasso = Lasso(alpha=0.0006, max_iter=10000)
bay = BayesianRidge()
# Create the model (Random Forest Classifier) and run with the train data
model = SVR(C=13, epsilon=0.03, gamma=0.001, kernel='rbf')
model.fit(X_train_scaled, y_train_transformed)

# Generate the predictions running the model in the test data
predictions = np.exp(model.predict(X_test_scaled))

# Create the output file 
output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")