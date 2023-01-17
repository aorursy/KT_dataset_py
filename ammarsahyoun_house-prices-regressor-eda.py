import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
pd.pandas.set_option("display.max_columns", None)
from scipy.stats import skew, kurtosis
from scipy import stats

training = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.DataFrame(columns=["Id", "SalePrice"])
submission["Id"] = test["Id"]
training.head()
train = training.drop(['SalePrice'], axis=1)
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

features = pd.concat([train, test]).reset_index(drop=True)
print("The whole features shape: ",features.shape)
categFeat = features.describe(include = ['object']).columns.values
print(f"We have: {categFeat.size} categorical features.")
features.describe(include = ['object'])
sns.boxplot(x='PoolQC',y=training['SalePrice'], data=training)
features['PoolQC'].isnull().mean()
sns.distplot(training['SalePrice'], fit=norm)
print(f"Skewness: {training['SalePrice'].skew()}")
print(f"Kurtosis: {training['SalePrice'].kurt()}")
numerFeat = [var for var in features.columns if features[var].dtypes != 'O']
print(f"We have {len(numerFeat)} numerical features including the Id and SalePrice: ")
features[numerFeat].describe()
yearFeat = [var for var in numerFeat if 'Yr' in var or 'Year' in var]

def plotYearVariable(df, var):
    df[var] = df["YrSold"]-df[var]
    plt.scatter(df[var], df["SalePrice"])
    plt.ylabel("Sale Price")
    plt.xlabel(var)
    plt.show()

for var in yearFeat:
    if var !="YrSold":
        plotYearVariable(training, var)
discFeat = [var for var in numerFeat if len(
    features[var].unique()) < 20 and var not in yearFeat+['Id']]
print(f"We have: '{len(discFeat)} discrete features")
contFeat = [
    var for var in numerFeat if var not in discFeat + yearFeat + ['Id']]

print(f"We have: {len(contFeat)} continuous features")
def analyse_discrete(df, var):
    df = df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylabel('Median SalePrice')
    plt.show()
    
for var in discFeat:
    analyse_discrete(training, var)
def analyse_continuous(df, var):
    df[var].hist(bins=30)
    plt.ylabel('Number of houses')
    plt.xlabel(var)
    plt.title(var)
    plt.show()

for var in contFeat:
    analyse_continuous(features, var)
def outliers(df, var):
    if any(df[var] <= 0):
        pass
    else:
        df[var] = np.log(df[var])
        df.boxplot(column=var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()

for var in contFeat:
    outliers(features, var)
corr = training.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);
vars_with_naN = [var for var in features.columns if features[var].isnull().sum() > 0]
print(f"We have {len(vars_with_naN)} features with NaN value:\n\n {vars_with_naN}")
# features[vars_with_naN]isnull().mean()
NanVariables = [var for var in training.columns if training[var].isnull().sum() > 0]
training[NanVariables].isnull().mean()
features['PoolQC'].fillna("NA", inplace= True)
features['MiscFeature'].fillna("NA", inplace= True)
features['Alley'].fillna("NA", inplace=True)
features['Fence'].fillna("NA", inplace= True)
features['FireplaceQu'].fillna("NA", inplace= True)
median = features['LotFrontage'].median()
features['LotFrontage'].fillna(median, inplace=True)

features['GarageCond'].fillna('NA', inplace= True)
features['GarageFinish'].fillna('NA', inplace= True)
features['GarageQual'].fillna('NA', inplace= True)
features['GarageType'].fillna('NA', inplace= True)
features['GarageYrBlt'].fillna(0, inplace= True)

features['BsmtExposure'].fillna('NA', inplace= True)
features['BsmtCond'].fillna('NA', inplace= True)
features['BsmtQual'].fillna('NA', inplace= True)
features['BsmtFinType1'].fillna('NA', inplace= True)
features['BsmtFinType2'].fillna('NA', inplace= True)

features['MasVnrType'].fillna('None', inplace= True)
features['MasVnrArea'].fillna(0, inplace= True)

features['MSZoning'].fillna(features['MSZoning'].mode()[0], inplace= True)
features['Functional'].fillna(features['Functional'].mode()[0], inplace= True)
features['BsmtHalfBath'].fillna(features['BsmtHalfBath'].mode()[0], inplace= True)
features['BsmtFullBath'].fillna(features['BsmtFullBath'].mode()[0], inplace= True)

features['Utilities'].fillna(features['Utilities'].mode()[0], inplace= True)
features['Electrical'].fillna(features['Electrical'].mode()[0], inplace= True)
features['Exterior1st'].fillna(features['Exterior1st'].mode()[0], inplace= True)
features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0], inplace= True)

features['GarageCars'].fillna(features['GarageCars'].mode()[0], inplace= True)
features['GarageArea'].fillna(features['GarageArea'].mode()[0], inplace= True)
features['KitchenQual'].fillna(features['KitchenQual'].mode()[0], inplace= True)
features['BsmtFinSF1'].fillna(features['BsmtFinSF1'].mode()[0], inplace= True)

features['SaleType'].fillna(features['SaleType'].mode()[0], inplace= True)
features['TotalBsmtSF'].fillna(features['TotalBsmtSF'].mode()[0], inplace= True)
features['BsmtUnfSF'].fillna(features['BsmtUnfSF'].mode()[0], inplace= True)
features['BsmtFinSF2'].fillna(features['BsmtFinSF2'].mode()[0], inplace= True)
categorical_columns = features.select_dtypes(include= ['object']).columns
print(categorical_columns)
one_hot_parameters = ['MSZoning' ,'Street', 'Alley', 'LandContour','LotConfig', 'Neighborhood','Condition1', 'Condition2',
                      'RoofStyle', 'RoofMatl','Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating','GarageType', 
                     'PavedDrive', 'MiscFeature','SaleType', 'SaleCondition' ]
encoders = []

for col in categorical_columns :
    if col not in one_hot_parameters :
        encoders.append(col)
len(encoders) + len(one_hot_parameters) == len(categorical_columns)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for col in encoders :
    features[col] = encoder.fit_transform(features[col].astype(str))
Hot_features = pd.get_dummies(features[one_hot_parameters], drop_first= True)
features = pd.concat([features.drop(one_hot_parameters, axis=1), Hot_features], axis=1)
features.columns
train_df = features.iloc[:1460,:]  
train_df['SalePrice'] = training['SalePrice']
test_df = features.iloc[1460 :,:]  
X_train = train_df.drop('SalePrice', axis=1)
y_train = train_df['SalePrice']
X_test = test_df
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_log_error, r2_score

n_folds = 5

cv = KFold(n_splits= 5, shuffle= True, random_state= 42).get_n_splits(X_train.values)

def test_1(model) :
    msle = make_scorer(mean_squared_log_error)
    rmsle = np.sqrt(cross_val_score(model, X_train, y_train, cv = cv, scoring= msle))
    score_rmsle = [rmsle.mean()]
    return score_rmsle

def test_2(model) :
    r2 = make_scorer(r2_score)
    r2_error = cross_val_score(model, X_train, y_train, cv = cv, scoring= r2)
    score_r2 = [r2_error.mean()]
    return score_r2
import xgboost as xgb
xg_boost = xgb.XGBRegressor(n_estimators= 1000)
test_1(xg_boost)
from sklearn.ensemble import BaggingRegressor

bagging_regressor = BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False,
                                     max_features=1.0, max_samples=1.0, n_estimators=1000,
                                     n_jobs=None, oob_score=False, random_state=51, verbose=0, warm_start=False)

test_1(bagging_regressor)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=500, copy_X=True, fit_intercept=True, max_iter=None, 
              normalize=False,  random_state=None, solver='auto', tol=0.001)

test_1(ridge)
from sklearn.ensemble import GradientBoostingRegressor

gradient_boosting_reg = GradientBoostingRegressor()

test_1(gradient_boosting_reg)
sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
sub_df.info()
gradient_boosting_reg.fit(X_train, y_train)
predictions = gradient_boosting_reg.predict(X_test)
sub_df['SalePrice'] = predictions
sub_df.to_csv('Gradient_Boosting_Regressor.txt', index= False)
