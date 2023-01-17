import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from scipy import stats
from scipy.stats import norm, skew #for some statistics
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_view = pd.read_csv('../input/train.csv')
train_ID = train['Id']
test_ID = test['Id']
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
print("reading files completed")
train_view.head()
train_view.info()
sns.distplot(train['SalePrice'], fit=norm)
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
sns.boxplot(data = train_view, x = "OverallQual", y = "SalePrice")
sns.boxplot(data = train_view, x = "OverallCond", y = "SalePrice")
sns.boxplot(data = train_view, x = "ExterQual", y = "SalePrice")
sns.boxplot(data = train_view, x = "ExterCond", y = "SalePrice")
sns.boxplot(data = train_view, x = "Condition1", y = "SalePrice")
sns.boxplot(data = train_view, x = "Condition2", y = "SalePrice")
grid = sns.FacetGrid(train_view, col = "OverallCond", row = "ExterCond", palette = 'seismic')
grid = grid.map(plt.scatter, "OverallQual",'SalePrice')
grid
sns.violinplot(data = train_view, x = 'MSZoning', y = "SalePrice")
sns.violinplot(data = train_view, x = 'Street', y = "SalePrice")
sns.violinplot(data = train_view, x = 'Alley', y = "SalePrice")
sns.violinplot(data = train_view, x = 'LandSlope', y = "SalePrice")
train_view[['SalePrice','Neighborhood']].groupby('Neighborhood').mean().plot.bar()
grid = sns.FacetGrid(train_view, col = "BldgType", row = "HouseStyle")
grid = grid.map(sns.kdeplot,'SalePrice')
grid
sns.violinplot(data = train_view, x = 'Foundation', y = "SalePrice")
g = sns.FacetGrid(train_view, col='MasVnrType')
g = g.map(plt.scatter, 'MasVnrArea', 'SalePrice')
sns.distplot(train_view['YearBuilt'])
train_view.plot.scatter(x='YearBuilt', y='SalePrice')
sns.distplot(train_view['YearRemodAdd'])
train_view.plot.scatter(x='YearRemodAdd', y='SalePrice')
sns.boxplot(data = train_view, x = 'SaleType', y = "SalePrice") # new house have a higher price
sns.boxplot(data = train_view, x = 'SaleCondition', y = "SalePrice") # partial has higher price
sns.boxplot(data = train_view, x='YrSold', y = "SalePrice") # no relation
sns.boxplot(data = train_view, x='MoSold', y = "SalePrice") # no relation
train_view['TotalSF'] = train_view['TotalBsmtSF'] + train_view['1stFlrSF'] + train_view['2ndFlrSF']
grid = sns.FacetGrid(train_view, col = "OverallQual", row = "MSZoning", palette = 'seismic')
grid = grid.map(plt.scatter, "TotalSF",'SalePrice')
grid
g = sns.factorplot(x="TotRmsAbvGrd",y="SalePrice",data=train_view,kind="bar", size = 6 , 
palette = "muted")
train_view.plot.scatter(x='GrLivArea', y='SalePrice')
sns.distplot(train_view['LotArea'])
train_view[train_view['LotArea']<4000]['LotArea'].plot.hist()
sns.distplot(train_view['LotFrontage'].dropna())
grid = sns.FacetGrid(train_view, col = "LotShape", row = "LotConfig", palette = 'seismic')
grid = grid.map(plt.scatter, "LotArea",'SalePrice')
grid
grid = sns.FacetGrid(train_view, col = "LotShape", row = "LotConfig", palette = 'seismic')
grid = grid.map(plt.scatter, "LotFrontage",'SalePrice')
grid
sns.boxplot(data = train_view, x = 'BsmtCond', y = "SalePrice")
sns.boxplot(data = train_view, x = 'BsmtQual', y = "SalePrice")
sns.violinplot(data = train_view, x = 'BsmtFinType1', y = "SalePrice")
train_view.plot.scatter(x='TotalBsmtSF', y='SalePrice')
train_view['ToFullBath'] = train_view['BsmtFullBath']+train_view['FullBath']
train_view['ToHalfBath'] = train_view['BsmtHalfBath']+train_view['HalfBath']
sns.boxplot(data = train_view, x = 'ToFullBath', y = "SalePrice")
sns.boxplot(data = train_view, x = 'ToHalfBath', y = "SalePrice")
grid = sns.FacetGrid(train_view, col = "GarageQual", row = "GarageCond", palette = 'seismic')
grid = grid.map(plt.scatter, "GarageArea",'SalePrice')
grid
grid = sns.FacetGrid(train_view, col = 'GarageType')
grid = grid.map(plt.scatter, "GarageYrBlt",'SalePrice')
grid
grid = sns.FacetGrid(train_view, col = 'GarageFinish')
grid = grid.map(plt.scatter, "GarageCars",'SalePrice')
grid
grid = sns.FacetGrid(train_view, col = "PoolQC")
grid = grid.map(plt.scatter, "PoolArea",'SalePrice')
grid
train_view['TotalPorch']=train_view['OpenPorchSF']+train_view['WoodDeckSF']+train_view['EnclosedPorch']+train_view['3SsnPorch']+train_view['ScreenPorch']
train_view.plot.scatter(x='TotalPorch', y='SalePrice')
g = sns.factorplot(x="Heating", y="SalePrice", data=train_view,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="CentralAir", y="SalePrice", data=train_view,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="Electrical", y="SalePrice", data=train_view,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="RoofStyle", y="SalePrice", data=train_view,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="Functional", y="SalePrice", data=train_view,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="Fireplaces", y="SalePrice", data=train_view,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="PavedDrive", y="SalePrice", data=train_view,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="KitchenQual", y="SalePrice", data=train_view,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="Fence", y="SalePrice", data=train_view,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="MiscFeature", y="SalePrice", data=train_view,
                   size=6, kind="bar", palette="muted")
print("data visualization completed")
train.head()
train['SalePrice'] = np.log(train['SalePrice']) #log
sns.distplot(train['SalePrice'], fit=norm)
from collections import Counter
def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,['GrLivArea','LotArea','SalePrice','TotalBsmtSF','GarageArea',])

train.loc[Outliers_to_drop]
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True) #delete Outliers in train
train.shape
train.plot.scatter(x='GrLivArea', y='SalePrice')
ntrain = train.shape[0]
ntest = test.shape[0]
Y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True) #combine datasets
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.shape
all_data.head()
all_data = all_data.replace(np.inf, np.nan) # important
all_data_na = (all_data.isnull().sum() / len(all_data))
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data
all_data = all_data.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'])
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data['Utilities'] = all_data['Utilities'].fillna("None")
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data
all_data.head()
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
all_data, cat_cols = one_hot_encoder(all_data, nan_as_category= True)
#years
all_data['builtYr'] = all_data['YrSold']-all_data['YearBuilt']
all_data['remodelYr']= all_data['YrSold']-all_data['YearRemodAdd']
all_data['garageYr']= all_data['YrSold']-all_data['GarageYrBlt']
#total area
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#bathrooms
all_data['TotBathrooms']=all_data['FullBath'] + (all_data['HalfBath']*0.5) + all_data['BsmtFullBath'] + (all_data['BsmtHalfBath']*0.5)
#porch
all_data['porcharea'] = all_data['OpenPorchSF']+all_data['EnclosedPorch']+all_data['3SsnPorch']+all_data['ScreenPorch']
all_data.shape
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head()
skewness = skewness[abs(skewness) > 0.25]
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
all_data.head()
all_data = all_data.replace(np.inf, np.nan) # important
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data
all_data["garageYr"] = all_data["garageYr"].fillna(0)
all_data["remodelYr"] = all_data["remodelYr"].fillna(0)
all_data["builtYr"] = all_data["builtYr"].fillna(0)
all_data.shape
print ("feature engineering completed")
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train,Y_train)
features = X_train.columns.values
importances = rf.feature_importances_
std = np.std([rf.feature_importances_ for each in rf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='bar')
feat_importances
from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(rf, prefit=True)
all_data = model.transform(all_data)
all_data.shape
print("feature importance completed")
from sklearn.model_selection import GridSearchCV
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=40))
score = rmsle_cv(lasso)
score.mean(), score.std()
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=40))
score = rmsle_cv(ENet)
score.mean(), score.std()
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)
score.mean(), score.std()
rf = RandomForestRegressor(n_estimators=500)
score = rmsle_cv(rf)
score.mean(), score.std()
GBoost = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =40)
score = rmsle_cv(GBoost)
score.mean(), score.std()
model_xgb = xgb.XGBRegressor(colsample_bytree=0.46, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             n_estimators=500,
                             reg_alpha=0.4640, reg_lambda=0.85,
                             subsample=0.5, 
                             random_state =40, nthread = -1)
score = rmsle_cv(model_xgb)
score.mean(), score.std()
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(model_lgb)
score.mean(), score.std()
print("base models completed")
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)  
averaged = AveragingModels(models = (ENet, GBoost, KRR, lasso))
score = rmsle_cv(averaged)
score.mean(), score.std()
print("model stacking completed")
averaged.fit(X_train, Y_train)
ave = np.expm1(averaged.predict(X_test))
ave
model_xgb.fit(X_train, Y_train)
xgb_pre = np.expm1(model_xgb.predict(X_test))
xgb_pre
model_lgb.fit(X_train, Y_train)
lgb_pre = np.expm1(model_lgb.predict(X_test))
lgb_pre
y_test=0.7*ave +0.15*xgb_pre+0.15*lgb_pre
len(y_test)
print("prediction completed")
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = y_test
sub.to_csv('submission.csv',index=False)
sub.head()
