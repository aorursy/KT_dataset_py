import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style="ticks", rc={'figure.figsize':(9,8)})
sns.set_context(rc = {"font.size":15, "axes.labelsize":15}, font_scale=2)
sns.set_palette('colorblind');
from pandas.api.types import CategoricalDtype

from scipy import stats

from scipy.stats import pearsonr,spearmanr, boxcox_normmax, chi2_contingency, chi2, f, shapiro, probplot
from scipy.special import boxcox1p

import statsmodels.api as sm
from statsmodels.formula.api import ols 

# pandas defaults
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

import warnings
warnings.filterwarnings('ignore')

from time import time
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print("Training data shape: ", train.shape)
print("Testing data shape: ", test.shape)
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace = True)
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace = True)
train.reset_index(drop=True, inplace=True)
train.head()
train.shape
sale_price_df = train[['SalePrice']].copy()
data = pd.concat([train.drop(columns='SalePrice'), test], axis = 0)
data.shape
round((data.isnull().sum()[data.isnull().sum()!=0]/data.shape[0])*100,2)
data.drop(columns = 'Id', inplace = True)
data.drop(columns = ['LowQualFinSF',  'BsmtHalfBath', 'ScreenPorch',  'Street', 'Alley', 'Utilities', 'LandSlope', 
                     'Condition1', 'Condition2', 'BldgType', 'RoofMatl', 'ExterCond', 'BsmtCond',  'BsmtFinType2', 
                     'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageQual', 'GarageCond', 'PavedDrive', 
                     'PoolQC', 'Fence', 'MiscFeature', 'SaleType'], inplace = True) 
data.shape
for col in ['BsmtFullBath']:
    data[col] = data[col].fillna(data[col].value_counts().idxmax())
    
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
    
for col in ['BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea', 'MasVnrArea', 'LotFrontage']:
    data[col] = data[col].fillna(data[col].median())
for col in ['MasVnrType']:
    data[col] = data[col].fillna(data[col].value_counts().idxmax())
    
for col in ['GarageType', 'GarageFinish', 'BsmtQual', 'BsmtExposure',  'BsmtFinType1', 'FireplaceQu']:
    data[col] = data[col].fillna('None')
    
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].value_counts().index[0])
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].value_counts().index[0])
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].value_counts().index[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].value_counts().index[0])
data.isnull().sum()[data.isnull().sum()!=0]
data['House_Qual'] = data['OverallQual'] + data['OverallCond']
data['Total_bathrooms'] = data['BsmtFullBath'] + data['FullBath'] + data['HalfBath']
data['Total_basement_SF'] = data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['TotalBsmtSF']
data['Total_sqr_footage'] = data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['1stFlrSF'] + data['2ndFlrSF']
data['MSSubClass'] = data['MSSubClass'].astype(str)
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)
data.drop(columns = ['MiscVal', 'PoolArea', '3SsnPorch', 'EnclosedPorch', 'BsmtFinSF2', 'KitchenQual'], inplace = True)
cat_type = CategoricalDtype(['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True)
for col in ['BsmtQual', 'FireplaceQu']:
    data[col] = data[col].astype(cat_type)
    
cat_type = CategoricalDtype(['IR3', 'IR2', 'IR1', 'Reg'], ordered=True)
data['LotShape'] = data['LotShape'].astype(cat_type)

cat_type = CategoricalDtype(['None', 'No', 'Mn', 'Av', 'Gd'], ordered=True)
data['BsmtExposure'] = data['BsmtExposure'].astype(cat_type)

cat_type = CategoricalDtype(['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], ordered=True)
for col in ['BsmtFinType1']:
    data[col] = data[col].astype(cat_type)
cat_type = CategoricalDtype(['Low', 'HLS', 'Bnk', 'Lvl'], ordered=True)
data['LandContour'] = data['LandContour'].astype(cat_type)

cat_type = CategoricalDtype(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True)
for col in ['ExterQual']:
    data[col] = data[col].astype(cat_type)
    
cat_type = CategoricalDtype(['None', 'Detchd', 'CarPort', 'BuiltIn', 'Basment', 'Attchd', '2Types'],ordered=True)
data['GarageType'] = data['GarageType'].astype(cat_type)

cat_type = CategoricalDtype(['None', 'Unf', 'RFn', 'Fin'],ordered=True)
data['GarageFinish'] = data['GarageFinish'].astype(cat_type)

data['Has_Garage'] = np.where(data['GarageArea']>0,1,0)
data['Has_basement'] = np.where(data['Total_basement_SF'] > 0, 1, 0)
data['Has_fireplace'] = np.where(data['Fireplaces']>0,1,0)
for col in data.filter(regex = '^Has', axis = 'columns').columns:
    print(data[col].value_counts(normalize = True))
data.drop(columns = ['Has_Garage', 'Has_basement'], inplace = True)
data.shape
sns.distplot(sale_price_df['SalePrice']);
plt.title("Skew: {} | Kurtosis: {}".format(sale_price_df['SalePrice'].skew(), sale_price_df['SalePrice'].kurt()));
numeric_columns = data.select_dtypes('number').columns
numeric_columns = numeric_columns[~numeric_columns.str.contains('^Has')]
numeric_columns
X_train = data.iloc[:len(train),:].copy()
X_test = data.iloc[len(train):,:].copy()
X_train = pd.concat([X_train,sale_price_df['SalePrice']], axis = 1)
def generate_heatmap(df):
    # Generate a heatmap with the upper triangular matrix masked
    # Compute the correlation matrix
    corr = df.corr(method="spearman")
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    plt.figure(figsize = (15,9));
    # Draw the heatmap with the mask 
    sns.heatmap(corr, mask=mask, cmap='coolwarm', fmt = '.2f', linewidths=.5, annot = True);
    plt.title("Correlation heatmap");
    return
import math
correlation_results_list = []
ncols = 3
nrows = math.ceil(len(numeric_columns)/ncols)
fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*4.5,nrows*3))
axes_list = [item for sublist in axes for item in sublist] 
for col in numeric_columns:
    ax = axes_list.pop(0) # Take the first axes of the axes_list
    sns.regplot(X_train[col], X_train['SalePrice'], ax = ax)
    stp = spearmanr(X_train[col], X_train['SalePrice'])
    str_title = "r = " + "{0:.2f}".format(stp[0])
    ax.set_title(str_title,fontsize=11)
    correlation_results_list.append((col, abs(stp[0])))
    
plt.tight_layout(); 
plt.show();
correlation_df = pd.DataFrame(correlation_results_list, columns = ['column_name', 'correlation'])
correlation_df.sort_values(by = 'correlation', ascending = False, inplace = True)
correlation_df.head()
columns_with_high_corr = correlation_df.loc[correlation_df['correlation']>=0.6, 'column_name'].to_list()
columns_with_high_corr
columns_with_low_corr = correlation_df[correlation_df['correlation']<0.6]['column_name'].to_list()
data.drop(columns = columns_with_low_corr, inplace = True)
X_train.drop(columns = columns_with_low_corr, inplace = True)
generate_heatmap(X_train[columns_with_high_corr+['SalePrice']])
data.drop(columns = ['House_Qual', 'GarageArea', 'GarageYrBlt', 'Total_sqr_footage', 'YearBuilt'], inplace = True)
X_train.drop(columns = ['House_Qual', 'GarageArea', 'GarageYrBlt', 'Total_sqr_footage', 'YearBuilt'], inplace = True)
category_cols = X_train.select_dtypes(['object', 'category', 'int32']).columns
category_cols
X_train['SalePrice'] = np.log1p(X_train['SalePrice'])
categ_columns_with_high_association = []
categ_columns_with_low_association = []
def perform_anova_and_its_results(categ_col, num_col='SalePrice', df = X_train):
    df_sst = len(df[num_col])-1
    df_ssb = df[categ_col].nunique() - 1
    df_ssw = df_sst - df_ssb
    F_critical = f.ppf(0.95, df_ssb, df_ssw)
#     print("F_Critical: {0:.3f}".format(F_critical))
    results = ols('{} ~{}'.format(num_col, categ_col), data = df).fit()
    aov_table = sm.stats.anova_lm(results, typ = 1)  
    F_stat = aov_table.loc[categ_col, 'F']
#     print("F_statistic: {0:.3f}".format(F_stat))
    if (F_stat > F_critical):
#         print("F-statistic is more than F-critical")
#         print("There is an association between {} and {}".format(categ_col,num_col))
        categ_columns_with_high_association.append(categ_col)
    else:
#         print("F-statistic is less than F-critical")
#         print("There is no association between {} and {}".format(categ_col,num_col))
        categ_columns_with_low_association.append(categ_col)
#     print('-'*30)
for col in category_cols:
    perform_anova_and_its_results(col)
categ_columns_with_low_association
data.drop(columns = categ_columns_with_low_association, inplace = True)
X_train.drop(columns = categ_columns_with_low_association, inplace = True)
def phi_coefficient(a,b):
    temp = pd.crosstab(a,b)
    nr = (temp.iloc[1,1] * temp.iloc[0,0]) - (temp.iloc[0,1]*temp.iloc[1,0])
    dr = np.sqrt(np.product(temp.apply(sum, axis = 'index')) * np.prod(temp.apply(sum, axis = 'columns')))
    return(nr/dr)
cat_binary_cols = []
for col in X_train.select_dtypes(['int32', 'object']).columns:
    if (X_train[col].nunique()==2):
        cat_binary_cols.append(col)
        
cat_binary_cols
def cramers_v(a,b):
    crosstab = pd.crosstab(a,b)
    chi2 = chi2_contingency(crosstab)[0]  # chi-squared value
    n = crosstab.sum().sum()
    phi2 = chi2/n
    r, k = crosstab.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return(np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))))


from collections import Counter
def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    :param x: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :param y: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :return: float
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy


def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = stats.entropy(p_x)
    if s_x == 0:
        return(1)
    else:
        return((s_x - s_xy)/s_x)
category_cols = X_train.select_dtypes(['object', 'category', 'int32']).columns.to_list()
category_cols
temp = pd.DataFrame(columns=category_cols, index=category_cols).fillna(0)
cat_col_correlation = []
for row in category_cols:
    a = row
    for col in category_cols:
        b = col
        temp.loc[a,b] = theils_u(X_train[a],X_train[b])
        temp.loc[b,a] = temp.loc[a,b]
        cat_col_correlation.append((a,b,temp.loc[a,b]))
cat_cols_corr_df = pd.DataFrame(cat_col_correlation, columns = ['col1', 'col2', 'correlation'])
cat_cols_corr_df.head()
cat_cols_corr_df.shape
high_corr_cat=cat_cols_corr_df[(cat_cols_corr_df['correlation']>=0.6) & (cat_cols_corr_df['col1']!=cat_cols_corr_df['col2'])]
high_corr_cat.head()
high_corr_cat
data.drop(columns = ['Has_fireplace', 'Exterior1st', 'MSSubClass', 'MSZoning'], inplace = True)
X_train.drop(columns = ['Has_fireplace', 'Exterior1st', 'MSSubClass', 'MSZoning'], inplace = True)
t = X_train.select_dtypes(['int64', 'float64']).columns.tolist()
t.remove('SalePrice')
t
X_train.select_dtypes(['int64', 'float64']).columns
nrows = 2
ncols = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.5,nrows*3))
axes_list = [item for sublist in axes for item in sublist]
for col in t:
    ax = axes_list.pop(0) # Take the first axes of the axes_list
    probplot(X_train[col], dist = 'norm', plot = ax)
    str_title = "QQ-plot of {}".format(col)
    ax.set_title(str_title,fontsize=11)

    
plt.tight_layout(); 
plt.show();

# Now use the matplotlib .remove() method to 
# delete anything we didn't use
for ax in axes_list:
    ax.remove()
for col in t:
    X_train[col] = boxcox1p(X_train[col], boxcox_normmax(X_train[col]+1))
    data[col] = boxcox1p(data[col], boxcox_normmax(data[col]+1))
nrows = 2
ncols = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.5,nrows*3))
axes_list = [item for sublist in axes for item in sublist]
for col in t:
    ax = axes_list.pop(0) # Take the first axes of the axes_list
    probplot(X_train[col], dist = 'norm', plot = ax)
    str_title = "QQ-plot of {}".format(col)
    ax.set_title(str_title,fontsize=11)

    
plt.tight_layout(); 
plt.show();

# Now use the matplotlib .remove() method to 
# delete anything we didn't use
for ax in axes_list:
    ax.remove()
nominal_cols = X_train.select_dtypes('object').columns.to_list()
nominal_cols
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder,OrdinalEncoder, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
train = data.iloc[:len(X_train),:].copy()
train['SalePrice'] = X_train['SalePrice'].copy()
test = data.iloc[len(X_train):,:].copy()
print("Train shape: ", train.shape)
print("Test shape: ", test.shape)
X = train.loc[:,train.columns!='SalePrice']
y = train['SalePrice'].copy()
X.shape
ohe_dict_cols = {}
for col in X.select_dtypes(include=['object']).dtypes.index:
    ohe_dict_cols[col] = pd.Series(X[col].unique()).to_list()
    
# For one-hot encoder
t_k = []  # nominal column names
t_v = []  # values of nominal columns
for k,v in ohe_dict_cols.items():
    t_k.append(k)
    t_v.append(v)
ord_encod_dict = {}
for col in X.select_dtypes(include='category').columns:
    ord_encod_dict[col] = pd.Series(data[col].unique().sort_values()).to_list()

# For ordinal encoder
ordinal_cols = []
ordinal_vals = []
for k,v in ord_encod_dict.items():
    ordinal_cols.append(k)
    ordinal_vals.append(v)
number_cols = X.select_dtypes('number').columns.tolist()
number_cols
# Setup cross validation folds
kf = KFold(n_splits = 10, random_state=1, shuffle=True)
import xgboost as xgb

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', random_state=1, n_jobs = -1, max_depth = 3, min_child_weight = 5,
                       alpha = 1e-5, gamma = 0, subsample=0.8, colsample_bytree=0.7, learning_rate =0.1, n_estimators = 400)

colT = ColumnTransformer([
    ('dummy_col', OneHotEncoder(drop = 'first', categories = t_v), t_k),
    ('ordinal_cols', OrdinalEncoder(categories = ordinal_vals), ordinal_cols)
], remainder = 'passthrough')


xg_pipeline = Pipeline(steps = [('colt', colT), ("xg", xg_reg)])
xg_pipeline.fit(X, y)

# prediction_XG_1 = xg_pipeline.predict(test)

scores = -cross_val_score(xg_pipeline, X, y, cv = kf, n_jobs = -1,scoring = 'neg_root_mean_squared_error')
print('RMSE scores: ', scores)
print('RMSE scores mean: {:.3f}'.format(scores.mean()))
print('RMSE scores std deviation: {:.3f}'.format(scores.std()))
from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor(random_state=1, n_jobs = -1, max_depth =  8, max_samples = None, min_samples_leaf =  2, 
                              n_estimators = 120)

colT = ColumnTransformer([
    ('dummy_col', OneHotEncoder(drop = 'first', categories = t_v), t_k),
    ('ordinal_cols', OrdinalEncoder(categories = ordinal_vals), ordinal_cols)
], remainder = 'passthrough')

rf_pipeline = Pipeline(steps = [('colt', colT), ("rf", rfreg)])
rf_pipeline.fit(X, y)

scores = -cross_val_score(rf_pipeline, X, y, cv = kf, n_jobs = -1,scoring = 'neg_root_mean_squared_error')
print('RMSE scores: ', scores)
print('RMSE scores mean: {:.3f}'.format(scores.mean()))
print('RMSE scores std deviation: {:.3f}'.format(scores.std()))
from sklearn.linear_model import Ridge
ridge_reg = Ridge(random_state = 1, alpha = 25)

rb_scaler = RobustScaler(with_centering=False)

colT = ColumnTransformer([
    ('dummy_col', OneHotEncoder(drop = 'first', categories = t_v), t_k),
    ('ordinal_cols', OrdinalEncoder(categories = ordinal_vals), ordinal_cols)
], remainder = 'passthrough')

# skb = SelectKBest(f_regression)

ridge_pipeline = Pipeline(steps = [('colt', colT), ('rb', rb_scaler), ("ridge", ridge_reg)])
ridge_pipeline.fit(X, y)

scores = -cross_val_score(ridge_pipeline, X, y, cv = kf, n_jobs = -1,scoring = 'neg_root_mean_squared_error')
print('RMSE scores: ', scores)
print('RMSE scores mean: {:.3f}'.format(scores.mean()))
print('RMSE scores std deviation: {:.3f}'.format(scores.std()))
from sklearn.linear_model import Lasso
lasso_reg = Lasso(random_state = 1, alpha = 0.0005)

rb_scaler = RobustScaler(with_centering=False)

colT = ColumnTransformer([
    ('dummy_col', OneHotEncoder(drop = 'first', categories = t_v), t_k),
    ('ordinal_cols', OrdinalEncoder(categories = ordinal_vals), ordinal_cols)
], remainder = 'passthrough')

# Bigger the alpha values the lesser the features that will be selected

lasso_pipeline = Pipeline(steps = [('colt', colT),  ('rb', rb_scaler), ("lasso", lasso_reg)])
lasso_pipeline.fit(X, y)

scores = -cross_val_score(lasso_pipeline, X, y, cv = kf, n_jobs = -1,scoring = 'neg_root_mean_squared_error')
print('RMSE scores: ', scores)
print('RMSE scores mean: {:.3f}'.format(scores.mean()))
print('RMSE scores std deviation: {:.3f}'.format(scores.std()))
from sklearn.linear_model import ElasticNet
elastic_reg = ElasticNet(random_state = 1, l1_ratio = 0.095, alpha = 0.0005)

rb_scaler = RobustScaler(with_centering=False)

colT = ColumnTransformer([
    ('dummy_col', OneHotEncoder(drop = 'first', categories = t_v), t_k),
    ('ordinal_cols', OrdinalEncoder(categories = ordinal_vals), ordinal_cols)
], remainder = 'passthrough')

# Bigger the alpha values the lesser the features that will be selected

elastic_pipeline = Pipeline(steps = [('colt', colT),  ('rb', rb_scaler), ("elastic", elastic_reg)])
elastic_pipeline.fit(X, y)

scores = -cross_val_score(elastic_pipeline, X, y, cv = kf, n_jobs = -1,scoring = 'neg_root_mean_squared_error')
print('RMSE scores: ', scores)
print('RMSE scores mean: {:.3f}'.format(scores.mean()))
print('RMSE scores std deviation: {:.3f}'.format(scores.std()))
from catboost import CatBoostRegressor
cat_obj_cols = X.select_dtypes(['category', 'object']).columns.to_list()
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1)

cat = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, eval_metric = 'RMSE', loss_function='RMSE',
                       l2_leaf_reg = 1)
cat.fit(X_train, y_train, cat_features = cat_obj_cols, eval_set= (X_validation, y_validation), plot=True);
import lightgbm as lgb
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(boosting_type = 'gbdt' ,random_state=1, n_jobs=-1, learning_rate = 0.1, max_depth = 3, 
                     min_child_weight = 0.001, min_split_gain = 0, n_estimators = 250, reg_alpha = 0.01)


colT = ColumnTransformer([
    ('dummy_col',OneHotEncoder(drop = 'first',categories = t_v), t_k),
    ('ordinal_cols', OrdinalEncoder(categories = ordinal_vals), ordinal_cols)
], remainder = 'passthrough')


lgbm_pipeline = Pipeline(steps = [('colt', colT), ("lgbm", lgbm)])
lgbm_pipeline.fit(X, y)

scores = -cross_val_score(lgbm_pipeline, X, y, cv = kf, n_jobs = -1,scoring = 'neg_root_mean_squared_error')
print('RMSE scores: ', scores)
print('RMSE scores mean: {:.3f}'.format(scores.mean()))
print('RMSE scores std deviation: {:.3f}'.format(scores.std()))
from mlxtend.regressor import StackingCVRegressor

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(lasso_pipeline['lasso'], ridge_pipeline['ridge'], xg_pipeline['xg'],
                                            lgbm_pipeline['lgbm'], elastic_pipeline['elastic']),
                                meta_regressor = xg_pipeline['xg'], use_features_in_secondary=True)


colT = ColumnTransformer([
    ('dummy_col',OneHotEncoder(drop = 'first',categories = t_v), t_k),
    ('ordinal_cols', OrdinalEncoder(categories = ordinal_vals), ordinal_cols)
], remainder = 'passthrough')


stack_gen_pipeline = Pipeline(steps = [('colt', colT), ("stack_gen", stack_gen)])
stack_gen_pipeline.fit(X, y)

scores = -cross_val_score(stack_gen_pipeline, X, y, cv = kf, n_jobs = -1,scoring = 'neg_root_mean_squared_error')
print('RMSE scores: ', scores)
print('RMSE scores mean: {:.3f}'.format(scores.mean()))
print('RMSE scores std deviation: {:.3f}'.format(scores.std()))
# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.05 * lasso_pipeline.predict(X)) + \
            (0.05 * ridge_pipeline.predict(X)) + \
            (0.5 * xg_pipeline.predict(X)) + \
            (0.1 * lgbm_pipeline.predict(X)) + (0.3 * stack_gen_pipeline.predict(X)))
# Define error metrics
from sklearn.metrics import mean_squared_error
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
# Get final precitions from the blended model
blended_score = rmsle(y, blended_predictions(X))
print('RMSE blended : {:.3f}'.format(blended_score))
predictions_blended = blended_predictions(test)
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sub_pred = np.expm1(predictions_blended)
df_output = pd.DataFrame()
df_output['Id'] = test['Id']
df_output['SalePrice'] = sub_pred
df_output.head()
df_output.to_csv('submission.csv', index = False)