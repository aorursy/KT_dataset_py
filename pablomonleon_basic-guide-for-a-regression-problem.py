import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sns.distplot(df_train.SalePrice)
sns.boxplot(df_train['SalePrice'])
print('Skewness: ',df_train['SalePrice'].skew())
print('Kurtosis: ',df_train['SalePrice'].kurt())
df_train['SalePrice']=np.log1p(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'])
fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
df_train.drop(df_train[df_train['GrLivArea']>4000].index, inplace=True)
ntrain = df_train.shape[0] #we do this because we are going to concatenate train and test and we will need this later
ntest = df_test.shape[0] 

y_train = df_train.SalePrice.values

df_all = pd.concat((df_train, df_test)).reset_index(drop=True)
df_all.drop(['SalePrice'], axis=1, inplace=True)

df=df_all
def overview(df):
    print('SHAPE: ', df.shape)
    print('columns: ', df.columns.tolist())
    col_nan=df.columns[df.isnull().any()].tolist()
    print('columns with missing data: ',df[col_nan].isnull().sum())

overview(df)
all_data_na = (df.isnull().sum() / len(df)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data
corrmat = df_train.corr()
corrmat['SalePrice'].sort_values(ascending=False)
import matplotlib.pyplot as plt#visualization
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
plt.figure(figsize=(10,10))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
cat_cols = df.select_dtypes('object').columns.tolist()

for i in cat_cols:
    print(df[i].value_counts())
int_cols =  df.select_dtypes(['int64','float64']).columns.tolist()
df[int_cols].describe().T
def handling_missing(df):
    cols_none=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageFinish','GarageQual','GarageCond',
               'GarageType','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','Exterior2nd',
               'Exterior1st']
    for i in cols_none:
        df[i] = df[i].fillna('None')
    
    cols_zero=['GarageYrBlt','BsmtHalfBath','BsmtFullBath','MasVnrArea','TotalBsmtSF','BsmtFinSF2','BsmtFinSF1',
               'BsmtUnfSF']
    for i in cols_zero:
        df[i] = df[i].fillna(0)
    
    cols_mode=['MasVnrType','MSZoning','Utilities','SaleType','GarageArea','GarageCars','KitchenQual','Electrical']
    for i in cols_mode:
        df[i] = df[i].fillna(df[i].mode()[0])
    
    df["Functional"] = df["Functional"].fillna("Typ") #tells you to do this in the data description
    
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean()) 
    
    df=df.drop(['Id'],axis=1) # Let's drop the Id column while we are at it
    
    return df
df=handling_missing(df)
qual_dict = {'None': 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

df["ExterQual"] = df["ExterQual"].map(qual_dict).astype(int)
df["ExterCond"] = df["ExterCond"].map(qual_dict).astype(int)
df["BsmtQual"] = df["BsmtQual"].map(qual_dict).astype(int)
df["BsmtCond"] = df["BsmtCond"].map(qual_dict).astype(int)
df["HeatingQC"] = df["HeatingQC"].map(qual_dict).astype(int)
df["KitchenQual"] = df["KitchenQual"].map(qual_dict).astype(int)
df["FireplaceQu"] = df["FireplaceQu"].map(qual_dict).astype(int)
df["GarageQual"] = df["GarageQual"].map(qual_dict).astype(int)
df["GarageCond"] = df["GarageCond"].map(qual_dict).astype(int)
# Divide up the years between 1871 and 2010 in slices of 20 years.
year_map = pd.concat(pd.Series("YearBin" + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))

df['YearBuilt']=df['YearBuilt'].map(year_map)
df['YearRemodAdd']=df['YearRemodAdd'].map(year_map)
df['GarageYrBlt']=df['GarageYrBlt'].map(year_map)
df['YrSold']=df['YrSold'].map(year_map)
cols_numcat=['MSSubClass','MoSold']

for i in cols_numcat:
    df[i]=df[i].astype('object')
from scipy.stats import skew

numeric_features = df.dtypes[df.dtypes != "object"].index

numeric_df=df[numeric_features]
  
skewed = df[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index

df[skewed]=np.log1p(df[skewed])
from sklearn.preprocessing import StandardScaler

std=StandardScaler()
scaled=std.fit_transform(df[numeric_features])
scaled=pd.DataFrame(scaled,columns=numeric_features)

df_original=df.copy()
df=df.drop(numeric_features,axis=1)

df=df.merge(scaled,left_index=True,right_index=True,how='left')
df=pd.get_dummies(df)
train = df[:ntrain]
test = df[ntrain:]
#this is the metric we use to validate the model
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb

alphas_ridge = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas_lasso = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# Kernel Ridge Regression : made robust to outliers
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_ridge, cv=kfolds))

# LASSO Regression : made robust to outliers
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                    alphas=alphas_lasso,random_state=42, cv=kfolds))

# XGBOOST Regressor : The parameters of this model I took from another notebook
regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)


lasso_model = lasso.fit(train,y_train)
ridge_model = ridge.fit(train,y_train)
xgb_model = regr.fit(train,y_train)

# model blending function using fitted models to make predictions
def blend_models(X):
    return ((xgb_model.predict(X)) + (lasso_model.predict(X)) + (ridge_model.predict(X)))/3

y_pred=blend_models(train)

print("blend score on training set: ", rmse(y_train, y_pred))
y_pred_blend = blend_models(test)
y_pred_exp_blend = np.exp(y_pred_blend)

pred_df_blend = pd.DataFrame(y_pred_exp_blend, index=df_test["Id"], columns=["SalePrice"])
pred_df_blend.to_csv('output.csv', header=True, index_label='Id')