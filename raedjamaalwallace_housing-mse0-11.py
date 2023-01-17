import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import math
from sklearn import feature_selection 
from sklearn.model_selection import GridSearchCV,train_test_split,KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, FunctionTransformer
from sklearn.ensemble import ExtraTreesClassifier
from category_encoders import OneHotEncoder 
from scipy.stats import skew
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder 
house_ts = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col=0)
house_tr = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col=0)

# check shape of the train and test sets
print(house_ts.shape)
print(house_tr.shape)

# Separate the price from train set
y_tr = house_tr['SalePrice']
print(y_tr.shape)
cols = house_tr.columns.intersection(house_ts.columns)

# Train shape and Length
house_tr = house_tr[cols]
print(house_tr.shape)
train_length = len(house_tr)

house_ts = house_ts[cols]
print(house_ts.shape)
test_length = len(house_ts)
# Concat the two dataframes
train_test = pd.concat([house_tr,house_ts],axis=0,sort=False)
print(train_test.shape)
# Heatmap highlighting Null counts from train set
plt.figure(figsize = (28,20))
sns.heatmap(train_test.isnull(), yticklabels=False,cmap="PiYG", cbar=False)
# Check all features with more than half columns nans from test set
for column in train_test.columns:
  if train_test[column].isnull().sum()> len(train_test)/2:
    print(column)

# Delete these columns
train_test = train_test.dropna(thresh=len(train_test)/2,axis=1)
print('New Shape ',train_test.shape)
# object datatypes for the joint datasets
obj_tr = train_test.select_dtypes(include=['object']).copy()
print(obj_tr.shape)

# numeric datatypes from the joint datasets
num_tr = train_test.select_dtypes(include=['int64','float64']).copy()
print(num_tr.shape)
# Train Set
num_trn = num_tr.loc[0:train_length,:]
obj_trn = obj_tr.loc[0:train_length,:]

# Test Set
num_tst = num_tr.loc[train_length+1:,:]
obj_tst = obj_tr.loc[train_length+1:,:]
# Replace numeric nans with zero in the train set
tr_num_nan=[feature for feature in num_trn.columns if num_trn[feature].isnull().sum()>=1 and num_trn[feature].dtypes!='O']
for feature in tr_num_nan:
    # num_tr[feature].fillna(0,inplace=True)
  
    ## We will replace by using median since there are outliers
    median_value=num_trn[feature].median()
    num_trn[feature].fillna(median_value,inplace=True)

# Count null in train set 
num_trn[tr_num_nan].isnull().sum()

# Replace numeric nans with zero in the test set
tr_num_nan=[feature for feature in num_tst.columns if num_tst[feature].isnull().sum()>=1 and num_tst[feature].dtypes!='O']
for feature in tr_num_nan:
  
    ## We will replace by using median since there are outliers
    median_value=num_tst[feature].median()
    num_tst[feature].fillna(median_value,inplace=True)

# Count null in test set 
num_tst[tr_num_nan].isnull().sum()
num_tr_ts = pd.concat([num_trn,num_tst],axis=0,sort=False)
print(num_tr_ts.shape)
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import boxcox
from scipy.special import inv_boxcox

numeric_features = list(num_tr_ts.columns)

#Absolute Skewness of features
from scipy.stats import skew
skewed_features = num_tr_ts[numeric_features].apply(lambda x: abs(skew(x))).sort_values(ascending=False)
print(skewed_features)
  
#test
high_skewness = skewed_features[skewed_features >= 1]
skewed_features = high_skewness.index
# skewed features Train set
num_tr_tsk = num_tr_ts[list(skewed_features)]
num_tr_tsk = num_tr_tsk.iloc[0:train_length,:]

# plt.style.use('dark_background')
fig, axes = plt.subplots(10, 2,figsize=(20,80))
fig.subplots_adjust(hspace=0.6)
colors=[plt.cm.prism_r(each) for each in np.linspace(0, 1, len(num_tr_tsk.columns))]
for i,ax,color in zip(num_tr_tsk,axes.flatten(),colors):
    sns.regplot(x=num_tr_tsk[i], y= y_tr, fit_reg=True,marker='o',scatter_kws={'s':50,'alpha':0.8},color=color,ax=ax)
    plt.xlabel(i,fontsize=12)
    plt.ylabel('SalePrice',fontsize=12)
    ax.set_title('SalePrice'+' - '+str(i),color=color,fontweight='bold',size=20)
# Create the dataframe of top 5 most skeweed features for Vizualization
# top_5 = list(skewed_features)[0:5]
# top_5sk = num_tr_ts.loc[:,top_5]

# fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(22,5))

# # flatten axes for easy iterating
# for i, ax in enumerate(axes.flatten()):
#   sns.distplot(top_5sk.iloc[:, i], ax=ax)

# fig.tight_layout()

# # plot the top 4 skewed features before
# fig, ax=plt.subplots(1,2,figsize=(20,5))
# sns.distplot(num_tr_ts['MiscVal'], ax=ax[0])
# sns.distplot(num_tr_ts['PoolArea'], ax=ax[1])
# sns.distplot(num_tr_ts['LotArea'], ax=ax[2])
# sns.distplot(num_tr_ts['LowQualFinSF'], ax=ax[3])
# Apply a Power Transformer since power transformers are good at reducing Heteroskedacity
num_unskwed = PowerTransformer(method='yeo-johnson').fit_transform(num_tr_ts[list(skewed_features)])
num_unskw = pd.DataFrame(num_unskwed,columns=list(skewed_features))
numeric_after = list(num_unskw.columns)

skewed_after = num_unskw[numeric_after].apply(lambda x: skew(x)).sort_values(ascending=False)
print(skewed_after)
from scipy import stats
fig, ax=plt.subplots(1,2,figsize=(20,5))
ax[0].set_title('Histogram of SalePrice')
sns.distplot(y_tr, ax=ax[0])

ax[1].set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
stats.probplot(y_tr, plot=ax[1])
fig, ax=plt.subplots(1,2,figsize=(20,5))

ax[0].set_title('Log Transformed SalePrice')
sns.distplot(np.log1p(y_tr), ax=ax[0])

stats.probplot(np.log1p(y_tr), plot=ax[1])
# Unskwed features Train set
num_unskw_Tr = num_unskw.iloc[0:train_length,:]

# plt.style.use('dark_background')
fig, axes = plt.subplots(10, 2,figsize=(20,80))
fig.subplots_adjust(hspace=0.6)
colors=[plt.cm.prism_r(each) for each in np.linspace(0, 1, len(num_unskw_Tr.columns))]
for i,ax,color in zip(num_unskw_Tr,axes.flatten(),colors):
    sns.regplot(x=num_unskw_Tr[i], y= np.log1p(y_tr), fit_reg=True,marker='o',scatter_kws={'s':50,'alpha':0.8},color=color,ax=ax)
    plt.xlabel(i,fontsize=12)
    plt.ylabel('SalePrice',fontsize=12)
    # ax.set_yticks(np.arange(-100000,900001,100000))
    ax.set_title('SalePrice'+' - '+str(i),color=color,fontweight='bold',size=20)
from sklearn.preprocessing import MinMaxScaler

skew_over_1 = num_unskw[['PoolArea','KitchenAbvGr','3SsnPorch','BsmtHalfBath','BsmtFinSF2','LowQualFinSF','EnclosedPorch']]
num_unskwed_MinMax = MinMaxScaler().fit_transform(skew_over_1 )
num_unskwed_MinMax = pd.DataFrame(num_unskwed_MinMax,columns=skew_over_1.columns)
num_unskwed_MinMax
print(num_unskwed_MinMax['PoolArea'].value_counts())
print(num_unskwed_MinMax['KitchenAbvGr'].value_counts())
print(num_unskwed_MinMax['3SsnPorch'].value_counts())
print(num_unskwed_MinMax['BsmtHalfBath'].value_counts())
print(num_unskwed_MinMax['BsmtFinSF2'].value_counts())
print(num_unskwed_MinMax['LowQualFinSF'].value_counts())
print(num_unskwed_MinMax['EnclosedPorch'].value_counts())
# Custom Transformer
from sklearn.base import BaseEstimator, TransformerMixin

class ExperimentalTransformer_2(BaseEstimator, TransformerMixin):
  # add another additional parameter, just for fun, while we are at it
  def __init__(self, feature_name):  
    # print('\n>>>>>>>init() called.\n')
    self.feature_name = feature_name

  def fit(self, X, y = None):
    # print('\n>>>>>>>fit() called.\n')
    return self

  def transform(self, X, y = None):
    # print('\n>>>>>>>transform() called.\n')
    X_ = X.copy() # creating a copy to avoid changes to original dataset
    # Function to Binarize the data
    X_[self.feature_name] = np.where(X_[self.feature_name] > 0.5, 1,0)
    return X_

Transformed_df = ExperimentalTransformer_2(['PoolArea','KitchenAbvGr','3SsnPorch','BsmtHalfBath','BsmtFinSF2','LowQualFinSF','EnclosedPorch']).fit_transform(num_unskwed_MinMax)
Transformed_df
# x_y['SalePrice1'] = np.log1p(x_y.SalePrice)
# x_y['SalePrice2'] = np.expm1(x_y.SalePrice1)
# xy_stdScaler = StandardScaler().fit_transform(x_y[['SalePrice2']])
# xy_Scaler = StandardScaler().fit_transform(x_y[['SalePrice1']])

# fig, ax=plt.subplots(1,4,figsize=(20,5))
# sns.distplot(x_y['SalePrice1'], ax=ax[0])
# sns.distplot(x_y['SalePrice2'], ax=ax[1])
# sns.distplot(xy_stdScaler, ax=ax[2])
# sns.distplot(xy_Scaler, ax=ax[3])
train_test_obj = pd.concat([obj_trn,obj_tst],axis=0,sort=False)
print(train_test_obj.shape)
train_test_obj['MasVnrType']=train_test_obj['MasVnrType'].fillna('MissingMasVnrType')
train_test_obj['BsmtQual']=train_test_obj['BsmtQual'].fillna('MissingBsmtQual')
train_test_obj['BsmtCond']=train_test_obj['BsmtCond'].fillna('MissingBsmtCond')
train_test_obj['BsmtExposure']=train_test_obj['BsmtExposure'].fillna('MissingBsmtExposure')
train_test_obj['BsmtFinType1']=train_test_obj['BsmtFinType1'].fillna('MissingBsmtFinType1')
train_test_obj['BsmtFinType2']=train_test_obj['BsmtFinType2'].fillna('MissingBsmtFinType2')
train_test_obj['FireplaceQu']=train_test_obj['FireplaceQu'].fillna('MissingFireplaceQu')
train_test_obj['GarageType']=train_test_obj['GarageType'].fillna('MissingGarageType')
train_test_obj['GarageFinish']=train_test_obj['GarageFinish'].fillna('MissingGarageFinish')
train_test_obj['GarageQual']=train_test_obj['GarageQual'].fillna('MissingGarageQual')
train_test_obj['GarageCond']=train_test_obj['GarageCond'].fillna('MissingGarageCond')
# Ordinal Features to numeric
cleanup_nums = {"Utilities":{"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
                "ExterQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 3},
                "ExterCond": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 3},
                "BsmtQual": {'MissingBsmtQual': 0,"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 4},
                "BsmtCond": {'MissingBsmtCond': 0,"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 4},
                "BsmtExposure":{'MissingBsmtExposure': 0,"Po": 1, "Fa": 2, "TA": 2, "Gd": 4},
                "BsmtFinType1":{'MissingBsmtFinType1': 0,"Unf": 1, "LwQ": 2, "Rec": 2, "BLQ": 3, "ALQ": 3,"GLQ":4},
                "BsmtFinType2":{'MissingBsmtFinType2': 0,"Unf": 1, "LwQ": 2, "Rec": 2, "BLQ": 3, "ALQ": 3,"GLQ":6},
                "HeatingQC":{"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 3},
                "CentralAir":{"N": 0, "Y": 1},
                "KitchenQual":{"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 3},
                "Functional":{'Sal':0,'Sev': 0,"Maj2": 1, "Maj1": 1, "Mod": 2, "Min2": 3, "Min1": 3,"Typ":4},
                "FireplaceQu":{'MissingFireplaceQu': 0,"Po": 1, "Fa": 2, "TA": 2, "Gd": 3, "Ex": 3},
                "GarageQual":{'MissingGarageQual': 0,"Po": 1, "Fa": 2, "TA": 2, "Gd": 3, "Ex": 3},
                "GarageCond":{'MissingGarageCond': 0,"Po": 1, "Fa": 2, "TA": 2, "Gd": 3, "Ex": 3}}
#Observe the values b4 as all objects
train_test_obj.dtypes.value_counts()
# Apply the ordinal Encode
train_test_obj.replace(cleanup_nums,inplace=True)
# See the Ordinal Values after are now numeric
train_test_obj.dtypes.value_counts()
train_test_obj.shape
# Ordinal features only
obj_tr_tst_ordinal = train_test_obj.select_dtypes(include = ['int64','float64'])
# one hot encoding for nominal variable
obj_1h_tr_ts = train_test_obj.select_dtypes(include=['object']).copy()

myEncoder = OneHotEncoder(handle_unknown='ignore')

codestr = myEncoder.fit_transform(obj_1h_tr_ts)
obj_1h_tr_ts = pd.DataFrame(codestr)
obj_1h_tr_ts.shape
# View each shape
print(obj_1h_tr_ts.shape)
print(num_tr_ts.shape)
print(obj_tr_tst_ordinal.shape)
# Merge them
obj_tr_try = pd.concat([obj_1h_tr_ts,obj_tr_tst_ordinal,num_tr_ts],axis =1)
obj_tr_try.shape
Train_Set = obj_tr_try.iloc[0:train_length,:]
print(Train_Set.shape)
Test_Set = obj_tr_try.iloc[train_length:,:]
print(Test_Set.shape)
# Concat the Y
obj_tr_nw = pd.concat([Train_Set,y_tr],axis =1)
obj_tr_nw.shape

corr_new_train=obj_tr_nw.corr().abs()
plt.figure(figsize=(8,25))
# Top 40 correlations to price in descending order
top_40 = corr_new_train[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(50)
sns.heatmap(top_40,annot_kws={"size": 16},vmin=-1, cmap='PiYG', annot=True)
sns.set(font_scale=2)
# find the absolute covariances 
corrmatrix =obj_tr_nw.corr().abs()

# Create a mask for cov matrix and plot the masp
mask = np.triu(np.ones_like(corrmatrix,dtype=bool))
x_y_msk = corrmatrix.mask(mask)
# f, ax = plt.subplots(figsize=(20,15))
# sns.heatmap(x_y_msk, vmax=0.8, square=True)
# plt.title('Covariance Matrix of Features before Dropping',fontsize =16)

#Create a list of all correlated features > 0.90
drop_vars = [c for c in x_y_msk.columns if any(x_y_msk[c] > 0.90)]

# rankin highest correlated pairs
s = corrmatrix.unstack()
so = s.sort_values(ascending=False)
print('Top correlated Variables over 90%')
print(so[237:262])
top_cov = list(top_40.index)
# 1 of pair 2 be removed
corr_pair = {'SalePrice','SaleCondition_3','Exterior1st_1','GarageFinish_4','GarageType_5','LotShape_2' } 
top_cov = [ele for ele in top_cov if ele not in corr_pair] 
# Select only the Top rated features X
Train_Set = Train_Set.dropna(axis=0,how='any')
Train_Set = Train_Set[top_cov]
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler,RobustScaler,FunctionTransformer
from sklearn.preprocessing import StandardScaler

#Scale the data
rbst_scaler = RobustScaler()
train_std = rbst_scaler.fit_transform(Train_Set)
# train_rbst = X

pca=PCA().fit(train_std)
#pca=PCA(35).fit(X)
plt.figure(figsize=(24,15))
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.xticks(np.arange(0, 48, 1))
plt.xlabel('Number of components',fontweight='bold',size=14)
plt.ylabel('Explanined variance ratio for number of components',fontweight='bold',size=14)

pca_50 = PCA(27)
train_pca=pca_50.fit_transform(train_std)
train_pca.shape

# Variance of the 22 principal components
Var_count = list(pca_50.explained_variance_ratio_)
print(sum(Var_count))
print(Var_count)
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)

nbrs = neigh.fit(train_std)
distances, indices = nbrs.kneighbors(train_std)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(15,15))
plt.plot(distances)
plt.title('K Distance Curve')
plt.xlabel('Number of datapoints',fontweight='bold',size=14)
plt.ylabel('Normalized Distance away from nearest neighbour',fontweight='bold',size=14)
# plt.yticks(np.arange(0, 8, 0.5))
from sklearn.cluster import DBSCAN
# Initialize the DB Scan
dbscan = DBSCAN(eps=4, min_samples= 20).fit(train_pca)
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels=dbscan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
plt.style.use('Solarize_Light2')
unique_labels = set(labels)
plt.figure(figsize=(12,12))
colors = [plt.cm.prism(each)  for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    
    xy = train_pca[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = train_pca[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
# label these outliers above as -1
labels=pd.DataFrame(labels,columns=['Classes'])
print(labels[labels['Classes']==-1])
# Concat outliers with Train Set
X=pd.concat([Train_Set,labels],axis=1)

X = X.dropna(axis=0,how='any')

# located outlier indicies
out_inx = list(X[X.Classes==-1].index)
(out_inx)

#Drop the outliers
X.drop(out_inx,axis=0,inplace=True)
# Merge X and Y
x_y = pd.concat([X, y_tr],axis =1)

# Remove Nans and Infinitis
x_y = x_y[~x_y.isin([np.nan, np.inf, -np.inf]).any(1)]

# Drop the classes column column
x_y.drop(['Classes'],axis=1,inplace=True)
x_y.shape
Y = x_y['SalePrice']

X = x_y.drop(['SalePrice'],axis=1)

# Train,Test Split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1, random_state=0)
# This is the Y Test!!!
samp_sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv',index_col=0)
samp_sub.head(10)
# Ensure the real test set uses only the same variables from the X_test
Test_Set_red = pd.DataFrame(Test_Set, columns=X_test.columns)

#Check any Nans
Test_Nans = Test_Set_red[Test_Set_red.isna().any(axis=1)]
print('Row with Nan Values:\n\n',Test_Nans)

# We see the kitchenQual has a Nan,and as we did with the train X,we replace "Nan" with "0"
Test_Set_red["KitchenQual"] = Test_Set_red["KitchenQual"].fillna(0)

# This is the X Test!!!!
Test_Set_red
# Intersection Function needed 
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3


# The list of variables to unskew the first time with Powertransformer
fix_1st = intersection(top_cov, numeric_after)

# The list of variables to unskew the second time with MinMax and ExperimentalTransformer
fix_2st = intersection(top_cov, skew_over_1)
# Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer


#Create the list of column transformers
Skew_Transf = [('pow_trans', PowerTransformer(method='yeo-johnson'), fix_1st),
                ("MinMax", MinMaxScaler(), fix_2st),
                  ('custom_trans',ExperimentalTransformer_2(intersection(top_cov, skew_over_1)),fix_2st )]


# Apply to the Column Transformer function
col_tran = ColumnTransformer(transformers=Skew_Transf,remainder='passthrough')

# Set the pipeline function for the independent variables
pipeline = Pipeline(steps=[('prep',col_tran),('Lasso', Lasso())])

# Initialize the trans reg to rescale the target 
model = TransformedTargetRegressor(regressor=pipeline,func=np.log1p,inverse_func=np.expm1) #, transformer = transformer

# Tuning Parameters
parameters={'regressor__Lasso__alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,10,100,300,500,700,1000,10000]}

cv = KFold(n_splits=5, random_state=1, shuffle=True)

Lasso_reg_MSE=GridSearchCV(model,parameters, scoring=['neg_mean_squared_error'], cv=cv,refit='neg_mean_squared_error')
#Lasso_reg_MSE.get_params().keys()
Lasso_reg_MSE.fit(X_train, y_train)
#yhat 
yhat = Lasso_reg_MSE.predict(Test_Set_red)
#print(Lasso_reg_MSE.best_params_)

print('Test MSE',mean_squared_error(np.log1p(samp_sub), np.log1p(yhat)))

# Lets output 15 values to compare
pred = pd.DataFrame((yhat),columns=['Predicted SalePrice'])
pred.reset_index(drop=True, inplace=True)

obs = pd.DataFrame((samp_sub))
obs.reset_index(drop=True, inplace=True)

lasso_y = pd.concat([obs,pred],axis=1,join='inner')
lasso_y
#Create the list of column transformers
Skew_Transf = [('pow_trans', PowerTransformer(method='yeo-johnson'), fix_1st),
                ("MinMax", MinMaxScaler(), fix_2st),
                  ('custom_trans',ExperimentalTransformer_2(intersection(top_cov, skew_over_1)),fix_2st)]


# Apply to the Column Transformer function
col_tran = ColumnTransformer(transformers=Skew_Transf,remainder='passthrough')

# Set the pipeline function for the independent variables
pipeline = Pipeline(steps=[('prep',col_tran),('Ridge', Ridge())])

# Initialize the trans reg to rescale the target 
modelrd = TransformedTargetRegressor(regressor=pipeline,func=np.log1p,inverse_func=np.expm1) 

# Ridge Parameters
parameters={'regressor__Ridge__alpha':[1e-15,1e-10,1e-8,1e-3,1,5,10,20,30,40,50,70,100,200,400,700,1000,100000]}

#Initialize the cross validation
cv = KFold(n_splits=5, random_state=1, shuffle=True)

# Gradserach for optimal parameters
ridge_reg_MSE=GridSearchCV(modelrd,parameters,scoring=['neg_mean_squared_error'],cv=cv,refit='neg_mean_squared_error')
# fit the model
ridge_reg_MSE.fit(X_train,y_train)
#ridge_reg_MSE.get_params().keys()
#predict the y's
yhatrd= ridge_reg_MSE.predict(Test_Set_red)

print('Test MSE',mean_squared_error(np.log1p(samp_sub), np.log1p(yhatrd)))

# Lets output 15 values to compare
pred_15 = pd.DataFrame((yhatrd)[0:15],columns=['Predicted SalePrice'])
pred_15.reset_index(drop=True, inplace=True)

obs_15 = pd.DataFrame((samp_sub)[0:15])
obs_15.reset_index(drop=True, inplace=True)
rid_y = pd.concat([obs_15,pred_15],axis=1,join='inner')
rid_y

#Create the list of column transformers
Skew_Transf = [('pow_trans', PowerTransformer(method='yeo-johnson'), fix_1st),
                ("MinMax", MinMaxScaler(), fix_2st),
                  ('custom_trans',ExperimentalTransformer_2(intersection(top_cov, skew_over_1)),fix_2st)]


# Apply to the Column Transformer function
col_tran = ColumnTransformer(transformers=Skew_Transf,remainder='passthrough')

# Set the pipeline function for the independent variables
pipeline = Pipeline(steps=[('prep',col_tran),('xgb', xgb.XGBRegressor())])

# Initialize the trans reg to rescale the target 
modelxgb = TransformedTargetRegressor(regressor=pipeline,func=np.log1p,inverse_func=np.expm1) 
parameters = {'regressor__xgb__learning__rate': [0.01,0.05, 0.07],
              'regressor__xgb__max__depth': [2,3,4,5, 6, 7],
              'regressor__xgb__n__estimators': [100,200,300,600,1200,2400]}

#Initialize the cross validation
cv = KFold(n_splits=5, random_state=1, shuffle=True)

# Gradserach for optimal parameters
XGB_Grid =GridSearchCV(modelxgb,parameters,scoring=['neg_mean_squared_error'],cv=cv,refit='neg_mean_squared_error')
# modelxgb.get_params().keys()
#modelxgb.get_params().keys()
XGB_Grid.fit(X_train,y_train)
pred_XG = XGB_Grid.predict(Test_Set_red)

print('Test MSE',mean_squared_error(np.log1p(samp_sub), np.log1p(pred_XG)))

#inverse transform the Y values
# Lets output 15 values to compare
pred_15 = pd.DataFrame((pred_XG)[0:15],columns=['Predicted SalePrice'])
pred_15.reset_index(drop=True, inplace=True)

obs_15 = pd.DataFrame((samp_sub)[0:15])
obs_15.reset_index(drop=True, inplace=True)

XGB_pred = pd.concat([obs_15,pred_15],axis=1,join='inner')
XGB_pred
# Linear model with Recurrsive Feature Elimination

#Create the list of column transformers
Skew_Transf = [('pow_trans', PowerTransformer(method='yeo-johnson'), fix_1st),
                ("MinMax", MinMaxScaler(), fix_2st),
                  ('custom_trans',ExperimentalTransformer_2(intersection(top_cov, skew_over_1)),fix_2st)]


# Make the pipeline
pipeline = Pipeline([('pow_trans',PowerTransformer(method='yeo-johnson')),
                     ('s', RFE(estimator = LinearRegression())),
                      ('m',LinearRegression())])

# Transform the SalePrice
fmodel = TransformedTargetRegressor(regressor=pipeline,func=np.log1p,inverse_func=np.expm1) 

parameters={'regressor__s__n_features_to_select': [1,2,3,4,5,6,7,8,9,10,15,16,17,18,20,25,26,27,28,29,30,35,40]}

cv = KFold(n_splits=5, random_state=1, shuffle=True)

RFE_MS = GridSearchCV(fmodel,parameters,scoring=['neg_mean_squared_error'],cv=cv,refit='neg_mean_squared_error',verbose=1)
# RFE_MS.get_params().keys() 
RFE_MS.fit(X_train,y_train)
y_predrfe = RFE_MS.predict(Test_Set_red)

print('Test MSE',mean_squared_error(np.log1p(samp_sub), np.log1p(y_predrfe)))

# Lets output 15 values to compare
pred_15 = pd.DataFrame((y_predrfe),columns=['Predicted SalePrice'])
pred_15.reset_index(drop=True, inplace=True)

obs_15 = pd.DataFrame((samp_sub))
obs_15.reset_index(drop=True, inplace=True)

RFE_res = pd.concat([obs_15,pred_15],axis=1,join='inner')
RFE_res 
# Submission from Lasso
results_lasso = lasso_y.copy()

results_lasso = results_lasso.iloc[:,[1]]

results_lasso.rename(columns={"Predicted SalePrice": "SalePrice"},inplace=True)
results_lasso = results_lasso[["SalePrice"]].round(2)

results_lasso['Id'] = results_lasso.index + 1461

results_lasso = results_lasso[["Id", "SalePrice"]]
results_lasso.head(5)
# results_lasso.to_csv(r"C:\Users\Rae-Djamaal\Anaconda3\Lib\Git_Uploads\house-prices-advanced-regression-techniques\submission.csv",index = False)