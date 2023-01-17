import numpy as np
import pandas as pd
import datetime
import random

# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000
df_train=pd.read_csv('../input/home-data-for-ml-course/train.csv')
df_test=pd.read_csv("../input/home-data-for-ml-course/test.csv")
corrmat=df_train.corr()
f, ax = plt.subplots(figsize=(10,12))
sns.heatmap(corrmat,mask=corrmat<0.75,linewidth=0.5,cmap="Blues", square=True)


corrmat=df_train.corr()
corrmat['SalePrice'].sort_values(ascending=False).head(10)


k = 10 
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
f, ax = plt.subplots(figsize=(8,10))
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
df_train.plot.scatter(x='GrLivArea', y='SalePrice')
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train=df_train.drop(df_train[df_train['Id']==1299].index)
df_train=df_train.drop(df_train[df_train['Id']==524].index)


df_train.plot.scatter(x='TotalBsmtSF', y='SalePrice')
df_train.sort_values(by='TotalBsmtSF',ascending=False)[:2]
#df_train.drop(df_train[df_train['Id']==333].index,inplace=True)
df_train[df_train['TotalBsmtSF']>2000].sort_values(by='SalePrice',ascending=True)[:2]
df_train.drop(df_train[df_train['Id']==1224].index,inplace=True)
df_train.plot.scatter(x='TotalBsmtSF', y='SalePrice')
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x="OverallQual", y="SalePrice", data=df_train)
fig.axis(ymin=0, ymax=800000);
df_train.drop(df_train[(df_train.OverallQual==4) & (df_train.SalePrice>200000)].index,inplace=True)
#df_train.drop(df_train[(df_train.OverallQual==8) & (df_train.SalePrice>500000)].index,inplace=True)
df_train.plot.scatter(x='LotFrontage', y='SalePrice')
df_train.drop(df_train[df_train['LotFrontage'] > 200].index,inplace=True)
df_train.plot.scatter(x='LotArea', y='SalePrice')
df_train[df_train['LotArea']>100000]['OverallQual']
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearBuilt']
sns.pairplot(df_train[cols], size = 3)
plt.show()
#df_train.drop(df_train[df_train['TotalBsmtSF'] > 3000].index,inplace=True)
# This outlier is subjective, if you think houses with larger basements than ground level should be rare
#df_train[df_train['GrLivArea']<1500].sort_values(by='TotalBsmtSF',ascending=False)[:1]
#df_train.drop(df_train[df_train['Id']==154].index,inplace=True)
df_train.plot.scatter(y='TotalBsmtSF', x='GrLivArea')
df_train.plot.scatter(x='YearBuilt', y='SalePrice')
df_train.drop(df_train[(df_train.YearBuilt < 1900) & (df_train.SalePrice > 200000)].index,inplace=True)
df_train.drop(df_train[(df_train.YearBuilt < 2000) & (df_train.SalePrice > 650000)].index,inplace=True)
df_train.plot.scatter(x='YearBuilt', y='SalePrice')
corrmat1=df_train.corr()
corrmat1['SalePrice'].sort_values(ascending=False).head(10)
# concatenate training and testing sets to create new features and fill in missing values
target=df_train['SalePrice'].reset_index(drop=True)
trainx=df_train.drop(['SalePrice'],1)
all_features=pd.concat([trainx,df_test]).reset_index(drop=True)
total = all_features.isnull().sum().sort_values(ascending=False)
percent = (all_features.isnull().sum()/all_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)
missing_data['Percent'].head(15).plot.bar()
all_features['LotFrontage'] = all_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
all_features['GarageArea'] = all_features.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))
all_features['MSZoning'] = all_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
all_features['YrSold'] = all_features['YrSold'].astype(str)
all_features['MoSold'] = all_features['MoSold'].astype(str)
all_features['Functional'] = all_features['Functional'].fillna('Typ')
all_features['Electrical'] = all_features['Electrical'].fillna("SBrkr")
all_features['KitchenQual'] = all_features['KitchenQual'].fillna("TA")
all_features['Exterior1st'] = all_features['Exterior1st'].fillna(all_features['Exterior1st'].mode()[0])
all_features['Exterior2nd'] = all_features['Exterior2nd'].fillna(all_features['Exterior2nd'].mode()[0])
all_features['SaleType'] = all_features['SaleType'].fillna(all_features['SaleType'].mode()[0])
all_features['MSZoning'] = all_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#all_features = all_features.drop(['Functional','Electrical','Alley','RoofStyle','RoofMatl','GarageYrBlt',
#'Street','GarageFinish','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtFinSF1','BsmtFinSF2','Utilities','LandContour','LotShape'],1)

objects = []
for i in all_features.columns:
    if all_features[i].dtype == object:
        objects.append(i)
all_features.update(all_features[objects].fillna('None'))
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)
all_features.update(all_features[numeric].fillna(0))

# Important categorical features
from matplotlib.ticker import MaxNLocator
cat_feats=['MSZoning','Neighborhood','Condition1','BldgType','MasVnrType','ExterQual','BsmtQual','GarageQual']
def srt_box(y='SalePrice', df=df_train):
    fig, axes = plt.subplots(4, 2, figsize=(20,30))
    axes = axes.flatten()

    for i, j in zip(cat_feats, axes):

        sortd = df.groupby([i])[y].median().sort_values(ascending=False)
        sns.boxplot(x=i,
                    y=y,
                    data=df,
                    palette='plasma',
                    order=sortd.index,
                    ax=j)
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=14))

        plt.tight_layout()
srt_box()
# Creating new features
all_features['YearRemodAdd']=all_features['YearRemodAdd'].astype(int)
all_features['Years_Since_Remod'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)
    
all_features['Age']=all_features['YrSold'].astype(int) - all_features['YearBuilt'].astype(int)
all_features['Newness']=all_features['Age']*all_features['Years_Since_Remod']
    
all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']

feats=['2ndFlrSF','GarageArea','TotalBsmtSF','Fireplaces','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']
for col  in feats:
    name='Has_'+str(col)
    all_features[name]=all_features[col].apply(lambda x: 1 if x > 0 else 0)
    
all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +all_features['EnclosedPorch'] 
                                  + all_features['ScreenPorch'] +all_features['WoodDeckSF'])

all_features['Bsmt_Baths'] = all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath'])

all_features['Total_BathAbvGrd'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']))
# Assume bathrooms are half size of other rooms
all_features['AvgRoomSize']=all_features['GrLivArea']/(all_features['TotRmsAbvGrd']+(0.4*all_features['Total_BathAbvGrd']))
#Also captures size of rooms:
all_features['BedBath']=all_features['BedroomAbvGr']*all_features['Total_BathAbvGrd']

all_features['TotalLot'] = all_features['LotFrontage'] + all_features['LotArea']
all_features['sqft_feet_living']=all_features['TotalBsmtSF']+all_features['GrLivArea']

all_features.drop(['Id','BsmtFullBath','BsmtHalfBath'],1,inplace=True)

# Mapping neighborhood unique values according to the shades of the box-plot, nearly same median values
neigh_map={'None': 0,'MeadowV':1,'IDOTRR':1,'BrDale':1,
        'OldTown':2,'Edwards':2,'BrkSide':2,
        'Sawyer':3,'Blueste':3,'SWISU':3,'NAmes':3,
        'NPkVill':4,'Mitchel':4,'SawyerW':4,
        'Gilbert':5,'NWAmes':5,'Blmngtn':5,
        'CollgCr':6,'ClearCr':6,'Crawfor':6,
        'Somerst':8,'Veenker':8,'Timber':8,
         'StoneBr':10,'NoRidge':10,'NridgHt':10 } 
all_features['Neighborhood'] = all_features['Neighborhood'].map(neigh_map)
# Quality maps for external and basement

bsm_map = {'None': 0, 'Po': 1, 'Fa': 4, 'TA': 9, 'Gd': 16, 'Ex': 25}
#ordinal_map = {'Ex': 10,'Gd': 8, 'TA': 6, 'Fa': 5, 'Po': 2, 'NA':0}
ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']
for col in ord_col:
    all_features[col] = all_features[col].map(bsm_map)
all_features.shape

all_features[all_features['YrSold'].astype(int) < all_features['YearRemodAdd'].astype(int)]
all_features.at[2284,'Years_Since_Remod']=0
all_features.at[2538,'Years_Since_Remod']=0
all_features.at[2538,'Age']=0




#multi-collinearity
all_features.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','LotFrontage'], axis=1, inplace=True)

#Missing values
all_features.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)

# weakly correlated
all_features.drop(['MoSold','YrSold'], axis=1, inplace=True)

#features with same value
overfit_cat = []
for i in all_features.columns:
    counts = all_features[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(all_features) * 100 > 97:
        overfit_cat.append(i)
overfit_cat = list(overfit_cat)
#all_features.drop(overfit_cat,1,inplace=True)
print(overfit_cat)


overfit_cat=['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'LowQualFinSF', '3SsnPorch', 'Has_TotalBsmtSF', 'Has_3SsnPorch']
all_features.drop(overfit_cat,1,inplace=True)
all_features.isnull().sum().sort_values(ascending=True).head()
all_features.shape
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)

skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
print(skew_features.head(5))
#skew_features.tail(10)
for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
    
    
    
skew_features1 = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
print(skew_features1.head(3))
skew_features1.head(10)
target = np.log1p(df_train['SalePrice']).reset_index(drop=True)
sns.distplot(target, fit=norm);
fig = plt.figure()
from scipy import stats
res = stats.probplot(target, plot=plt)
def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res

log_features = ['BsmtUnfSF', 'TotalBsmtSF', 'GrLivArea', 'FireplaceQu', 'GarageArea', 'OpenPorchSF', 'EnclosedPorch', 
                'ScreenPorch', 'Years_Since_Remod', 'Newness','Total_Home_Quality', 'Total_porch_sf', 'AvgRoomSize', 
                'TotalLot', 'sqft_feet_living','BsmtFinSF1','BedBath']
all_features1=all_features
loged_features = logs(all_features, numeric)
all_features=logs(all_features,log_features)
feats=['2ndFlrSF','GarageArea','Fireplaces','WoodDeckSF','OpenPorchSF','EnclosedPorch','ScreenPorch']
for col  in feats:
    name1='Has_'+str(col) + '_log'
    loged_features.drop(name1,1,inplace=True)
for o in ord_col:
    name=str(o)+'_log'
    loged_features.drop(name,1,inplace=True)
loged_features.drop('Neighborhood_log',1,inplace=True)
loged_features.head()
collist=loged_features.columns[-36:]

qfeat=[]
for col in collist:
    name=str(col)
    q=name[:-4]
    qfeat.append(q)
squared_features = ['Years_Since_Remod','Total_Home_Quality','Total_porch_sf','AvgRoomSize',
                'TotalBsmtSF','GrLivArea','BedBath','YearBuilt',
                'Fireplaces','GarageArea','MasVnrArea']
squared_features=log_features
sqrd_features=[]
for l in squared_features:
    a=l+str("_log")
    sqrd_features.append(a)


def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res 


sq_features = squares(all_features, qfeat)

log_sq_cols=squares(loged_features,qfeat)

all_features=squares(all_features,sqrd_features)

def sqrt(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.sqrt(res[l])).values)   
        res.columns.values[m] = l + '_sqroot'
        m += 1
    return res 
def cube(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]**3).values)   
        res.columns.values[m] = l + '_cube3'
        m += 1
    return res 

root_features = sqrt(sq_features, qfeat)
cube_features = cube(sq_features, qfeat)
var_feats=cube(log_sq_cols,qfeat)
var_feats=sqrt(var_feats,qfeat)
cube_features.isnull().sum().sort_values(ascending=False).head()
var_feats.isnull().sum().sort_values(ascending=False).head()
all_features=pd.get_dummies(all_features).reset_index(drop=True)
all_features1=pd.get_dummies(all_features1).reset_index(drop=True)
loged_features=pd.get_dummies(loged_features).reset_index(drop=True)
cube_features=pd.get_dummies(cube_features).reset_index(drop=True)
var_feats=pd.get_dummies(var_feats).reset_index(drop=True)
log_sq_cols=pd.get_dummies(log_sq_cols).reset_index(drop=True)
def get_splits(all_features,target): 
    df=pd.concat([all_features,target],1)
    X_train=df.iloc[:len(target),:]
    X_test=all_features.iloc[len(target):,:]
    return X_train,X_test
def get_valid(df,target,valid_fraction=0.2):
    validrows=int(len(df)*valid_fraction)
    trains=df[:-validrows]
    valids=df[-validrows:]
    feature_col=df.columns.drop(target)
    return trains,valids,feature_col

# Split for feature engineering:
X_train,X_test=get_splits(all_features,target)
train,valid,feature_col=get_valid(X_train,'SalePrice')

X_train0,X_test0=get_splits(all_features1,target)
train0,valid0,feature_col0=get_valid(X_train0,'SalePrice')

X_train1,X_test1=get_splits(log_sq_cols,target)
train1,valid1,feature_col1=get_valid(X_train1,'SalePrice')

X_tr, X_te=get_splits(var_feats,target)
feat_col=X_tr.columns.drop('SalePrice')

tr_log,te_log=get_splits(loged_features,target)
log_feats=tr_log.columns.drop('SalePrice')
corr1=X_tr.corr()
corr1["SalePrice"].sort_values(ascending=False).head(15)
best_columns=['sqft_feet_living','GarageArea_log_sq','Age','BedBath','AvgRoomSize_log_sq','TotalLot_log_sq','Total_porch_sf_log_sq']
fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12,20))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(X_train[best_columns]), 1):
    plt.subplot(len(list(best_columns)), 2, i)
    sns.scatterplot(x=feature, y='SalePrice', hue=None, data=X_train)
        
    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 10})
        
plt.show()

from sklearn.linear_model import Lasso
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
kf = KFold(n_splits=12, random_state=7, shuffle=True)
def cv_rmse(model, X=X_train,target=target):
    rmse = np.sqrt(-cross_val_score(model, X, target, scoring="neg_mean_squared_error", cv=kf,n_jobs=-1))
    return (rmse)

#ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30,40, 50, 100]
feature_col.shape
## Function used for submitting predictions in the Kaggle competition scorer
def submission(preds,pred):
    submission=pd.read_csv("../input/home-data-for-ml-course/sample_submission.csv")
    submission.iloc[:,1] = np.floor(np.expm1(preds))

    q1 = submission['SalePrice'].quantile(0.0045)
    q2 = submission['SalePrice'].quantile(0.99)
    submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
    submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
    #submission.to_csv("%s1.csv" %pred, index=False)
    # Scale predictions
    submission['SalePrice'] *= 1.001619
    submission.to_csv('%s mycsvfile.csv'%pred,index=False)
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
X,y=X_train1[feature_col1],target

alphas = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,5e-3,5e-2]
Xscaled=RobustScaler().fit_transform(X)
lassopipe =LassoCV(max_iter=1e7, alphas=alphas, random_state=42,n_jobs=-1)   

clf=lassopipe.fit(Xscaled,y)
importance=np.abs(clf.coef_)
frame=pd.DataFrame(importance,index=feature_col1)
frame.sort_values(by=0,ascending=False).head(10)
imps=frame.sort_values(by=0,ascending=False).index
log_sq_cols.shape
# Using all features
from sklearn.pipeline import Pipeline
alphas=np.linspace(1,stop=50)
pipe=Pipeline([('scaler',RobustScaler()), ('ridgemodel',RidgeCV(alphas=alphas, 
                                                                cv=None,store_cv_values=True ))])
model=pipe.fit(X_train1[feature_col1],target)
print('Alpha: %f'%model['ridgemodel'].alpha_)
print('Score: %f'%np.sqrt(-model['ridgemodel'].best_score_))


# Using selected features
pred=model.predict(X_test1[feature_col1])
#submission(pred,pred='ridge2319')


alphas=np.linspace(1,stop=50,num=50)
pipe=Pipeline([('scaler',RobustScaler()), ('ridgemodel',RidgeCV(alphas=alphas, cv=None,store_cv_values=True))])
numfeats=[298,290,280,275,270,260,250]
for n in numfeats:
    model=pipe.fit(X_train1[imps[:n]],target)
    print(n)
    print('Alpha: %f'%model['ridgemodel'].alpha_)
    print('Score: %f'%np.sqrt(-model['ridgemodel'].best_score_))
    print('')
model=pipe.fit(X_train1[imps[:280]],target)
ridgepred=model.predict(X_test1[imps[:280]])
#submission(ridgepred,'ridge1619')



lightgbm = LGBMRegressor(objective='regression', num_leaves=5,learning_rate=0.007, n_estimators=3500,max_bin=163,
                       bagging_fraction=0.35711,bagging_freq=4, bagging_seed=8,feature_fraction=0.1294, feature_fraction_seed=8,
                       min_data_in_leaf = 8,  verbose=-1, random_state=42,n_jobs=-1)
lightgbmod=lightgbm.fit(X_train[feature_col],target)
lightpred=lightgbmod.predict(X_test[feature_col])
                   
(np.sqrt(-cross_val_score(lightgbm, X_train[feature_col], target, scoring="neg_mean_squared_error", cv=5))).mean()
#submission(lightpred,'lgpreds258')
from sklearn.pipeline import make_pipeline
svr = make_pipeline(RobustScaler(),
                    SVR(C=21, epsilon=0.0099, gamma=0.00017, tol=0.000121))
svrmodel=svr.fit(X_train[feature_col],target)
svrpred=svrmodel.predict(X_test[feature_col])


(cv_rmse(svr,X_train[feature_col])).mean()

#submission(svrpred,pred='svrpred258')
xgboost = XGBRegressor(
    learning_rate=0.0139,
    n_estimators=4500,
    max_depth=4,
    min_child_weight=0,
    subsample=0.7968,
    colsample_bytree=0.4064,
    nthread=-1,
    scale_pos_weight=2,
    seed=42,
)
xgboo=xgboost.fit(X_train[feature_col],target)
xgboostpred=xgboo.predict(X_test[feature_col])



stack_gen = StackingCVRegressor(regressors=(lightgbm, pipe,svr,xgboost),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
stack=stack_gen.fit(np.array(X_train[feature_col]),np.array(target))
stackpreds=stack.predict(np.array(X_test[feature_col]))



(cv_rmse(xgboost,X_train[feature_col])).mean()
subm= (0.2 * lightpred)+(0.4 * ridgepred)+(0.1 * stackpreds)+(0.3*svrpred)
submission(subm,pred='blended1705')
# First lets look at importance a naive model will attribute
import eli5
from eli5.sklearn import PermutationImportance
fc1=all_features1.drop(['sqft_feet_living','Age','Newness','Years_Since_Remod','Total_Home_Quality','TotalLot',
                   'Total_porch_sf','Bsmt_Baths','Total_BathAbvGrd','AvgRoomSize','BedBath'],1)
fc=fc1.columns

alphas=np.linspace(1,stop=50,num=50)
pipe0=Pipeline([('scaler',RobustScaler()), ('ridgemodel',RidgeCV(alphas=alphas, 
                                                                 cv=None,
                                                                 store_cv_values=True))])
model0=pipe0.fit(X_train0[fc],target)

perm0 = PermutationImportance(model0['ridgemodel'], random_state=1,cv=5).fit(X_train0[fc], 
                                                                             target)
eli5.show_weights(perm0, feature_names = X_train0[fc].columns.tolist())


# Importance attributed by best performing ridge model
import eli5
from eli5.sklearn import PermutationImportance

alphas=np.linspace(1,stop=50,num=50)
pipe1=Pipeline([('scaler',RobustScaler()), ('ridgemodel',RidgeCV(alphas=alphas, 
                                                                 cv=None,
                                                                 store_cv_values=True))])
x=X_train1[imps[:270]]
model1=pipe1.fit(x,target)

perm1 = PermutationImportance(model1['ridgemodel'], random_state=1,cv=5).fit(x, target)

eli5.show_weights(perm1, feature_names =x.columns.tolist())

#.drop(['YearRemodAdd','Newness','Age_sq','Years_Since_Remod_log','2ndFlrSF_log','sqft_feet_living_sq','ScreenPorch_sq'],1)

import shap  # package used to calculate Shap values

explainer = shap.LinearExplainer(model1['ridgemodel'],X_train1[imps[:270]])
shap_values = explainer.shap_values(X_train1[imps[:270]])
shap.summary_plot(shap_values, X_train1[imps[:270]],max_display=6)

x=tr_log[log_feats].drop(['2ndFlrSF','TotalBsmtSF','GarageArea','BsmtFinSF1','BsmtUnfSF'],1)
model1=pipe1.fit(x,target)
explainer = shap.LinearExplainer(model1['ridgemodel'],x)
shap_values = explainer.shap_values(x)
shap.summary_plot(shap_values, x,max_display=10)

