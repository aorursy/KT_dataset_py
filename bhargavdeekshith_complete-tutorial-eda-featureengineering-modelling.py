import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
from scipy.stats import norm
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train.shape
train.info()
train.drop('Id',axis=1,inplace=True)
num_feat = train.select_dtypes(include=[np.number])
num_feat.shape
num_feat.columns
cat_feat = train.select_dtypes(include='object')
cat_feat.shape
cat_feat.columns
def filtercol(df):
    numerical = []
    for i in df.columns:
        if len(df[i].value_counts().index)>20 and 'Yr' not in i and 'Year' not in i and 'Mo' not in i:
            numerical.append(i)
    return numerical
numerical_features = filtercol(num_feat)
numerical_features
numerical_features = numerical_features[:-1]+['3SsnPorch']+['PoolArea']
numerical_features
train.corr()['SalePrice'].sort_values(ascending=False)[1:]
percentage = ((train.isnull().sum() / len(train)) * 100).sort_values(ascending=False)
missing_count = train.isnull().sum().sort_values(ascending=False)
missing_df = pd.concat([missing_count,percentage],axis=1,keys=['Missing_count','Percentage'])
missing_df[missing_df['Percentage']>0.00]
train['PoolQC'] = train['PoolQC'].fillna('NA')
train['MiscFeature'] = train['MiscFeature'].fillna('NA')
train['Alley'] = train['Alley'].fillna('NA')
train['Fence'] = train['Fence'].fillna('NA')
train['FireplaceQu'] = train['FireplaceQu'].fillna('NA')
train.drop(['LotFrontage'],axis=1,inplace=True)
train['GarageType'] = train['GarageType'].fillna('NA')
train['GarageCond'] = train['GarageCond'].fillna('NA')
train['GarageFinish'] = train['GarageFinish'].fillna('NA')
train['GarageQual'] = train['GarageQual'].fillna('NA')
train['GarageYrBlt'] = train['GarageYrBlt'].fillna('NA')
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('NA')
train['BsmtExposure'] = train['BsmtExposure'].fillna('NA')
train['BsmtQual'] = train['BsmtQual'].fillna('NA')
train['BsmtCond'] = train['BsmtCond'].fillna('NA')
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('NA')
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
train['MasVnrType'] = train['MasVnrType'].fillna('None')
train['Electrical'].mode()
train['Electrical'] = train['Electrical'].fillna('SBrkr')
train.isnull().sum()
from scipy.stats import norm
sns.distplot(train['SalePrice'],fit=norm)
sns.barplot(y = train['SalePrice'],x = train['OverallQual'])
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected = True)
sns.scatterplot(x = train['GrLivArea'],y = train['SalePrice'])
train[train['GrLivArea']>4000][['GrLivArea','TotalBsmtSF','SalePrice']]
train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index,inplace=True)
sns.scatterplot(x = train['GrLivArea'],y = train['SalePrice'])
plt.figure(figsize=(8,6))
sns.barplot(train['GarageCars'],train['SalePrice'])
print(train[train['GarageCars']==3].shape)
print(train[train['GarageCars']==4].shape)
sns.scatterplot(train['TotalBsmtSF'],train['SalePrice'])
sns.boxplot(train['FullBath'],train['SalePrice'])
sns.scatterplot(train['YearBuilt'],train['SalePrice'])
def valuecounts(dataframe):
    for i in dataframe.columns:
        print(dataframe[i].value_counts())
valuecounts(cat_feat)
cat_feat.columns
from scipy.stats import ttest_ind, f_oneway
s1 = train[train['Street']=='Grvl']['SalePrice']
s2 = train[train['Street']=='Pave']['SalePrice']
ttest_ind(s1,s2)
train.drop('Street',axis=1,inplace=True)
train['RoofMatl'].value_counts()
r1 = train[train['RoofMatl']=='CompShg']['SalePrice']
r2 = train[train['RoofMatl']=='Tar&Grv']['SalePrice']
r3 = train[train['RoofMatl']=='WdShngl']['SalePrice']
r4 = train[train['RoofMatl']=='WdShake']['SalePrice']
r5 = train[train['RoofMatl']=='Membran']['SalePrice']
r6 = train[train['RoofMatl']=='Metal']['SalePrice']
r7 = train[train['RoofMatl']=='Roll']['SalePrice']
f_oneway(r1,r2,r3,r4,r5,r6,r7)
p1 = train[train['PavedDrive']=='Y']['SalePrice']
p2 = train[train['PavedDrive']=='N']['SalePrice']
p3 = train[train['PavedDrive']=='P']['SalePrice']
f_oneway(p1,p2,p3)
numerical_features = numerical_features[1:]
numerical_features
for i in numerical_features[:-1]:
    plt.scatter(train[i],train['SalePrice'])
    plt.title(i)
    plt.show()    
corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(train[top_corr_features].corr(),annot=True,cbar=True)
num = train[numerical_features]
num.shape
num.head(10)[['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']]
num['TotalPorch'] = num['WoodDeckSF']+num['OpenPorchSF']+num['EnclosedPorch']+num['3SsnPorch']+num['ScreenPorch']
num[num['GrLivArea']==(num['1stFlrSF']+num['2ndFlrSF']+num['LowQualFinSF'])].shape
num[num['TotalBsmtSF'] == num['BsmtFinSF1']+num['BsmtFinSF2']+num['BsmtUnfSF']].shape
num.columns
label_df = train[['MSSubClass','OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure',
                  'HeatingQC','KitchenQual','Fireplaces','FireplaceQu','GarageQual','GarageCond','GarageCars','PoolQC','BsmtFullBath',
                  'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd']]
label_df.replace({'Ex':5,'Gd':4,'TA':3,'Av':3,'Mn':2,'Fa':2,'Po':1,'No':1,'NA':0},inplace=True)
label_df.head()
label_df.shape
dummies_df = train[['MSZoning','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
                    'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
                    'Exterior2nd','MasVnrType','Foundation','BsmtFinType1','BsmtFinType2','Heating','CentralAir',
                    'Electrical','Functional','GarageType','GarageFinish','PavedDrive','Fence','MiscFeature','MoSold',
                    'SaleType','SaleCondition']]
dummies_df.shape
year_df = []
for i in train.columns:
    if 'Yr' in i or 'Year' in i:
        year_df.append(i)
print(year_df)
year_df = train[year_df]
year_df.head()
year_df.shape
year_df['Age'] = year_df['YrSold'] - year_df['YearBuilt']
year_df['Age'].head()
year_df['AgeAfterRemod'] = year_df['YrSold'] - year_df['YearRemodAdd']
year_df['AgeAfterRemod'].head()
year_df['Age'].shape,year_df['AgeAfterRemod'].shape
df_multi = pd.concat([num,label_df,year_df['Age'],year_df['AgeAfterRemod']],axis=1)
df_multi.head()
df_multi.shape
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
calc_vif(df_multi)
numfeat_reg = df_multi.copy()
numfeat_reg.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GarageCars','WoodDeckSF',
           'OpenPorchSF','EnclosedPorch','ScreenPorch','3SsnPorch','BedroomAbvGr','ExterQual','BsmtQual','KitchenQual',
           'GarageQual','ExterCond','BsmtCond','GarageCond','KitchenAbvGr','HeatingQC','TotRmsAbvGrd','PoolQC','FireplaceQu']
          ,axis=1,inplace=True)
calc_vif(numfeat_reg)
numfeat_reg.shape
num_reg = filtercol(numfeat_reg)
num_reg
num_reg = df_multi[num_reg]
num_reg.head()
from numpy import mean
from numpy import std
out_per=[]
for i in num_reg.columns:
    data_mean, data_std = mean(num_reg[i]), std(num_reg[i])
# identify outliers
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    print(i,': \n')
# identify outliers
    outliers = [x for x in num_reg[i] if x < lower or x > upper]
    
    num_out=len(outliers)
    print('Identified outliers: %d' % num_out)
    outliers_removed = [x for x in num_reg[i] if x >= lower and x <= upper]
    num_nout=len(outliers_removed)
    print('Non-outlier observations: %d' % num_nout)
    outlier_percent=(num_out/(num_out+num_nout))*100
    print('percent of outlers:',outlier_percent ,'\n')
    out_per.append(outlier_percent)
Outliers=pd.DataFrame({'Feature':num_reg.columns,'% Of Outliers':out_per})
outlier_sorted=Outliers.sort_values('% Of Outliers',ascending=False)
outlier_sorted
plt.figure(figsize=(10,8))
sns.barplot(y=outlier_sorted['Feature'],x=outlier_sorted['% Of Outliers'],palette='GnBu_d')
plt.title('Percent of Outliers by columns')
plt.ylabel('Column Name')
for i, v in enumerate(list(outlier_sorted['% Of Outliers'])):
    plt.text(v,i-(-0.1),round(list(outlier_sorted['% Of Outliers'])[i],2),fontsize=8)
    
    plt.savefig('Outliers.jpeg',bbox_inches='tight',dpi=150)
len(numerical_features)
import scipy.stats as st
for column in num_reg.columns:
    
    #original transformation
    print('\033[1m'+column.upper()+'\033[0m','\n')
    plt.figure(figsize=(22,25))
    
    raw_skewness = num_reg[column].skew()
    
    plt.subplot(8,2,1)
    plt.hist(num_reg[column])
    plt.title('Original distribution')
    
    plt.subplot(8,2,2)
    st.probplot(num_reg[column],dist='norm',plot=plt)
    
    
    #log transformation
    log_transform = np.log(num_reg[column]+1)
    log_skew=log_transform.skew()
    
    plt.subplot(8,2,3)
    plt.hist(log_transform)
    plt.title('Log Transformation')
    
    plt.subplot(8,2,4)
    st.probplot(log_transform,dist='norm',plot=plt)
    
    #Reciprocal Transformation  
    
    recip_transform = 1/(num_reg[column]+1)
    recip_skew=recip_transform.skew()
    
    plt.subplot(8,2,5)
    plt.hist(recip_transform)
    plt.title('Reciprocal Transformation')
    
    plt.subplot(8,2,6)
    st.probplot(recip_transform,dist='norm',plot=plt)
    
    #Exponential Transformation
    
    exp_2 = num_reg[column]**0.2
    exp_2_skew=exp_2.skew()
    
    plt.subplot(8,2,7)
    plt.hist(exp_2)
    plt.title('exp_2 Transformation')
    
    plt.subplot(8,2,8)
    st.probplot(exp_2,dist='norm',plot=plt)
    
    exp_3 = num_reg[column]**0.3
    exp_3_skew=exp_3.skew()
    
    plt.subplot(8,2,9)
    plt.hist(exp_3)
    plt.title('exp_3 Transformation')
    
    plt.subplot(8,2,10)
    st.probplot(exp_3,dist='norm',plot=plt)
    
    #Square Root Transformation
    
    sqrt_transform = num_reg[column]**(1/2)
    sqrt_transform_skew=sqrt_transform.skew()
    
    plt.subplot(8,2,11)
    plt.hist(sqrt_transform)
    plt.title('Square Root Transformation')
    
    plt.subplot(8,2,12)
    st.probplot(sqrt_transform,dist='norm',plot=plt)
    
    #Cube Root Transformation
    
    cube_transform = num_reg[column]**(1/3)
    cube_transform_skew=cube_transform.skew()
    
    plt.subplot(8,2,13)
    plt.hist(cube_transform)
    plt.title('Cube Root Transformation')
    
    plt.subplot(8,2,14)
    st.probplot(cube_transform,dist='norm',plot=plt)
    
    #Boxcox Transformation
    
    box,param = st.boxcox(num_reg[column]+1)
    boxcox_skew=pd.DataFrame(box).skew()
    
    plt.subplot(8,2,15)
    plt.tight_layout()
    plt.hist(pd.DataFrame(box))
    plt.title('Boxcox Transformation')
    
    plt.subplot(8,2,16)
    st.probplot(box,dist='norm',plot=plt)
    
    trans_result= {'Actual':raw_skewness, 'Log':log_skew,'Reciprocal':recip_skew,'Exponential power 0.2':exp_2_skew,
                       'Exponential power 0.3':exp_3_skew,'Square Root':sqrt_transform_skew,
                       'Cube Root':cube_transform_skew,'Boxcox':boxcox_skew[0]}
    print(pd.DataFrame(trans_result.items(), columns=['Transformation', 'Skew']).to_string(index=False))
    
    lst=list(trans_result.values())
    idx = min((abs(x), x) for x in lst)[1]
    for i in trans_result:
        if (trans_result[i]==idx):
            print('\n','Best Transformation for ',column,':','\n',i,'=',trans_result[i])
    plt.tight_layout() 
    
    plt.show()
num_reg.columns
num_reg['LotArea'],i = st.boxcox(num_reg['LotArea']+1)
num_reg['MasVnrArea'] = 1/(num_reg['MasVnrArea']+1)
num_reg['GrLivArea'],m = st.boxcox(num_reg['GrLivArea']+1)
num_reg['MiscVal'],r = st.boxcox(num_reg['MiscVal']+1)
num_reg['TotalPorch'] = num_reg['TotalPorch']**0.5
numfeat_reg.columns
numfeat_reg = pd.concat([num_reg,numfeat_reg[['MSSubClass', 'OverallQual',
       'OverallCond', 'BsmtExposure', 'Fireplaces', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath']]],axis=1)
numfeat_reg.head()
numfeat_reg.shape
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
numfeat_reg = ss.fit_transform(numfeat_reg)
numfeat_reg
numfeat_reg.shape
numfeat_reg = pd.DataFrame(numfeat_reg,columns=['LotArea', 'MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea',
       'MiscVal', 'TotalPorch', 'Age', 'AgeAfterRemod', 'MSSubClass', 'OverallQual',
       'OverallCond', 'BsmtExposure', 'Fireplaces', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath'])
numfeat_reg.head()
numfeat_reg.shape
numfeat_reg.isnull().sum()
dummies_df = pd.get_dummies(dummies_df,drop_first=True)
dummies_df.shape
dummies_df.drop('MoSold',axis=1,inplace=True)
pd.set_option('display.max_seq_items', None)
dummies_df.columns
df_final_reg = pd.DataFrame(np.hstack([numfeat_reg,dummies_df]),columns=['LotArea', 'MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea',
       'MiscVal', 'TotalPorch', 'Age', 'AgeAfterRemod', 'MSSubClass', 'OverallQual',
       'OverallCond', 'BsmtExposure', 'Fireplaces', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath','MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Alley_NA', 'Alley_Pave', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'Utilities_NoSeWa', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'LandSlope_Mod', 'LandSlope_Sev', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe',
       'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_ImStucc', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn',
       'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc', 'Exterior2nd_MetalSd', 'Exterior2nd_Other', 'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_NA', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_NA', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'CentralAir_Y', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ',
       'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_NA', 'GarageFinish_NA', 'GarageFinish_RFn', 'GarageFinish_Unf', 'PavedDrive_P', 'PavedDrive_Y', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'Fence_NA', 'MiscFeature_NA', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial'])
df_final_reg.shape
df_final_reg.head()
df_final_reg.isnull().sum()
x = df_final_reg.copy()
x.shape
y = train['SalePrice']
y.shape
import statsmodels.api as sm
cols = list(x.columns)
pmax = 1
while (len(cols) > 0):
    p = []
    xc = x[cols]
    xc = sm.add_constant(xc)
    model = sm.OLS(y.values.reshape(-1,1),xc).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)
    pmax = max(p)
    feature_with_pmax = p.idxmax()
    if( pmax > 0.05 ):
        cols.remove(feature_with_pmax)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
len(selected_features_BE)
x_fs = x[selected_features_BE]
x_fs.shape
import statsmodels.api as sm
Xc = sm.add_constant(x_fs)
model = sm.OLS(y.values.reshape(-1,1),Xc).fit()
model.summary()
from sklearn.model_selection import train_test_split
x_trainreg,x_testreg,y_trainreg,y_testreg = train_test_split(x_fs,y,test_size = 0.3 ,random_state = 0)
x_tree = train.drop('SalePrice',axis=1)
y_tree = train['SalePrice']
from sklearn.model_selection import train_test_split
x_traintree,x_testtree,y_traintree,y_testtree = train_test_split(x_tree,y_tree,test_size = 0.3 ,random_state = 0)
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
LR = LinearRegression()
LR.fit(x_trainreg,y_trainreg)
y_predreg = LR.predict(x_testreg)
from sklearn import metrics
RMSE_LR = np.sqrt(metrics.mean_squared_error(y_testreg,y_predreg))
RMSE_LR
R2_LR = metrics.r2_score(y_testreg,y_predreg)
R2_LR
RD = Ridge(normalize=True)
LS = Lasso(normalize=True)
EN = ElasticNet(normalize=True)
from sklearn.model_selection import GridSearchCV
param_rd = {'alpha':np.arange(0.01,1,0.01)}
param_ls = {'alpha':np.arange(0.01,1,0.01)}
param_en = {'alpha':np.arange(0.01,1,0.01),'l1_ratio':np.arange(0.01,1,0.01)}
GS_RD = GridSearchCV(RD,param_rd,cv=5,scoring='neg_mean_squared_error')
GS_RD.fit(x_trainreg,y_trainreg)
GS_RD.best_params_
RD = Ridge(alpha = 0.09)
RD.fit(x_trainreg,y_trainreg)
y_pred_ridge = RD.predict(x_testreg)
RMSE_Ridge = np.sqrt(metrics.mean_squared_error(y_testreg,y_pred_ridge))
RMSE_Ridge
R2_Ridge = metrics.r2_score(y_testreg,y_pred_ridge)
R2_Ridge
GS_LS= GridSearchCV(LS,param_ls,cv=5,scoring='neg_mean_squared_error')
GS_LS.fit(x_trainreg,y_trainreg)
GS_LS.best_params_
LS = Lasso(alpha = 0.99)
LS.fit(x_trainreg,y_trainreg)
y_pred_lasso = LS.predict(x_testreg)
RMSE_Lasso = np.sqrt(metrics.mean_squared_error(y_testreg,y_pred_lasso))
RMSE_Lasso
R2_Lasso = metrics.r2_score(y_testreg,y_pred_lasso)
R2_Lasso
LS.coef_
GS_EN = GridSearchCV(EN,param_en,cv=5,scoring='neg_mean_squared_error')
GS_EN.fit(x_trainreg,y_trainreg)
GS_EN.best_params_
EN = ElasticNet(alpha = 0.01, l1_ratio= 0.98)
EN.fit(x_trainreg,y_trainreg)
y_pred_elastic = EN.predict(x_testreg)
RMSE_Elastic = np.sqrt(metrics.mean_squared_error(y_testreg,y_pred_elastic))
RMSE_Elastic
R2_Elastic = metrics.r2_score(y_testreg,y_pred_elastic)
R2_Elastic
df_tree = train.copy()
df_tree.shape
df_cat = df_tree.select_dtypes(include = 'object')
df_cat.columns
df_cat.head()
df_cat['GarageYrBlt'].dtype
df_cat.drop('GarageYrBlt',axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in df_cat.columns:
    df_cat[i] = le.fit_transform(df_cat[i])
df_cat.head()
df_num = df_tree.select_dtypes(include = [np.number])
df_num.head()
df_num.shape
df_num.info()
df_num['GarageYrBlt'] = train['GarageYrBlt']
df_num.shape
df_tree = pd.concat([df_num,df_cat],axis=1)
df_tree.head()
df_tree.shape
df_tree['GarageYrBlt']= df_tree['GarageYrBlt'].replace('NA',0)
df_tree['GarageYrBlt'].value_counts()
x_tree = df_tree.drop('SalePrice',axis=1)
y_tree = df_tree['SalePrice']
from sklearn.model_selection import train_test_split
x_traintree,x_testtree,y_traintree,y_testtree = train_test_split(x_tree,y_tree,test_size = 0.3 ,random_state = 0)
from sklearn.tree import DecisionTreeRegressor  
dtr = DecisionTreeRegressor(random_state = 4)
from sklearn.model_selection import RandomizedSearchCV
params = {'max_depth':[2,3,4,5,6],
         'min_samples_leaf':[1,2,3,4,5,6,7],
         'min_samples_split':[2,3,4,5,6,7,8,9,10],
         'criterion':['mse']}

rsearch_d = RandomizedSearchCV(dtr, param_distributions= params, cv = 5, scoring = 'neg_mean_squared_error',n_iter = 200,random_state = 4,return_train_score = True)
rsearch_d.fit(x_traintree,y_traintree)
rsearch_d.best_params_
dt = DecisionTreeRegressor(max_depth=5,min_samples_split=8,min_samples_leaf=2,
                           criterion='mse')
dt.fit(x_traintree,y_traintree)
y_pred_dt = dt.predict(x_testtree)
RMSE_DT = np.sqrt(metrics.mean_squared_error(y_testtree,y_pred_dt))
RMSE_DT
R2_DT = metrics.r2_score(y_testreg,y_pred_dt)
R2_DT
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint as sp_randint

rfc_tunned=RandomForestRegressor(n_estimators=100,random_state=0)
params={'n_estimators':sp_randint(1,100),
        'max_features':sp_randint(1,70),
        'max_depth': sp_randint(2,10),
        'min_samples_split':sp_randint(2,20),
        'min_samples_leaf':sp_randint(1,20),
        'criterion':['mse','r2']}

rsearch_rfc=RandomizedSearchCV(rfc_tunned,params,cv=3,scoring='neg_mean_squared_error',n_jobs=-1,random_state=0)

rsearch_rfc.fit(x_traintree,y_traintree)
rsearch_rfc.best_params_
rfr = RandomForestRegressor(criterion='mse',max_depth=9,max_features=65,min_samples_leaf=4,min_samples_split=5,n_estimators=10)
rfr.fit(x_traintree,y_traintree)
y_pred_rfr = rfr.predict(x_testtree)
RMSE_RF = np.sqrt(metrics.mean_squared_error(y_testtree,y_pred_rfr))
RMSE_RF
R2_RF = metrics.r2_score(y_testtree,y_pred_rfr)
R2_RF
metrics.r2_score(rfr.predict(x_traintree),y_traintree)
np.sqrt(metrics.mean_squared_error(rfr.predict(x_traintree),y_traintree))
from sklearn.svm import SVR
svr = SVR(kernel='rbf')
from sklearn.model_selection import GridSearchCV 
  
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(svr, param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(x_trainreg, y_trainreg)
grid.best_params_
svr = SVR(**grid.best_params_)
svr.fit(x_trainreg,y_trainreg)
y_pred_svr = svr.predict(x_testreg)
R2_SVM = metrics.r2_score(y_testreg, y_pred_svr)
print("SVM Regressor: ",R2_SVM)
RMSE_SVM = np.sqrt(metrics.mean_squared_error(y_testreg,y_pred_svr))
RMSE_SVM
from sklearn.ensemble import AdaBoostRegressor

params_AdbR_GS = {'learning_rate':[0.05,0.1,0.2,0.6,0.8,1],
        'n_estimators': [50,60,100],
                 'loss' : ['linear', 'square', 'exponential']}
model_AdaR_GS = GridSearchCV(AdaBoostRegressor(), param_grid = params_AdbR_GS, cv = 5, scoring='r2')
model_AdaR_GS.fit(x_traintree,y_traintree)
model_AdaR_GS.best_params_
ada = AdaBoostRegressor(**model_AdaR_GS.best_params_)
ada.fit(x_traintree,y_traintree)
pred_AdaR_GS = ada.predict(x_testtree)
R2_Ada = metrics.r2_score(y_testtree, pred_AdaR_GS)
print("Ada boost Regressor R2: ",R2_Ada)
RMSE_Ada = np.sqrt(metrics.mean_squared_error(y_testreg, pred_AdaR_GS))
print("AdaBoost Regressor RMSE: ",RMSE_Ada)
metrics.r2_score(ada.predict(x_traintree),y_traintree)
print("AdaBoost Regressor: ",np.sqrt(metrics.mean_squared_error(ada.predict(x_traintree),y_traintree)))
from xgboost import XGBRegressor

params_xgbR_GS = {"max_depth": [3,4,5,6,7,8],
              "min_child_weight" : [4,5,6,7,8],
            'learning_rate':[0.05,0.1,0.2,0.25,0.8,1],
            'n_estimators': [10,30,50,70,80,100]}
import warnings
warnings.filterwarnings("ignore")
 
model_xgbR_GS = GridSearchCV(XGBRegressor(), param_grid = params_xgbR_GS,cv = 5, scoring = 'r2')
model_xgbR_GS.fit(x_traintree,y_traintree)
model_xgbR_GS.best_params_
model_xgbR_GS.best_estimator_
xgb = XGBRegressor(**model_xgbR_GS.best_params_)
xgb.fit(x_traintree,y_traintree)
pred_xgb_GS = xgb.predict(x_testtree)
R2_XGB = metrics.r2_score(y_testtree, pred_xgb_GS)
print("XGboost Regressor R2: ",R2_XGB)
RMSE_XGB = np.sqrt(metrics.mean_squared_error(y_testreg, pred_xgb_GS))
print("XGBoost Regressor RMSE: ",RMSE_XGB)
metrics.r2_score(xgb.predict(x_traintree),y_traintree)
print("XGBoost Regressor: ",np.sqrt(metrics.mean_squared_error(xgb.predict(x_traintree),y_traintree)))
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection
r2_mean = []
for val in np.arange(1,100):
    GrBoost = GradientBoostingRegressor(n_estimators=val,random_state=0)
    kfold = model_selection.KFold(shuffle = True,n_splits = 5,random_state=0)
    result = model_selection.cross_val_score(GrBoost,x_tree,y_tree,cv=kfold,scoring='r2')
    r2_mean.append(np.mean(result))
plt.plot(np.arange(1,100),r2_mean)
r2_mean
np.argmax(r2_mean)
gb = GradientBoostingRegressor(n_estimators=98)
gb.fit(x_traintree,y_traintree)
pred_gb_GS = gb.predict(x_testtree)
metrics.r2_score(y_testtree, pred_gb_GS)
print("Gradient Boost Regressor: ",metrics.r2_score(y_testtree, pred_gb_GS))
print("Gradient Boost Regressor RMSE: ",np.sqrt(metrics.mean_squared_error(y_testtree, pred_gb_GS)))
print("Gradient Boosting Regressor: ",np.sqrt(metrics.mean_squared_error(gb.predict(x_traintree),y_traintree)))
metrics.r2_score(gb.predict(x_traintree),y_traintree)
from sklearn.model_selection import KFold
from sklearn import metrics
kf=KFold(n_splits=5,shuffle=True,random_state=0)
for model,name in zip([LR,LS,RD,EN],['Linear_Regression','Lasso Regression','Ridge Regression','ElasticNet']):
    rmse=[]
    for train_idx,test_idx in kf.split(x_fs,y):
        xtrain,xtest=x_fs.iloc[train_idx,:],x_fs.iloc[test_idx,:]
        ytrain,ytest=y.iloc[train_idx],y.iloc[test_idx]
        model.fit(xtrain,ytrain)
        y_pred=model.predict(xtest)
        mse=metrics.mean_squared_error(ytest,y_pred)
        rmse.append(np.sqrt(mse))
    print('RMSE scores: %0.03f (+/- %0.5f)[%s]' % (np.mean(rmse),np.std(rmse,ddof=1),name))
from sklearn.model_selection import KFold
from sklearn import metrics
kf=KFold(n_splits=5,shuffle=True,random_state=0)
for model,name in zip([dt,rfr],['Decision Tree','Random Forest']):
    rmse=[]
    for train_idx,test_idx in kf.split(x_tree,y_tree):
        xtrain,xtest=x_tree.iloc[train_idx,:],x_tree.iloc[test_idx,:]
        ytrain,ytest=y_tree.iloc[train_idx],y_tree.iloc[test_idx]
        model.fit(xtrain,ytrain)
        y_pred=model.predict(xtest)
        mse=metrics.mean_squared_error(ytest,y_pred)
        rmse.append(np.sqrt(mse))
    print('RMSE scores: %0.03f (+/- %0.5f)[%s]' % (np.mean(rmse),np.std(rmse,ddof=1),name))
from sklearn.model_selection import KFold
from sklearn import metrics
kf=KFold(n_splits=5,shuffle=True,random_state=0)
for model,name in zip([gb,xgb,ada],['Gradient Boost Regressor','XGBoost Regression','AdaBoost Regression']):
    rmse=[]
    for train_idx,test_idx in kf.split(x_tree,y_tree):
        xtrain,xtest=x_tree.iloc[train_idx,:],x_tree.iloc[test_idx,:]
        ytrain,ytest=y_tree.iloc[train_idx],y_tree.iloc[test_idx]
        model.fit(xtrain,ytrain)
        y_pred=model.predict(xtest)
        mse=metrics.mean_squared_error(ytest,y_pred)
        rmse.append(np.sqrt(mse))
    print('RMSE scores: %0.03f (+/- %0.5f)[%s]' % (np.mean(rmse),np.std(rmse,ddof=1),name))