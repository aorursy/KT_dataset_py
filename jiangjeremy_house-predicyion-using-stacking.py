import pandas as pd

import sklearn

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

from sklearn.linear_model import Lasso,Ridge

from xgboost.sklearn import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from scipy import stats

from scipy.stats import norm, skew

from vecstack import stacking

import warnings

warnings.filterwarnings('ignore')
#导入数据

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

y = df_train.SalePrice
#查看训练集、测试集行列数

print(df_train.shape,df_test.shape)
#查看训练集前5行

df_train.head(5)
#查看测试集前5行

df_test.head()
#合并训练集和测试集特征列

all_data = pd.concat([df_train.drop(columns=['SalePrice']), df_test])



#删除多余列。'Utilities'列因为整列几乎都为同一个值，故删除；Id列与预测值无关，删除。

all_data = all_data.drop(['Utilities'], axis=1)

all_data = all_data.drop(['Id'], axis=1)
#查看各列缺失值情况

all_data.isnull().sum()[all_data.isnull().sum()!=0].sort_values(ascending=False)
#类别数据列，缺失值有实际含义，表示没有此特征，以‘None’填充

for col in ("PoolQC","MiscFeature","Alley","Fence","FireplaceQu",'GarageType',

            'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond',

            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"MasVnrType"):

    all_data[col] = all_data[col].fillna('None')



#float型数据列，无garage，年份以yearbuilt填充

all_data.loc[all_data.GarageYrBlt.isnull(),'GarageYrBlt'] = all_data.loc[all_data.GarageYrBlt.isnull(),'YearBuilt']



#float型数据列，缺失值有实际含义，表示没有此特征，以0填充

for col in ('GarageArea', 'GarageCars','BsmtFinSF1', "MasVnrArea",

            'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)



#类别数据列，且有5个以下缺失值，以众数填充

for col in ('MSZoning',"Functional",'Exterior1st','Exterior2nd','Electrical','KitchenQual','SaleType'):

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])



#LotFrontage列根据含义采取分组后以中位数填充方式

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



#检查是否还有剩余的缺失值

all_data.isnull().sum().sum()
#根据含义，将以下两数值型转换为类别型，以便后续处理

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)



#有序类别型数据转换为数值型

LotShape_map={'Reg':0,'IR1':1,'IR2':2,'IR3':3}

LandSlope_map={'Gtl':1,'Mod':2,'Sev':3}

ExterQual_map={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

ExterCond_map={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

BsmtQual_map={'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

BsmtCond_map={'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

BsmtExposure_map={'None':0,'No':1,'Mn':2,'Av':3,'Gd':4}

BsmtFinType1_map={'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}

BsmtFinType2_map={'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}

HeatingQC_map={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

CentralAir_map={'N':0,'Y':1}

KitchenQual_map={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

FireplaceQu_map={'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

GarageFinish_map={'None':0,'Unf':1,'RFn':2,'Fin':3}

GarageQual_map={'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

GarageCond_map={'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

PavedDrive_map={'N':0,'P':1,'Y':2}

PoolQC_map={'None':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}



all_data['LotShape']=all_data['LotShape'].map(LotShape_map)

all_data['LandSlope']=all_data['LandSlope'].map(LandSlope_map)

all_data['ExterQual']=all_data['ExterQual'].map(ExterQual_map)

all_data['ExterCond']=all_data['ExterCond'].map(ExterCond_map)

all_data['BsmtQual']=all_data['BsmtQual'].map(BsmtQual_map)

all_data['BsmtCond']=all_data['BsmtCond'].map(BsmtCond_map)

all_data['BsmtExposure']=all_data['BsmtExposure'].map(BsmtExposure_map)

all_data['BsmtFinType1']=all_data['BsmtFinType1'].map(BsmtFinType1_map)

all_data['BsmtFinType2']=all_data['BsmtFinType2'].map(BsmtFinType2_map)

all_data['HeatingQC']=all_data['HeatingQC'].map(HeatingQC_map)

all_data['CentralAir']=all_data['CentralAir'].map(CentralAir_map)

all_data['KitchenQual']=all_data['KitchenQual'].map(KitchenQual_map)

all_data['FireplaceQu']=all_data['FireplaceQu'].map(FireplaceQu_map)

all_data['GarageFinish']=all_data['GarageFinish'].map(GarageFinish_map)

all_data['GarageQual']=all_data['GarageQual'].map(GarageQual_map)

all_data['GarageCond']=all_data['GarageCond'].map(GarageCond_map)

all_data['PavedDrive']=all_data['PavedDrive'].map(PavedDrive_map)

all_data['PoolQC']=all_data['PoolQC'].map(PoolQC_map)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']

all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +

                                 all_data['1stFlrSF'] + all_data['2ndFlrSF'])



all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +

                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))



all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +

                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +

                              all_data['WoodDeckSF'])



all_data['HasBasement'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['Has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['HasWoodDeck'] = all_data['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['HasPorch'] = all_data['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['IsNew'] = all_data['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

all_data['HasFireplaces'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#查看预测值列直方图和正态概率图，可以看出不是正态分布

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

sns.distplot(y, fit=stats.norm);

plt.subplot(1,2,2)

_=stats.probplot(y, plot=plt)
y=np.log(y)
from scipy import stats 
#使用log变换之后基本呈正态分布

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

sns.distplot(y, fit=stats.norm);

plt.subplot(1,2,2)

_=stats.probplot(y, plot=plt)
#提取出离散型数据特征列和连续型数据特征列名

dtypes = all_data.dtypes

cols_numeric = dtypes[dtypes != object].index.tolist()



col_nunique = dict()



for col in cols_numeric:

    col_nunique[col] = all_data[col].nunique()

    

col_nunique = pd.Series(col_nunique)



cols_discrete = col_nunique[col_nunique<13].index.tolist()

cols_continuous = col_nunique[col_nunique>=13].index.tolist()
#提取出类别型数据特征列名

cols_categ = dtypes[~dtypes.index.isin(cols_numeric)].index.tolist()



for col in cols_categ:

    all_data[col] = all_data[col].astype('category')
# 类别型特征列与预测列

fcols = 3

frows = round(len(cols_categ)/fcols)+1

plt.figure(figsize=(15,4*frows))



for i,col in enumerate(cols_categ):

    plt.subplot(frows,fcols,i+1)

    sns.violinplot(all_data[:1460][col],y)
# 离散型数据特征列与预测列

fcols = 3

frows = round(len(cols_discrete)/fcols)+1

plt.figure(figsize=(15,4*frows))



for i,col in enumerate(cols_discrete):

    plt.subplot(frows,fcols,i+1)

    sns.violinplot(all_data[:1460][col],y)
# 连续型数据特征列与预测列

fcols = 2

frows = len(cols_continuous)

plt.figure(figsize=(5*fcols,4*frows))



i=0

for col in cols_continuous:

    i+=1

    ax=plt.subplot(frows,fcols,i)

    sns.regplot(x=all_data[:1460][col], y=y, ax=ax, 

                scatter_kws={'marker':'.','s':3,'alpha':0.3},

                line_kws={'color':'k'});

    plt.xlabel(col)

    plt.ylabel('SalePrice')

    

    i+=1

    ax=plt.subplot(frows,fcols,i)

    sns.distplot(all_data[:1460][col], fit=stats.norm)

    plt.xlabel(col)
numeric_feats = all_data.dtypes[all_data.dtypes != "category"].index



# 考察数值型特征列数据偏度

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
#对偏度大于0.5的列使用boxcox变换

to_do_box=list(skewness[skewness>0.5].dropna().index)

from scipy.stats import boxcox

for feat in to_do_box:

    all_data[feat] = boxcox(all_data[feat]+1)[0]
all_data = pd.get_dummies(all_data)

print(all_data.shape)
df_train = all_data.iloc[:(len(df_train))]

df_test = all_data.iloc[(len(df_train)):]

X_train = df_train.values

X_test = df_test.values

X = all_data.values

y = y.astype(np.float64)
from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

sc.fit(X_train)

X_train = sc.transform(X_train)

X_test = sc.transform(X_test)
def find_outliers(model,X,y,sigma=3):

    model.fit(X,y)

    y_pred=model.predict(X)

    resid=y-y_pred

    mean_resid=resid.mean()

    std_resid=resid.std()

    z=(resid-mean_resid)/std_resid

    outliers=z[abs(z)>sigma].index

    return(list(outliers))
outliers=find_outliers(Ridge(),X_train,y,3)
X_train=pd.DataFrame(X_train).drop(outliers).values
y=y.drop(outliers).values
#交叉验证，模型评价指标

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train,y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)



from sklearn.linear_model import Lasso



alphas = np.linspace(0.00, 0.003, 50)

cv_lasso = []

for alpha in alphas:

    model=Lasso(alpha = alpha,max_iter=5000)

    model.fit(X_train,y)

    cv_lasso.append(rmse_cv(model).mean())



alphas_rmse=pd.DataFrame()

alphas_rmse['alpha']=alphas

alphas_rmse['rmse']=cv_lasso

sns.lineplot(data=alphas_rmse,x='alpha',y='rmse')



#寻找最优参数

alphas_rmse.iloc[alphas_rmse['rmse'].idxmin(),:]
model_lasso =Lasso(alpha = alphas_rmse.iloc[alphas_rmse['rmse'].idxmin(),:]['alpha'],max_iter=5000)
alphas = np.linspace(0.1, 10, 50)

cv_ridge = []

for alpha in alphas:

    model = Ridge(alpha = alpha)

    model.fit(X_train,y)

    cv_ridge.append(rmse_cv(model).mean())

alphas_rmse=pd.DataFrame()

alphas_rmse['alpha']=alphas

alphas_rmse['rmse']=cv_ridge

sns.lineplot(data=alphas_rmse,x='alpha',y='rmse')
#寻找最优参数

alphas_rmse.iloc[alphas_rmse['rmse'].idxmin(),:]
model_ridge = Ridge(alpha = alphas_rmse.iloc[alphas_rmse['rmse'].idxmin(),:]['alpha'])
model_forest=RandomForestRegressor()

model_xgb=XGBRegressor()
models = [model_lasso,model_ridge,model_xgb,model_forest]
S_train, S_test = stacking(models,                     

                           X_train, y, X_test,   

                           regression=True,            

                           mode='oof_pred_bag',       

                           save_dir=None,             

                           metric=mean_squared_error, 

                           n_folds=5,                  

                           shuffle=True,               

                           random_state=0) 
model = LinearRegression()

model = model.fit(S_train, y)
test_y=model.predict(S_test)

test_price=pd.DataFrame(np.exp(test_y),columns=['SalePrice'])

test_price['Id']=[i for i in range(1461,2920,1)]

test_price=test_price.loc[:,['Id','SalePrice']]
test_price.to_csv('result.csv',index=False)