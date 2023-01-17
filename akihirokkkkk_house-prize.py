from sklearn import linear_model

clf = linear_model.LinearRegression()
clf.fit(X_train,y_train)

print(clf.coef_)#回帰係数
print(clf.intercept_)#切片
from sklearn import KMeans

km = KMeans(n_clusters=9, init="k-means++")
y_km = km.predict(train)

import pandas as pd
train_path = "../input/train.csv"

train = pd.read_csv(train_path)
train.head()
train.describe()
train.info()
#欠損値を補完
train["Alley"] = train["Alley"].fillna(0)
#要素の数を確認
train["MSZoning"].value_counts()
train.groupby("MSZoning").mean()
from matplotlib import pyplot as plt
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
import seaborn as sns
sns.distplot(train['SalePrice']);
train.columns[:20]
train.loc[1,:]
#correlation matrix
corrmat = train[train.columns[:40]].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#物件の広さを合計した変数を作成
df = train
df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"] + df["GrLivArea"]
fig = plt.figure(figsize=(9,6))
sns.regplot(x=df["TotalSF"], y=df["SalePrice"])
plt.xlabel("TotalSF")
plt.ylabel("SalePrice")
plt.show()
fig.savefig("figure4.png")


import pandas as pd
import numpy as np
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#id
test_id = test["Id"]

train = train.drop("Id",axis=1)
test = test.drop("Id",axis=1)

#Mssubclass
for i in [train,test]:
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-STORY 1946 & NEWER ALL STYLES" if x==20 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-STORY 1945 & OLDER" if x==30 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-STORY W/FINISHED ATTIC ALL AGES" if x==40 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-1/2 STORY - UNFINISHED ALL AGES" if x==45 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-1/2 STORY FINISHED ALL AGES" if x==50 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"2-STORY 1946 & NEWER" if x==60 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"2-STORY 1945 & OLDER" if x==70 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"2-1/2 STORY ALL AGES" if x==75 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"SPLIT OR MULTI-LEVEL" if x==80 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"SPLIT FOYER" if x==85 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"DUPLEX - ALL STYLES AND AGES" if x==90 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-STORY PUD (Planned Unit Development) - 1946 & NEWER" if x==120 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-1/2 STORY PUD - ALL AGES" if x==150 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"2-STORY PUD - 1946 & NEWER" if x==160 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"PUD - MULTILEVEL - INCL SPLIT LEV/FOYER" if x==180 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"2 FAMILY CONVERSION - ALL STYLES AND AGES" if x==190 else x)
del i

train['MSSubClass'] = train['MSSubClass'].fillna("None")
test['MSSubClass'] = test['MSSubClass'].fillna("None")

# poolqc
"""
PoolArea: Pool area in square feet

PoolQC: Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
"""
# poolQC
train["PoolQC"] = train["PoolQC"].fillna("None")
test["PoolQC"] = test["PoolQC"].fillna("None")
plt.scatter(train["LotFrontage"],train["SalePrice"])
#lotfrontage
train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


#MiscFeature
train["MiscFeature"] = train["MiscFeature"].fillna("None")
test["MiscFeature"] = test["MiscFeature"].fillna("None")

#Alley
train["Alley"] = train["Alley"].fillna("None")
test["Alley"] = test["Alley"].fillna("None")

#Fence
train["Fence"] = train["Fence"].fillna("None")
test["Fence"] = test["Fence"].fillna("None")

#fireplace
train["FireplaceQu"] = train["FireplaceQu"].fillna("None")
test["FireplaceQu"] = test["FireplaceQu"].fillna("None")
    
#garage系
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)
del col
# bsmtfin系
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)
    
for col in ("BsmtQual",'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train[col] = train[col].fillna("None")
    test[col] = test[col].fillna("None")
del col

# masvnr系
train["MasVnrType"] = train["MasVnrType"].fillna("None")
test["MasVnrType"] = test["MasVnrType"].fillna("None")

train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
test["MasVnrArea"] = test["MasVnrArea"].fillna(0)

#MSZoning
train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])

#Utilities
train = train.drop(['Utilities'], axis=1)
test = test.drop(['Utilities'], axis=1)

#Functional
train["Functional"] = train["Functional"].fillna("Typ")
test["Functional"] = test["Functional"].fillna("Typ")

#Electrical
train["Electrical"] = train["Electrical"].fillna("SBrkr")
test["Electrical"] = test["Electrical"].fillna("SBrkr")

# Kitchen
train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

#Exterior
train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

#saletype
train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])

#overallcond
train['OverallCond'] = train['OverallCond'].astype(str)
test['OverallCond'] = test['OverallCond'].astype(str)

#yrsold mosold
train['YrSold'] = train['YrSold'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)

# year
train['YrBltAndRemod']=train['YearBuilt']+train['YearRemodAdd']
test['YrBltAndRemod']=test['YearBuilt']+test['YearRemodAdd']

#total
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

train['Total_sqr_footage'] = (train['BsmtFinSF1'] + train['BsmtFinSF2'] + train['1stFlrSF'] + train['2ndFlrSF'])
test['Total_sqr_footage'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['1stFlrSF'] + test['2ndFlrSF'])

train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))
test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))

train['Total_porch_sf'] = (train['OpenPorchSF'] + train['3SsnPorch'] + train['EnclosedPorch'] + train['ScreenPorch'] + train['WoodDeckSF'])
test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])

#street
#train = train.drop('Street',axis=1)
#test = test.drop('Street',axis=1)


train = train.fillna(train.median())
test = test.fillna(test.median())

#train = train.sample(frac=1, random_state=0)

    
# 正規化
train["SalePrice"] = np.log(train["SalePrice"])
# encordingを定義する
from sklearn.preprocessing import LabelEncoder
def encoding(enc_train,enc_test):
    #エンコーディング
    co_box = []
    for co in enc_train.columns:
        try:
            sumup = enc_train[co].sum()
            if(type(sumup) == type("dokabenman")):
                co_box.append(co)
        except:
            print(co + ":エンコーディングおかしい！")

    for obj_col in co_box:
        le = LabelEncoder()
        enc_train[obj_col] = enc_train[obj_col].apply(lambda x:str(x))
        enc_train[obj_col] = pd.DataFrame({obj_col:le.fit_transform(enc_train[obj_col])})

        enc_test[obj_col] = enc_test[obj_col].apply(lambda x:str(x))
        enc_test[obj_col] = pd.DataFrame({obj_col:le.fit_transform(enc_test[obj_col])}) 
    
    return enc_train,enc_test

def train_only_encoding(enc_train):
    #エンコーディング
    co_box = []
    for co in enc_train.columns:
        try:
            sumup = enc_train[co].sum()
            if(type(sumup) == type("dokabenman")):
                co_box.append(co)
        except:
            print(co + ":エンコーディングおかしい！")

    for obj_col in co_box:
        le = LabelEncoder()
        enc_train[obj_col] = enc_train[obj_col].apply(lambda x:str(x))
        enc_train[obj_col] = pd.DataFrame({obj_col:le.fit_transform(enc_train[obj_col])})
    
    return enc_train
from sklearn import model_selection
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from IPython.display import clear_output 
from sklearn.model_selection import cross_val_score
# 分布がもお最も一致するrandom_stateを探す
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp

pur,exa = train["SalePrice"].copy(),train.drop("SalePrice",axis=1).copy()
pur = (pur - pur.mean())/pur.std()

p_value = []
mean_box = []
for i in range(0,500):
    kf = KFold(n_splits=3,random_state=i,shuffle=True)
    for tr ,te in kf.split(exa,pur,groups=None):
        X_train, y_train = exa.iloc[tr], pur.iloc[tr]
        X_test, y_test = exa.iloc[te], pur.iloc[te]
        p_value.append(ks_2samp(y_train,y_test)[1])

    mean_value = np.mean(p_value)
    mean_box.append(mean_value)
clear_output()
print(np.argmin(mean_box))
print(np.min(mean_box))
del pur,exa,kf,X_train, X_test, y_train, y_test
# ひとまずLGMRegressorを最適化する
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp

k_train = train.copy()
k_train = train_only_encoding(k_train)
pur,exa = k_train["SalePrice"].copy(),k_train.drop("SalePrice",axis=1).copy()
pur = (pur - pur.mean())/pur.std()
final_result = []
kf = KFold(n_splits=3,random_state=9,shuffle=True)
for i in range(20,100):
    result = []
    for tr ,te in kf.split(pur):
        X_train, y_train = exa.iloc[tr], pur.iloc[tr]
        X_test, y_test = exa.iloc[te], pur.iloc[te]
        model =lgb.LGBMRegressor(random_state=0,boosting_type="gbdt",objective="regression",metric="rmse",num_boost_round=i)
        model.fit(X_train,y_train)
        pre = model.predict(X_test)
        result.append(mean_squared_error(y_test,pre))
    clear_output()
    final_result.append(np.mean(result))
    
from matplotlib import pyplot as plt
x,y = np.arange(20,100),final_result
plt.plot(x,y)
plt.ylabel("RMSE")
plt.xlabel("num_boost_round")
plt.show()

del k_train,pur,exa,kf,X_train, X_test, y_train, y_test,model,pre,result,x,y
"""
[0.11009861503409565, 0.11913828267732557, 0.11980643889955646]
0.1163477788703259
"""
# 以上からCV評価は以下を用いる
# ひとまずLGMRegressorを最適化する
# 実際、結果を見ても安定したCV値が得られている
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp

pur,exa = train["SalePrice"].copy(),train.drop("SalePrice",axis=1).copy()
exa = train_only_encoding(exa)
pur = (pur - pur.mean())/pur.std()
kf = KFold(n_splits=3,random_state=9,shuffle=True)
result = []
for tr ,te in kf.split(pur):
    X_train, y_train = exa.iloc[tr], pur.iloc[tr]
    X_test, y_test = exa.iloc[te], pur.iloc[te]
    model =lgb.LGBMRegressor(random_state=0,boosting_type="gbdt",objective="regression",metric="rmse",num_boost_round=50)
    model.fit(X_train,y_train)
    pre = model.predict(X_test)
    result.append(mean_squared_error(y_test,pre))
clear_output()
print(result)
print(np.mean(result))

del pur,exa,kf,X_train, X_test, y_train, y_test,model,pre,result
0.11742963929941917
pd.set_option('display.max_columns', 50)
train.head(1)
# MSSubClass
"""
        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES
"""
for i in [train,test]:
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-STORY 1946 & NEWER ALL STYLES" if x==20 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-STORY 1945 & OLDER" if x==30 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-STORY W/FINISHED ATTIC ALL AGES" if x==40 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-1/2 STORY - UNFINISHED ALL AGES" if x==45 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-1/2 STORY FINISHED ALL AGES" if x==50 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"2-STORY 1946 & NEWER" if x==60 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"2-STORY 1945 & OLDER" if x==70 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"2-1/2 STORY ALL AGES" if x==75 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"SPLIT OR MULTI-LEVEL" if x==80 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"SPLIT FOYER" if x==85 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"DUPLEX - ALL STYLES AND AGES" if x==90 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-STORY PUD (Planned Unit Development) - 1946 & NEWER" if x==120 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"1-1/2 STORY PUD - ALL AGES" if x==150 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"2-STORY PUD - 1946 & NEWER" if x==160 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"PUD - MULTILEVEL - INCL SPLIT LEV/FOYER" if x==180 else x)
    i["MSSubClass"] = i["MSSubClass"].apply(lambda x:"2 FAMILY CONVERSION - ALL STYLES AND AGES" if x==190 else x)
del i






pur,exa = train["SalePrice"].copy(),train.drop("SalePrice",axis=1).copy()
mean = pur.mean()
std = pur.std()
pur = (pur - mean) / std
model =lgb.LGBMRegressor(random_state=0,boosting_type="gbdt",objective="regression",metric="rmse",num_boost_round=100)
model.fit(exa,pur)
pre = model.predict(test)
pre = pre*std + mean
pre = np.exp(pre)

submission = pd.DataFrame({
    "Id": test_id,
    "SalePrice": pre
})
submission.to_csv('submission.csv', index=False)
del pur,exa,model,pre
submission.head()

import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
from matplotlib import pyplot as plt
all_data = pd.concat((train.drop("SalePrice",axis=1),test)).copy()
sign = all_data["PoolArea"].copy()
sign = all_data[all_data["PoolArea"]!=0]["PoolArea"]
print(sign.min())
sign = np.log(sign)
sign = np.sqrt(sign)


sns.distplot(sign,fit = norm)
(mu, sigma) = norm.fit(sign)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('Distribution')
fig = plt.figure()
res = stats.probplot(sign, plot=plt)
plt.show()
print(stats.shapiro(sign))
del sign,all_data,mu,sigma,res