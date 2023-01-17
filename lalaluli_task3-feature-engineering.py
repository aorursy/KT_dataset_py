#filter warningrr
import warnings
warnings.filterwarnings('ignore')

#for dataframe
import pandas as pd
import numpy as np

#display all the columns in pandas
pd.set_option('display.max_columns', None)

#data visualization
import matplotlib.pyplot as plt
import seaborn as sns

import math
import scipy.stats as st
import re
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets, ensemble
from sklearn.preprocessing import LabelEncoder
#import file
trainfile = r"E:\kaggle比赛\House Price\data\train.csv"
testfile = r"E:\kaggle比赛\House Price\data\test.csv"
train = pd.read_csv(trainfile)
test = pd.read_csv(testfile)

#print size
print("train set shape:", train.shape)
print("test set shape:", test.shape)
#first and last 5 rows
train.head().append(train.tail())
# change data type
var = ["MSSubClass","YrSold","MoSold"]
train[var] = train[var].astype(str)
test[var] = test[var].astype(str)
'''
This method will check the dtypes in dataframe. 

It will print categorical and numerical variable names.

It returns the name for categorical and numerical variable names as a list.
'''
def data_type(df):
    
    #mask and names
    mask = df.dtypes == object
    cats = list(df.columns[mask])
    nums = list(df.columns[~mask])
    
    #print result
#     print("Categorical variables are:",cats)
#     print()
#     print("Numerical variables are:", nums)
    
    #return
    return [cats,nums]
#train data
train_cats, train_nums = data_type(train)
print(train_cats)
print()
print(train_nums)
train["train"] = 1
test["train"] = 0 
data = pd.concat([train,test],ignore_index = True)
data.tail(10)
def save(df, name):
    df.to_csv(name,index = False)
def refresh_data():
    data = pd.concat([train,test],ignore_index = True)
# 查看全部缺失值
missing = data.isnull().sum()
missing = missing[missing>0].sort_values()
missing
y = train.SalePrice
#fill the NA
mask = data.PoolArea == 0
row = data[mask].index
data.loc[row,"PoolQC"] = "None"
# 在test集中仍然有缺失值
mask = data.PoolQC.isnull()
data[mask]
data.groupby("PoolQC")["PoolArea"].describe()
data.PoolQC = data.PoolQC.fillna("Gd")
def boxplot_comparison(col):
    #data
    ori = train[col]
    new = data[data.train == 1][col]
    
    #graphs
    fig,ax = plt.subplots(1,2,figsize = (10,5))
    sns.boxplot(ori, y,ax = ax[0])
    sns.boxplot(new, y,ax = ax[1])
boxplot_comparison("PoolQC")
data.PoolQC.value_counts()
# 创建哑变量并删除PoolQC
del data["PoolQC"]
data["HasPool"] = data.PoolArea.apply(lambda x: 1 if x >0 else 0)
#fill the none
temp = data[data.MiscVal == 0]
mask = data[(data.MiscVal ==0) & (data.MiscFeature.isnull()== True)]
row = mask.index
data.loc[row,"MiscFeature"] = "None"

# 检查是否还有空缺
remain = data.MiscFeature.isnull()
data[remain]
# 空缺值当做Gar2,在极端异常值范围内
data.groupby("MiscFeature")["MiscVal"].describe()
data.MiscFeature.value_counts()
# 箱线图
data.MiscFeature = data.MiscFeature.fillna("Gar2")
boxplot_comparison("MiscFeature")
# 特例
data[(data.MiscVal == 0) & (data.MiscFeature != "None")]
# 填充并且绘图
data.Alley = data.Alley.fillna("Missing")
boxplot_comparison("Alley")
data.Fence = data.Fence.fillna("MISSING")
boxplot_comparison("Fence")
# 如果Fireplaces为0， 则无Fireplace, FireplaceQu为None
mask = data.Fireplaces == 0
i = data[mask].index
data.loc[i,"FireplaceQu"] = "None"

boxplot_comparison("FireplaceQu")
data.FireplaceQu.value_counts()
from scipy import stats
# 画一下histgram
var = "LotFrontage"
data[var].hist()
data[data[var]>300]
# 用中位数填充
data[var] = data[var].fillna(data[var].median())
def hist_comparison(var):
    ori = train[var]
    new = data[data.train==1][var]
    
    fig,ax = plt.subplots(1,2,figsize = (8,4))
    ori.hist(ax = ax[0])
    new.hist(ax = ax[1])
# 比较转换结果
hist_comparison(var)
# 缺少量值，用众数填充
var = ['Electrical','Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType','Utilities', 'Functional']
for x in var:
    data[x] = data[x].fillna(data[x].mode()[0])
# 如果totalbsmt为0，则其他bsmt相关则为none
cols = ['BsmtQual','BsmtCond',"BsmtExposure","BsmtFinType1","BsmtFinType2"]
row = data[data.TotalBsmtSF==0].index
data.loc[row,cols] = "None"
#检查空缺值
data[cols].isnull().sum()
data[data.BsmtFinType2.isnull()]
# row 332, 1124+479+1603 = 3206，BsmtFinSF还未完成
data.loc[332,"BsmtFinType2"] = "Unf"

# 2120全空
cats = ['BsmtQual','BsmtCond',"BsmtExposure","BsmtFinType1","BsmtFinType2"]
nums = ["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","BsmtUnfSF","TotalBsmtSF"]
data.loc[2120,cols] = "None"
data.loc[2120,nums] = 0

#检查是否符合规律
mask = data.BsmtFinSF1+ data.BsmtFinSF2 == data.TotalBsmtSF - data.BsmtUnfSF
print(sum(mask) == data.shape[0])
#BsmtQual
data[data.BsmtQual.isnull()]
# 按众数填充
miss_i = [2217,2218]
data.loc[2217,"BsmtQual"] = data[(data.BsmtFinType1 == "Unf")]["BsmtQual"].mode()[0]
data.loc[2218,"BsmtQual"] = data[(data.BsmtCond == "TA") & (data.BsmtFinType1 == "Unf")]["BsmtQual"].mode()[0]

# 检查结果
data.loc[miss_i][cols]
#检查空缺值
temp = data[data.BsmtCond.isnull()]
temp_id = temp.index
data.loc[temp_id]
#填充并且检查
temp_id = [2040,2185,2524]
data.loc[temp_id, "BsmtCond"] = data.BsmtCond.mode()[0]
data.loc[temp_id,cols]
# 检查情况
data[data.BsmtExposure.isnull()][cols]
# 填充，检查
temp_index = [948,1487,2348]
data.BsmtExposure = data.BsmtExposure.fillna(data[(data.BsmtQual == "Gd") & (data.BsmtFinType1 == "Unf")]["BsmtExposure"].mode()[0])
data.loc[temp_index,cols]
# 检查发现，相关bsmt的内容完全没有
data[data.BsmtFullBath.isnull()]
# 填充，检查
miss_i = [2120,2188]
data.BsmtFullBath = data.BsmtFullBath.fillna(0)
data.BsmtHalfBath = data.BsmtHalfBath.fillna(0)
data.loc[miss_i,["BsmtFullBath","BsmtHalfBath"]]
# 添加新变量
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +
                               data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
data["has2ndfloor"] = data["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
data["hasBsmtSF"] = data["TotalBsmtSF"].apply(lambda x: 1 if x >0 else 0)

#删除旧变量
del data["TotalBsmtSF"]
#GarageCars ==0 and GarageArea ==0认为没有车位，填NA
cols = ["GarageCond","GarageQual","GarageFinish","GarageType"]
mask = (data.GarageCars ==0) & (data.GarageArea == 0)
row = data[mask].index
data.loc[row,cols] = "None"

# 生成哑变量
data["has_Garage"] = data.GarageYrBlt.apply(lambda x: 1 if x> 0 else 0)
data[data.GarageYrBlt>2200]
i = data[data.GarageYrBlt>2200].index
data.loc[i,"GarageYrBlt"] = 2007
# GarageYrBlt和其他变量有很强的的相关
correlation = train.corr()
temp = correlation['GarageYrBlt'].sort_values(ascending = False)
temp_name = temp.index
plt.figure(figsize = (8,8))
sns.barplot(x = temp, y = temp_name,orient = 'h',)
# fill YrBlt with 0
data["GarageYrBlt"] = data["GarageYrBlt"].fillna(0)
# 其他关于Garage的空值
cols = ["GarageType","GarageYrBlt","GarageCars","GarageArea","GarageQual","GarageCond"]
data[data.GarageCond.isnull()][cols]
# 填充，针对GarageType == Detchd
# row 2126
var = ["GarageYrBlt","GarageQual","GarageCond"]
for x in var:
    val = data[(data.GarageType == "Detchd")&(data.GarageCars==1)][x].mode()[0]
    data.loc[2126,x] = val

# row 2576
for x in cols:
    data.loc[2576,x] = data[data.GarageType=="Detchd"][x].mode()[0]
    
#检查填充
miss_i = [2126,2576]
data.loc[miss_i][cols]
# 缺GarageFinish
data[data.GarageFinish.isnull()]
#填充，检查
miss_i = [2126,2576]
val = data[(data.GarageType == "Detchd") & (data.GarageQual == "TA")
           & (data.GarageCars == 1|2)]["GarageFinish"].mode()[0]

data.GarageFinish = data.GarageFinish.fillna(val)
data.loc[miss_i]
del data["GarageYrBlt"]
temp = data[data.MasVnrArea.isnull()]
temp_id = temp.index
temp.head()
# 根据area=0, type = none
data.MasVnrArea = data.MasVnrArea.fillna(0)
data.loc[temp_id, "MasVnrType"] = "None"
# 检查结果
data[data.MasVnrType.isnull()]
# 再次填充，检查结果
val = data[data.MasVnrArea>0]["MasVnrType"].mode()[0]
data.MasVnrType = data.MasVnrType.fillna(val)
data[data.MSZoning.isnull()]
#缺失内容全在test set
var = "MSZoning"
miss_i = [1915,2216,2250,2904]
data.loc[1915,var] = data[data.MSSubClass == "30"][var].mode()[0]
data.loc[2216,var] = data[data.MSSubClass == "20"][var].mode()[0]
data.loc[2250,var] = data[data.MSSubClass == "70"][var].mode()[0]
data.loc[2904,var] = data[data.MSSubClass == "20"][var].mode()[0]

#填充后检查
data.loc[miss_i]
# 检查剩余是否有空缺
missing1 = data.isnull().sum()
missing1 = missing1[missing1>0]
missing1.sort_values()
th = 0.99
var = []
for col in data.columns:
    
    val = list(data[col].value_counts().values)[0]
    name = list(data[col].value_counts().index)[0]

    if (val/2919 > th):
        var.append(col)
        print(col,"出现频率最高的是",name)
        print("占比",val/2919)
        print(data[col].value_counts())
        print("--")

del data["Id"]
del data["Utilities"]
#del data["Street"]
# Central Air改为1,0,
replace = {}
replace["CentralAir"] = {'N':0, 'Y':1}
data.replace(replace, inplace = True)
sns.barplot(x = data[data.train ==1]["CentralAir"], y = y)
data["Remod"] = data["YearBuilt"] == data["YearRemodAdd"]
data["Remod"] = data["Remod"].apply(lambda x:1 if x==True else 0)
data["Age"] = data["YrSold"].astype(int) - data["YearRemodAdd"]
sns.regplot(x = data[data.train ==1]["Age"],y = data[data.train==1]["SalePrice"])
data[["Age","SalePrice"]].corr()
# 是否是新房
data["IsNew"] = data["YrSold"].astype(int) == data["YearBuilt"]
data["IsNew"] = data["IsNew"].apply(lambda x: 1 if x == True else 0)

data["IsNew"].value_counts()
sns.boxplot(x = data[data.train==1]["IsNew"],y = y)
n = ['YearRemodAdd', 'GarageArea', 'GarageCond']
data.drop(columns = n ,inplace = True)
data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +
                              data['EnclosedPorch'] + data['ScreenPorch'] +
                              data['WoodDeckSF'])
save(data,"data_tree.csv")
# 重新给categorical, numerical变量分类
df = data[data.train==1]
cats, nums = data_type(df)
def irq_cal(col,df):
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    return iqr

def ex_check(col, df, s):
    '''
    col: column name
    df: dataframe
    s: scale
    '''
    scale = s
    iqr = irq_cal(col,df)
    val_min = df[col].quantile(0.25) - scale * iqr
    val_max = df[col].quantile(0.75) + scale * iqr
    
    return val_min, val_max
for var in nums:
    vmin,vmax = ex_check(var,df,3.0)
    print(var,":")
    print("最小估计值是", vmin, ", ", df[df[var]<=vmin].shape[0], "个值低于最小估计值")
    print("最大估计值是", vmax, ",",  df[df[var]>=vmax].shape[0], "个值高于最大估计值")
    print("--------------------------------------------------------------")
var_all = ["LotFrontage","LotArea","MasVnrArea","BsmtFinSF1",
       "1stFlrSF","GrLivArea","WoodDeckSF","OpenPorchSF",
      "SalePrice"]

fig,ax = plt.subplots(4, 2, figsize=(20,25))

for n in range(0,8):
    sns.regplot(var_all[n],y,df,ax = ax[int(n/2),n%2])
# 峰度和偏度
sta = pd.DataFrame(df[var_all].skew(), columns = ["skew"])
sta["kurt"] = df[var_all].kurt()
sta
var = "LotFrontage"
data[var] = np.log(data[var])
hist_comparison(var)
def cap(col_name, df):
    miu = df[col_name].mean()
    max_cap = miu+3*df[col_name].std()
    min_cap = miu-3*df[col_name].std()
    return min_cap, max_cap

def capmax_replace(col_name,df):
    _,cmax = cap(col_name, df)
    i = df[df[col_name] > cmax].index
    df.loc[i,col_name] = cmax
# 修改
var = "LotArea"
data["LotArea"] = np.log(data["LotArea"])
print("现在的偏度和峰度")
print(data[var].skew(),data[var].kurt())
var = "MasVnrArea"
#data["HasMasVnrArea"] = data[var].apply(lambda x: 1 if x > 0 else 0)
x = pd.DataFrame(df[var])
data[var] = np.log1p(data[var])
hist_comparison(var)
var = "BsmtFinSF1"
#data["HasBsmtFinSF1"] = data[var].apply(lambda x: 1 if x > 0 else 0)
x = pd.DataFrame(df[var])
#capmax_replace(var,x)
data[var] = np.log1p(data[var])
hist_comparison(var)
var = "1stFlrSF"
x = pd.DataFrame(df[var])
data[var] = np.log1p(data[var])
hist_comparison(var)
var = "GrLivArea"
row = data[(data.train == 1) & (data.GrLivArea > 4000) & (data.SalePrice < 300000)].index
data.drop(row,axis = 0 ,inplace = True)
data[var] = np.log1p(data[var])
hist_comparison(var)
var = "WoodDeckSF"
#data["HasWoodDeck"] = data[var].apply(lambda x: 1 if x > 0 else 0)
data[var] = np.log1p(data[var])
hist_comparison(var)
var = "OpenPorchSF"
#data["HasOpenPorch"] = data[var].apply(lambda x: 1 if x > 0 else 0)
data[var] = np.log1p(data[var])
hist_comparison(var)
df = data[data.train ==1]
y = df["SalePrice"]
fig,ax = plt.subplots(4, 2, figsize=(20,25))

for n in range(0,8):
    sns.regplot(var_all[n],y,df,ax = ax[int(n/2),n%2])
_,nums = data_type(data)
skewness = []
for x in nums:
    skewness.append(data[x].skew())
sk = pd.DataFrame(data = skewness,index = nums,columns = ["skew"])
sk[abs(sk["skew"])>.8]
var_all = sk[abs(sk["skew"])>.8].index
var_drop = ["LotFrontage","CentralAir","HasPool","SalePrice","hasBsmtSF","has_Garage","IsNew"]
var_tran = list(set(var_all)-set(var_drop))
var_tran
for x in var_tran:
    data[x] = np.log1p(data[x])
_,nums = data_type(data)
skewness = []
for x in nums:
    skewness.append(data[x].skew())
sk = pd.DataFrame(data = skewness,index = nums,columns = ["skew"])
sk[abs(sk["skew"])>0.5]
# 给YearBuilt分箱
bins = [(i*10+1870) for i in range(16)]
data["YearBuilt2"] = pd.cut(data["YearBuilt"],bins,labels = False)
plt.hist(data["YearBuilt2"])
del data["YearBuilt"]
# # GarageYrBlt和其他变量有很强的的相关
# df = data[data.train == 1]
# correlation = df.corr()
# temp = correlation["SalePrice"].sort_values(ascending = False)
# temp_name = temp.index
# plt.figure(figsize = (20,20))
# sns.barplot(x = temp, y = temp_name,orient = 'h')
save(data,"data_reg.csv")
# 创建字典，把oridnal categorical variable换为数字
replace = {}

# 依次写入内容
replace["ExterQual"] = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1}
replace["ExterCond"] = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1}

#bsmt
replace["BsmtQual"] = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
replace["BsmtCond"] = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
replace["BsmtExposure"] = {'Gd':4,'Av':3, 'Mn':2, 'No':1, 'None':0}
replace["BsmtFinType1"] = {'GLQ':6, "ALQ":5, 'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0}
replace["BsmtFinType2"] = {'GLQ':6, 'ALQ':5, 'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0}

#garage
replace["GarageFinish"] = {'Fin': 3,  'RFn':2, 'Unf':1,'None':0}
replace["GarageQual"] = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}

#other
replace["HeatingQC"] = {'Ex':5, 'Gd':4, 'TA':3,'Fa':2, 'Po':1}
replace["KitchenQual"] = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1}
replace["FireplaceQu"] = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0}
#replace["PoolQC"] = {"Ex":4,"Gd":3,"TA":2,"Fa":1,"None":0}
replace["LotShape"] = {'IR3':0, 'IR2':1, 'IR1':2, 'Reg':3}
replace["Functional"] = {'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7}
#replace["Street"] = {'Grvl':0, 'Pave':1}
replace["PavedDrive"] = {'N': 0, 'P':1, 'Y':2}
replace["LandSlope"] = {"Gtl":3,"Mod":2,"Sev":1}
replace["MasVnrType"] = {"None":0, "BrkCmn":1, "BrkFace":2, "Stone":3}

data.replace(replace, inplace = True)
key = list(replace.keys())
correlation = data[key].corr()
correlation[correlation>0.5]
sns.heatmap(correlation,square = True, vmax = 1)
del data["BsmtFinType2"]
#data["BsmtFinType2"].hist()
print(data.YrSold.value_counts())
print(data.groupby("YrSold")["MoSold"].describe())
data["Recession"] = data.YrSold.apply(lambda x: 1 if x == "2008" else 0)
print(data.Recession.value_counts())
sns.boxplot(x = data["Recession"], y = np.log(data["SalePrice"]))
# ANOVA检测: 两组平均值是否相同
year_2008 = np.log(data[(data.Recession==1) & (data.train ==1)]["SalePrice"])
year_other =  np.log(data[(data.Recession==0) & (data.train ==1)]["SalePrice"])
fvalue, pvalue = stats.f_oneway(year_2008,year_other)
print(fvalue,pvalue)
# n-way ANOVA: 2006 - 2010 年各组平均值是否相同
year_2006 = np.log(data[(data.YrSold=="2006") & (data.train ==1)]["SalePrice"])
year_2007 = np.log(data[(data.YrSold=="2007") & (data.train ==1)]["SalePrice"])
year_2009 = np.log(data[(data.YrSold=="2009") & (data.train ==1)]["SalePrice"])
year_2010 = np.log(data[(data.YrSold=="2010") & (data.train ==1)]["SalePrice"])

fvalue, pvalue = stats.f_oneway(year_2006,year_2007,year_2008,year_2009, year_2010)
print(fvalue,pvalue)
# 数据偏度和峰度
print(year_2006.skew(),year_2006.kurt())
print(year_2007.skew(),year_2007.kurt())
print(year_2008.skew(),year_2008.kurt())
print(year_2009.skew(),year_2009.kurt())
print(year_2010.skew(),year_2010.kurt())
sns.distplot(year_2008,hist=False)
sns.distplot(year_2006,hist=False)
sns.distplot(year_2007,hist=False)
sns.distplot(year_2009,hist=False)
sns.distplot(year_2010,hist=False)
del data["Recession"]
data_tree = pd.get_dummies(data)
data_tree.shape
data_tree





# 设定x, y
train = data_tree[data_tree.train == 1]
names = list(set(data_tree.columns)-set(["SalePrice","train"])) # 除了y的变量
X,y = train[names],np.log(train["SalePrice"])

# 随机森林
rf = RandomForestRegressor(n_estimators = 100)
result = rf.fit(X, y)
result.score(X,y) # R-score
# 特征筛选
sel = SelectFromModel(rf).fit(X,y)
selected_feat= X.columns[(sel.get_support())]
print(len(selected_feat)) # 特征数量
print(selected_feat)      # 特征
# 按重要性将特征排序
importances = rf.feature_importances_
coef = list(importances[sel.get_support()])
select_fea = pd.DataFrame(data = coef, index = selected_feat, columns = ["coef"])
select_fea = select_fea.sort_values(by = "coef",ascending=False)

# 画图
fig,ax = plt.subplots(figsize = (20,10))
sns.barplot(x = select_fea.coef, y = select_fea.index,orient = 'h')
threshold = 10
drop_list = []
for x in data_tree:
    if (data_tree[x].nunique() == 2):
        counts = data_tree[x].value_counts()
        small = counts.values.min()
        if (small <= threshold):
            drop_list.append(x)
            print(x)
            print(counts)
            print("----------------------")
len(drop_list)
data_tree.drop(columns = drop_list,inplace = True)
print(data_tree.shape)
save(data_tree,"data_tree2.csv")
save(data,"data2.csv")
# dummy data
train = data_tree[data_tree.train == 1]
names = list(set(data_tree.columns)-set(["SalePrice","train"])) # 除了y的变量
X,y = train[names],np.log(train["SalePrice"])
#X,y = train[selected_feat],train["SalePrice"]

# set variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1)

params = {'n_estimators': 500,
          'learning_rate': 0.01,}

reg = ensemble.GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.01)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The Root of mean squared error (RMSE) on test set: {:.4f}".format(math.sqrt(mse)))

# test = data_tree[data_tree.train == 0]
test = data_tree[data_tree.train == 0]
test = test[names]

yhat = reg.predict(test)
yhat = np.exp(yhat)

prediction = pd.DataFrame(data = (test.index+1), columns = ["Id"])
prediction["SalePrice"] = yhat
prediction.SalePrice.hist()
prediction.to_csv("prediction.csv",index = False)

