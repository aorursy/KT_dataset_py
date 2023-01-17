import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
train0=pd.read_csv('../input/train.csv')
test0=pd.read_csv('../input/test.csv')
train0=train0[(train0.GrLivArea<=4000)|(train0["1stFlrSF"]<3000)|(train0.TotRmsAbvGrd==14)]
train_y=train0[["Id","SalePrice"]]
df_train=train0.drop("SalePrice",axis=1)
df_test=test0
df_train["train"]=True
df_test["train"]=False
df_all=pd.concat([df_train,df_test])
###overall information of categorical variables(查看分类变量的总体情况)
df_all["MSSubClass"]=df_all["MSSubClass"].astype("object")
df_all["MoSold"]=df_all["MoSold"].astype("object")
obj_col=df_all.select_dtypes(include = ["object"]).columns
### drop the columns of the dataset with less information(移除不包含有用信息量的列)
for element in obj_col:
    print(df_all[element].value_counts())
object_drop=['Utilities','Condition1','Condition2','Heating','Street','RoofMatl']
for element in object_drop:
    df_all.drop(element,axis=1,inplace=True)
num_col=df_all.dtypes[df_all.dtypes!='object'].index.tolist()
num_col.remove('Id')
num_col.remove("train")
import seaborn as sns
sns.heatmap(train0[num_col].corr(method='pearson'))
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
print("Skewness: %f" % train0['SalePrice'].skew())
print("Kurtosis: %f" % train0['SalePrice'].kurt())
train0['SalePrice'].plot.hist(bins=100)
train_y["Log1pSalePrice"] = np.log1p(train_y["SalePrice"])
check_null=['BsmtUnfSF','BsmtFinSF2','BsmtFinSF1','TotalBsmtSF']
df_all[check_null].isnull().sum()
for col in check_null:
    df_all[col]=df_all[col].fillna(df_all[col].median())
def transfer(df):
    bsmt_col=['BsmtUnfSF','BsmtFinSF2','BsmtFinSF1']
    year=['YearBuilt','YrSold','YearRemodAdd']
    for col in bsmt_col:
        df['bsmt_unf_percentage']=df[bsmt_col[0]]/df['TotalBsmtSF']
        df['bsemt_fin_percentage1']=df[bsmt_col[2]]/df['TotalBsmtSF']
        df['bsemt_fin_percentage2']=df[bsmt_col[1]]/df['TotalBsmtSF']
        df['year_built_sold']=df['YrSold']-df['YearBuilt']
        df['year_built_remodel']=df['YearRemodAdd']-df['YearBuilt']
    year.remove("YearBuilt")
    df1=df.drop(bsmt_col+year,axis=1)
    return df1
bsmt_none=['TotalBsmtSF','bsmt_unf_percentage','bsemt_fin_percentage1','bsemt_fin_percentage2']
df_all=transfer(df_all)
df_all["TotalArea"]=df_all["1stFlrSF"]+df_all["2ndFlrSF"]
df_all["LowQualP"]=(df_all["LowQualFinSF"]/df_all["TotalArea"])*100
negative_year_col=["year_built_remodel","year_built_sold"]
for col in negative_year_col:
    df_all[col]=df_all[col].apply(lambda x:df_all[col].mode()[0] if x<0 else x)
Area_col=["LowQualFinSF"]
df_all.drop(Area_col,axis=1,inplace=True)
numerical_features = df_all.select_dtypes(exclude = ["object"]).columns.tolist()
Numerical_Features=[element for element in numerical_features if element not in ['Id','train']]
df_all[Numerical_Features].skew()
for column in Numerical_Features:
    df_all[column]=np.log1p(df_all[column])
df_all[Numerical_Features].skew()
train_y["Log1pSalePrice"].skew()
def findingnull(df):
    dic={}
    tc=df.columns.tolist()
    for column in tc:
        a=df[column].isnull().sum()
        if a>0:
            dic[column]=a
    return dic     
n1=findingnull(df_all)
sorted(n1.items(), key=lambda d: d[1])
df_all['MasVnrArea']=df_all['MasVnrArea'].fillna(train0['MasVnrArea'].median())
df_all['Electrical'] = df_all['Electrical'].fillna(train0['Electrical'].mode()[0])
df_all['MasVnrType'].fillna('None',inplace=True)
### change categorical variables to numerical variables（分类变量变为数字变量）
#replaces categorical values by mapping numeric values
qual_mapping={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}
bsmt_mapping={'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1}
functional_mapping={'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0}
garagetype_mapping={'2Types':6,'Attchd':5,'Basment':4,'BuiltIn':3,'CarPort':2,'Detchd':1}
garagefinish_mapping={'Fin':3,'RFn':2,'Unf':1}
PoolQC_mapping={'Ex':4,'Gd':3,'TA':2,'Fa':1}
YN_mapping={'Y':1,'N':0}
BsmtExposure_mapping={'Gd':4,'Av':3,'Mn':2,'No':1}
Fence_mapping={"GdPrv":4,"MnPrv":3,"GdWo":2,"MnWw":1}
clist=['FireplaceQu','GarageQual','GarageCond','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual']
clist2=['BsmtFinType1','BsmtFinType2']
for column in clist:
    df_all[column]=df_all[column].map(qual_mapping)
for column in clist2:
    df_all[column]=df_all[column].map(bsmt_mapping)
df_all['CentralAir']=df_all['CentralAir'].map(YN_mapping)
df_all['Functional']=df_all['Functional'].map(functional_mapping)
df_all['GarageType']=df_all['GarageType'].map(garagetype_mapping)
df_all['GarageFinish']=df_all['GarageFinish'].map(garagefinish_mapping)
df_all['BsmtExposure']=df_all['BsmtExposure'].map(BsmtExposure_mapping)
df_all["PoolQC"]=df_all["PoolQC"].map(PoolQC_mapping)
df_all["Fence"]=df_all["Fence"].map(Fence_mapping)
df_all['Functional']=df_all['Functional'].fillna('Typ')
nullist=['PoolQC','FireplaceQu','BsmtQual','BsmtExposure','GarageFinish','GarageType','BsmtFinType1','BsmtFinType2','BsmtCond','GarageQual','GarageCond','Fence']
for column in nullist:
    df_all[column].fillna(0,inplace=True)
null=pd.DataFrame(df_all.select_dtypes(exclude=["object"]).isnull().sum(),columns=["freq"])
gb_garage_year = df_all['GarageYrBlt'].groupby(df_all['YearBuilt'])
for key,group in gb_garage_year:
    index = df_all['GarageYrBlt'].isnull() & (df_all['YearBuilt'] == key)
    df_all.loc[index,'GarageYrBlt'] = group.median()
gb_lf_ng = df_all['LotFrontage'].groupby(df_all['Neighborhood'])
for key,group in gb_garage_year:
    index = df_all['LotFrontage'].isnull() & (df_all['Neighborhood'] == key)
    df_all.loc[index,'LotFrontage'] = group.median()
df_all["GarageYrBlt"]=df_all["GarageYrBlt"].fillna(df_all["GarageYrBlt"].mode()[0])
df_all["LotFrontage"]=df_all["LotFrontage"].fillna(df_all["LotFrontage"].median())
### replace the missing values of the columns in tlist with the values with highest frequency(用最频繁出现的数替代缺失值)
tlist=df_all.columns.tolist()
for column in tlist:
    if df_all[column].isnull().sum()>0:
        df_all[column] = df_all[column].fillna(df_all[column].mode()[0])
mapc=clist+clist2
add=['Log1pSalePrice','Fence','PoolQC']
for col in add:
    mapc.append(col)
col_c=['FireplaceQu','GarageQual','GarageCond','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','BsmtFinType1','BsmtFinType2','Fence','PoolQC']
for column in col_c:
    df_all[column]=np.log1p(df_all[column])
df_train=df_all[df_all["train"]==True]
df_test=df_all[df_all["train"]==False]

df_train=pd.merge(df_train,train_y,on="Id")
corrt2=df_train.corr(method='pearson').Log1pSalePrice
corrt2=pd.DataFrame(corrt2).drop('Log1pSalePrice',axis=0)
a=['Id','SalePrice','train']
for element in a:
    corrt2.drop(element,axis=0,inplace=True)
corrt2.sort_values(by='Log1pSalePrice',ascending=False)
sns.lmplot(x="OverallQual",y="SalePrice",data=train0)
sns.lmplot(x="GrLivArea",y="SalePrice",data=train0)
sns.lmplot(x="1stFlrSF",y="SalePrice",data=train0)
sns.lmplot(x="GarageYrBlt",y="SalePrice",data=train0)
sns.lmplot(x="TotRmsAbvGrd",y="SalePrice",data=train0)
sns.lmplot(x="year_built_sold",y="Log1pSalePrice",data=df_train)
sns.lmplot(y="SalePrice",x="LotFrontage",data=train0)
sns.lmplot(y="GarageYrBlt",x="YearBuilt",data=train0)
droptarget2=corrt2[abs(corrt2['Log1pSalePrice'])<0.001].index.tolist()
df_train[['MiscFeature','LandContour','Alley']].describe()
more_missing_col=['MiscFeature','LandContour','Alley']

for col in more_missing_col:
    df_all[col]=df_all[col].fillna("None")
droplist=['LandContour']+droptarget2
adddrop=['GarageCars','TotRmsAbvGrd']
for element in adddrop:
    if element not in droplist:
        droplist.append(element)
for column in droplist:
    df_all.drop(column,axis=1,inplace=True)
### numerical variables to dummy variables(数字变量转化为哑变量)
Numerical_Features = df_all.select_dtypes(exclude = ["object"]).columns
categorical_features = df_all.select_dtypes(include = ["object"]).columns
df_cat1 = df_all[categorical_features]
df_num = df_all[Numerical_Features]

df_cat2 = pd.get_dummies(df_cat1)
df_all=pd.concat([df_cat2,df_num],axis=1)
df_train=df_all[df_all["train"]==True]
df_test=df_all[df_all["train"]==False]
df_train=pd.merge(df_train,train_y,on="Id")
df_train=df_train.drop(["SalePrice"],axis=1)
f=[col for col in df_train.columns if col not in ["Log1pSalePrice","Id","train"]]
features=df_train[f]
target=df_train["Log1pSalePrice"]
test=df_test[f]
ID=df_test["Id"]
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,r2_score, make_scorer
rfr5 = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
#def train_and_cross_val(df,model):
#    features=[col for col in df.columns if col not in ["Log1pSalePrice","Id","train"]]
#   f=df[features]
#    target=df_train["Log1pSalePrice"]
#    variance_values=[]
#    mse_values=[]
#    kf=KFold(n_splits=10,shuffle=True,random_state=3)
#    for train_index,test_index in kf.split(f):
#        x_train,x_test=f.iloc[train_index],f.iloc[test_index]
#        y_train,y_test=f.iloc[train_index],f.iloc[test_index]
#        model.fit(x_train,y_train)
#        prebiction=model.predict(x_test)
#        mse=mean_squared_error(prediction,y_test)
#        variance=np.var(prediction)
#        variance_values.append(variance)
#       mse_values.append(mse)
#    avg_mse=np.sum(mse_values)/len(mse_values)
#    avg_var=np.sum(variance_values)/len(variance_values)
#    return(avg_mse,avg_var)
#cross_val_score(rfr5, Trian0[target], y_train0, cv=5).mean()
cvs=cross_val_score(rfr5, features, target, cv=10).mean()
print(f"random forest model, the cvs={cvs} ")
rfr5.fit(features,target)
pred5=rfr5.predict(test)
my_submission = pd.DataFrame({'Id': ID, 'SalePrice': np.expm1(pred5)})
my_submission.to_csv('rdf.csv', sep='\t',index=False)
lassocv=linear_model.LassoCV(alphas=[0.0001,0.0002,0.0003]+list(np.arange(0.01,1,0.01)),cv=10,max_iter=100000,n_jobs=-1,random_state=10)
lassocv.fit(features,target)
train_pred=np.expm1(lassocv.predict(features))
train_target=np.expm1(target)
sqrt(mean_squared_error(train_pred,train_target))
r2_score(train_pred,train_target)
lasso_pred=lassocv.predict(test)
cv_score = cross_val_score(lassocv,features,target, cv=10,n_jobs=-1)
print(f"The cv score is {cv_score.mean()}")
my_submission = pd.DataFrame({'Id': ID, 'SalePrice': np.expm1(lasso_pred)})
my_submission.to_csv('lasso.csv', sep='\t',index=False)
from sklearn.linear_model import ElasticNetCV
reg4 = ElasticNetCV(cv=10, fit_intercept=True, normalize=False)
reg4.fit(features,target)
pred7 = reg4.predict(test)
cross_val_score(reg4, features, target, cv=10).mean()
my_submission = pd.DataFrame({'Id': ID, 'SalePrice': np.expm1(pred7)})
my_submission.to_csv('net.csv', sep='\t',index=False)
from sklearn.metrics import mean_squared_error
net_train_pred=reg4.predict(Trian0[target])
mean_squared_error(np.expm1(y_train0),np.expm1(net_train_pred))**(1/2)
