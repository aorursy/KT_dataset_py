import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
%matplotlib inline
#读入数据
df_train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
submission=pd.read_csv('../input/sample_submission.csv')
#1.分析待预测值
df_train['SalePrice'].describe()
#histogram
sns.distplot(df_train['SalePrice'])
#偏离正常分布。
#有明显的正偏态。
#显示尖锐度。
#skewness(偏度)kurtosis(峰度)
print('skewness:%f'%df_train['SalePrice'].skew())
print('kurtosis:%f'%df_train['SalePrice'].kurt())
#2.关联分析
var='GrLivArea'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#大致呈线性增加，有俩个离群点
#scatter plot TotalBsmtSF/SalePrice
var='TotalBsmtSF'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#有点呈指数关系
#与类别标签的关联关系
var='OverallQual'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
f,ax=plt.subplots(figsize=(8,6))#设置图的大小
fig=sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ylim=0,ymax=800000)
#评价越高，房价越高
var='YearBuilt'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig=sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ylim=0,ymax=800000);
plt.xticks(rotation=90);
df_train.shape
#相关矩阵（热图）只是numerical之间的关联，类别标签之间没有关联
corrmat=df_train.corr()
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True);
#相关系数越高，他们越线性相关，他们之间可以只取一个值就可以了
#热图放大
k=10
cols=corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm=np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.0)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.1f',annot_kws={'size':8},yticklabels=cols.values,xticklabels=cols.values)
plt.show()
#GarageCars and GarageArea相关性很大
#TotalBsmtSF and 1stFlrSF
#TotRmsAbvGrd and GrLivArea
#与SalePrice相关性较强的sandian
sns.set()
cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(df_train[cols],size=2.5)
plt.show()
#3.缺失值处理
total=df_train.isnull().sum().sort_values(ascending=False)
percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['total','percent'])
missing_data.head(20)


#当缺失值大于15%时，我们删除它，假设它从未出现，因为如果重要为啥会缺失那么多，同时根据我们自己对问题理解，这些值也不是那么重要
#GarageX这类变量，他们肯定是同一组数据里同时缺失的值，由于Garage最重要的信息在GarageCars，同时考虑到我们只在谈论5%的缺失值，我们将删除GarageX
#Bsmt同理
#MasVnrArea and MasVnrType 我们认为他们不是很重要，而他们与 YearBuilt and OverallQual有很强的关联关系，所以删除不会丢失信息
#test['TotalBsmtSF'].value_counts()
test_total=test.isnull().sum().sort_values(ascending=False)
test_percent=(test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
test_missing_data=pd.concat([test_total,test_percent],axis=1,keys=['test_total','test_percent'])
test_missing_data.head(34)
df_train=df_train.drop((missing_data[missing_data['total']>1]).index,axis=1)
df_train=df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()
test=test.drop((missing_data[missing_data['total']>1].index),axis=1)

test['MSZoning'].fillna(test['MSZoning'].value_counts().index[0],inplace=True);
test['BsmtHalfBath'].fillna(0,inplace=True)
test['Utilities'].fillna('AllPub',inplace=True)
test['Functional'].fillna('Typ',inplace=True)
test['KitchenQual'].fillna('TA',inplace=True)
test['GarageCars'].fillna(2.0,inplace=True)
test['GarageArea'].fillna(test['GarageArea'].mean(),inplace=True)
test['Exterior1st'].fillna(test['Exterior1st'].value_counts().index[0],inplace=True)
test['Exterior2nd'].fillna(test['Exterior2nd'].value_counts().index[0],inplace=True)
test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean(),inplace=True)
test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean(),inplace=True)
test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean(),inplace=True)
test['BsmtFullBath'].fillna(0,inplace=True)
test['SaleType'].fillna('WD',inplace=True)
test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean(),inplace=True)

print(test.shape)
print(df_train.shape)
#4.离群值
#4.1 单变量分析
#这一块的主要任务是找到一个阈值 然后定义观察值是否为离群点 。为了实现它，我们需要先标准化数据
from sklearn.preprocessing import StandardScaler
saleprice_scaled=StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
print('out range (low) of the distribution:\n',low_range)
high_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('high range (low) of the distribution:\n',high_range)
#low range 相似，距离0不远，high range 区别较大，且距离0较远
#4.2 二元变量分析
var='GrLivArea'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#最右侧有两个点没有服从数据的趋势，他们可能是什么原因导致的呢？可能是土地类型。因此他们应该被当做离群点
#最上面两个点服从数据趋势，但是非常大，我们猜测可能是high range 里7.的那两个值
#删除两个离群点 1298 523
df_train.sort_values(by='GrLivArea',ascending=False)[:2]
df_train=df_train.drop(df_train[df_train['Id']==1299].index)
df_train=df_train.drop(df_train[df_train['Id']==524].index)
#bivariate analysis saleprice/totalBstmSF
var='TotalBsmtSF'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#最右边的三个点像离群点，但猜测不值得删除
#5. 获得核心
#‘SalePrice’符合统计假设，使得我们能够应用多元技术
#根据Hair等人研究，4个假设应该被测试
#5.1 Normality(正太分布):有一些统计测试依赖正态性。单元正太不能保证多元正太
#5.2 Homoscedasticity(同方差):假设独立变量多结果影响相同
#5.3 Linearity(线性):直观方式是画散点图观察图形是否线性，如果不是的话，就要做数据转换
#5.4 Absence of correlated errors:相关错误，一个错误与另一个错误相联系。这个经常发生在时间序列里，有些模式是时间相关的。如果发现，需要添加一个变量
#来解释你观察到的现象。这是最一般的解决相关错误的方法。

#Let's start
#5.1 查看正太分布
# Histogram-峰度和偏度
# Normal probability plot-数据分布应该沿着对角线就代表是正太分布
sns.distplot(df_train['SalePrice'],fit=norm);
fig=plt.figure()
res=stats.probplot(df_train['SalePrice'],plot=plt)
#观察到'SalePrice'不是正太的 有峰度且正偏
#这种情况下，log转换可以解决
Y_label=df_train['SalePrice']
Y_label.head()
#应用log转换
df_train['SalePrice']=np.log(df_train['SalePrice'])
#验证
sns.distplot(df_train['SalePrice'],fit=norm);
fig=plt.figure()
res=stats.probplot(df_train['SalePrice'],plot=plt)
#检查 GrLivArea
sns.distplot(df_train['GrLivArea'],fit=norm)
fig=plt.figure()
res=stats.probplot(df_train['GrLivArea'],plot=plt)
#同上
df_train['GrLivArea']=np.log(df_train['GrLivArea'])
test['GrLivArea']=np.log(test['GrLivArea'])
sns.distplot(df_train['GrLivArea'],fit=norm)
fig=plt.figure()
res=stats.probplot(df_train['GrLivArea'],plot=plt)
#test log转换
sns.distplot(test['GrLivArea'],fit=norm)
fig=plt.figure()
res=stats.probplot(test['GrLivArea'],plot=plt)
#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'],fit=norm)
fgi=plt.figure()
res=stats.probplot(df_train['TotalBsmtSF'],plot=plt)
#这个比较特殊，有0值，不能应用log
#解决办法：因为地下室分为 有和没有，二元变量，我们可以新增一个变量，如果有地下室就取1，or取0
#为新的变量创建一列（因为是二元变量，所以一个就足够了）
#if(area>0) it gets 1,for area==0 it gets 0
df_train['HasBsmt']=pd.Series(len(df_train['TotalBsmtSF']),index=df_train.index)
df_train['HasBsmt']=0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt']=1

test['HasBsmt']=pd.Series(len(test['TotalBsmtSF']),index=test.index)
test['HasBsmt']=0
test.loc[test['TotalBsmtSF']>0,'HasBsmt']=1
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF']=np.log(df_train['TotalBsmtSF'])
test.loc[test['HasBsmt']==1,'TotalBsmtSF']=np.log(test['TotalBsmtSF'])
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'],fit=norm)
fgi=plt.figure()
res=stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'],plot=plt)
sns.distplot(test[test['TotalBsmtSF']>0]['TotalBsmtSF'],fit=norm)
fgi=plt.figure()
res=stats.probplot(test[test['TotalBsmtSF']>0]['TotalBsmtSF'],plot=plt)
#5.2 homoscedasticity
plt.scatter(df_train['GrLivArea'],df_train['SalePrice'])
#与之前的散点图相比，这个没有锥形，这是正太分布的魔力。通过确保正太分布，我们可以解决同方差问题
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'],df_train[df_train['TotalBsmtSF']>0]['SalePrice'])
df_train.drop('Id',axis=1,inplace=True)
df_train.drop('SalePrice',axis=1,inplace=True)
test.drop('Id',axis=1,inplace=True)
'''不能删除这些特征
dataset=[df_train,test]
for data in dataset:
    data.drop('TotRmsAbvGrd',axis=1,inplace=True)
    #data.drop('FullBath',axis=1,inplace=True)
    data.drop('BedroomAbvGr',axis=1,inplace=True)
    data.drop('BsmtFinSF1',axis=1,inplace=True)
    data.drop('BsmtFullBath',axis=1,inplace=True)
    data.drop('GarageArea',axis=1,inplace=True)
    data.drop('2ndFlrSF',axis=1,inplace=True)'''
df_train=pd.get_dummies(df_train)
test=pd.get_dummies(test)
print(df_train.shape)
print(test.shape)
test=test.reindex(columns=df_train.columns)
test.fillna(0,inplace=True)
test.isnull().sum().max()
#df_train.columns
from sklearn.decomposition import PCA
pca=PCA(n_components=175)
pca.fit(df_train)
pca.fit(test)
p_train=pca.transform(df_train)
p_test=pca.transform(test)
#p_train=pca.fit_transform(df_train)
#p_test=pca.fit_transform(test)
#df_train=pca.fit_transform(df_train)
#test=pca.fit_transform(test)
sum(pca.explained_variance_ratio_[:10])

p_train[:1]
#df_train[df_train.isnull().values==True]#返回空值的行列
#print(p_train.shape)
#print(p_test.shape)
#test=test.reindex(columns=df_train.columns)
#test.fillna(0,inplace=True)
#test.isnull().sum().max()
test.head()
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
Y_label.head()
#train_x,test_x,train_y,test_y=train_test_split(p_train,Y_label,random_state=0)
#train_x.dropna(axis=0,how='any');
#train_x,test_x,train_y,test_y=train_test_split(p_train,Y_label,random_state=0)
lassoCV=linear_model.LassoCV(alphas=[1,10,20,30,35,40,45,50,55,58,59,60,62,65,67,68,70,75,78,80,81,82,100,105,110,115,120,150],cv=5)
#lassoCV.fit(train_x,train_y)
lassoCV.fit(p_train,Y_label)
#predict=lassoCV.predict(test_x)
print('best alpha:',lassoCV.alpha_)
#print('RMSE:',np.sqrt(mean_squared_error(predict,test_y)))
'''ridgeCV=linear_model.RidgeCV(alphas=[2,3,5,6,6.5,7,7.5,8,9,10,11,12,13,20,30100],cv=5)
ridgeCV.fit(train_x,train_y)
predict=ridgeCV.predict(test_x)
print('best alpha:',ridgeCV.alpha_)
print('RMSE:%.8f'%np.sqrt(mean_squared_error(predict,test_y)))'''
#print(predict[:10])
#print(test_y[:10])
'''elasticNetCV=linear_model.ElasticNetCV(alphas=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2],l1_ratio=[0.1,0.2,0.5,0.7,0.8,0.85,0.9,0.95,1.0],cv=5)
elasticNetCV.fit(train_x,train_y)
#elasticNetCV.fit(df_train,Y_label)
predict=elasticNetCV.predict(test_x)
#predict=elasticNetCV.predict(test)
print('best alpha:',elasticNetCV.alpha_)
print('best l1_ratio_:',elasticNetCV.l1_ratio_)
print('RMSE:',np.sqrt(mean_squared_error(predict,test_y)))'''
#print(predict[:10])
#print(test_y[:10])
#predict=pd.DataFrame(predict)
#predict[predict[0]<0]
submission['SalePrice']=lassoCV.predict(p_test)
submission.head()
submission[submission['SalePrice']<0]
submission.to_csv('submission.csv',index=False)