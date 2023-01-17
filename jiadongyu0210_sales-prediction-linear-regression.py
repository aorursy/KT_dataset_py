import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

redbook=pd.read_csv('../input/linear-regression-sales-prediction/.csv')

#if do linear_regression:
#gender:0,1,unknown, get_dummies.  
#age:use mean \ median to fillna  
#engaged_last_30:0,1,unknown, get_dummies
#lifecycle:get_dummies
#3rd_party_stores:get_dummies

#points can be improve:
#1. outlier 可以通过过滤1.5IQR重新定义。
#2. 不填写信息的人可否定义为另一类型。 发现数据中gender, age, engaged_last_30_Days都是空，应该是系统性偏差。也可以以全部去除这些数据。
#3. P-VALUE 选择都小于0.05的最好！
#4.redbook.isnull().sum()/len(redbook)
#5.redbook.loc[~redbook['age'].isna(),'age'].describe()
redbook.describe()
redbook.info()
redbook.head()
#重新处理column 名字的空格问题' days_since_last_order'
redbook.columns = ['revenue','gender','age',
                     'engaged_last_30','lifecycle','days_since_last_order','previous_order_amount','3rd_party_stores']

#redbook.rename(columns={' days_since_last_order':'days_since_last_order'},inplace=True)
redbook.info()
#观察是否改回来
redbook['3rd_party_or_not']=redbook['3rd_party_stores'].apply(lambda x:1 if x>0 else 0)
#revenue分布 观察是否有异常值存在。
bins=[0,100,200,500,1000,5000,10000,50000,110000]
redbook['revenue_level']=pd.cut(redbook.revenue,bins,right=False)
redbook.groupby(['revenue_level'])['revenue'].describe()
sns.countplot(y='revenue_level',data=redbook)
#因为revenue=5000以上的数据较少 可以进行drop操作。
redbook=redbook.drop(redbook[redbook['revenue']>=5000].index)
#单变量分析
#性别分布分析
redbook['gender']=redbook.gender.fillna('unknown')
sns.countplot(x='gender',data=redbook)
#未知性别分布较多，不建议删除，后续回归分析可用get_dummies.
#现有数据年龄分布分析
#age fillna(0)
#age fillna(mean)
#age fillna(medium)
redbook['age_0']=redbook.age.fillna(0) #NAN 暂时替代为0，用来做年龄分组分析。
bins=[0,10,15,20,25,30,35,40,45,50]
redbook['age_level']=pd.cut(redbook.age_0,bins,right=False)
redbook.groupby(['age_level'])['age_0'].describe()
sns.countplot(y='age_level',data=redbook)
#[0,10]定义为未知年龄分组较多，不建议删除，后续回归分析可用get_dummies.
#不同年龄分组对应的revenue变化
plt.figure(figsize=(10,8))
sns.barplot(x='age_level',y='revenue',hue='revenue_level',data=redbook)
#单变量分析
#过去三十天是否engaged_last_30
redbook['engaged_last_30']=redbook.engaged_last_30.fillna('unknown')
sns.countplot(x='engaged_last_30',data=redbook)
#未知分布也较多,不建议删除，后续回归分析可用get_dummies.
#是否参与重要活动对应的revenue情况
sns.barplot(x='engaged_last_30',y='revenue',hue='revenue_level',data=redbook)
#days_since_last_order分布情况
sns.distplot(redbook['days_since_last_order'])
#暂时没有特别异常的值出现
#previous oder amount分布情况，观察是否有异常值出现。
bins=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,15000]
redbook['previous_order_amount_level']=pd.cut(redbook.previous_order_amount,bins,right=False)
redbook.groupby(['previous_order_amount_level'])['previous_order_amount'].describe()
sns.countplot(y='previous_order_amount_level',data=redbook)
#没有特别异常的情况出现，无需进一步处理。
#3rd_party_stores分布情况分析。回归分析时可用get_dummies进行分析
sns.countplot(x='3rd_party_stores',data=redbook)
sns.countplot(x='3rd_party_or_not',data=redbook)
sns.barplot(x='3rd_party_or_not',y='revenue' ,hue='revenue_level',data=redbook)
#生命周期分布情况管观察。回归分析时可用get_dummies进行分析。
sns.countplot(x='lifecycle',data=redbook)
#不同生命周期对应的revenue情况
sns.barplot(x='lifecycle',y='revenue',hue='3rd_party_or_not',data=redbook)
#不同生命周期对应的是否适用第三方用户分布情况
sns.countplot(x='lifecycle',hue='3rd_party_or_not',data=redbook)
#相关性可视化分析
#对特定columns进行dummy运算
#revenue与其他变量的相关性分析。
redbook['age_mean']=redbook.age.fillna(redbook.age.mean())
redbook['age_median']=redbook.age.fillna(redbook.age.median())
redbook=pd.get_dummies(redbook, prefix=['gender', 'engaged_last_30','3rd_party_or_not','lifecycle'], columns=['gender','engaged_last_30', '3rd_party_or_not','lifecycle'])

redbook.corr()[['revenue']].sort_values('revenue',ascending=False)
#可视化拟合TOP3线性关系 
#previous_order_amount 0.209676
#3rd_party_stores_0 0.096531
#days_since_last_order 0.087386

sns.regplot('previous_order_amount','revenue',data=redbook)
sns.regplot('3rd_party_or_not_0','revenue',data=redbook)
sns.regplot('days_since_last_order','revenue',data=redbook)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
#线性回归模型建立
#model1
#自变量选择如下：
#其中年龄缺失值用中值替代
#'previous_order_amount','3rd_party_stores_0','days_since_last_order','engaged_last_30_1.0','gender_unknown','engaged_last_30_unknown','lifecycle_C','gender_1.0','age_mean'
x1=redbook[['previous_order_amount','3rd_party_or_not_0','days_since_last_order','engaged_last_30_1.0','engaged_last_30_unknown','gender_unknown','age_median']]

y=redbook['revenue']

model.fit(x1,y)
coef_1=model.coef_
intercept_1=model.intercept_

print('model1_coef:',coef_1)
print('model1_intercept:',intercept_1)
#model valuation
score1=model.score(x1,y)
predictions1=model.predict(x1)
error1=predictions1-y
RMSE_1=(error1**2).mean()**.5
MAE_1=abs(error1).mean()
print('model1_RMSE:',RMSE_1)
print('model1_MAE:',MAE_1)
#Model2
model=LinearRegression()
#其中年龄缺失值用均值替代
#增加自变量变量
x2=redbook[['previous_order_amount','3rd_party_or_not_0','days_since_last_order','engaged_last_30_1.0','engaged_last_30_unknown','gender_unknown','3rd_party_or_not_1','age_mean','engaged_last_30_0.0','gender_0.0']]

model.fit(x2,y)
coef_2=model.coef_
intercept_2=model.intercept_

print('model2_coef:',coef_2)
print('model2_intercept:',intercept_2)
#model valuation
score2=model.score(x2,y)
predictions2=model.predict(x2)
error2=predictions2-y
RMSE_2=(error2**2).mean()**.5
MAE_2=abs(error2).mean()
print('model2_RMSE:',RMSE_2)
print('model2_MAE:',MAE_2)
#ols 
from statsmodels.formula.api import ols
model=ols('y~x1',redbook).fit()
print(model.summary())
#ols 
from statsmodels.formula.api import ols
model=ols('y~x2',redbook).fit()
print(model.summary())