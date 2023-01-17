#basic packages
import os

import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
pd.set_option('display.max_columns',None)

import seaborn as sns
sns.set(style='ticks',color_codes=True,font_scale=1.5)
color = sns.color_palette()
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


from scipy import stats
from scipy.stats import skew, norm, probplot, boxcox
from scipy.special import boxcox1p
import statsmodels.api as sm
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head()
test.head()
# save Id columns
train_Id = train['Id']
test_Id = test['Id']

train.drop('Id',axis=1,inplace =True)
test.drop('Id',axis=1,inplace =True)

test['SalePrice']=0
test.head()
print(train.shape)
print(test.shape)
train.describe().T
def analysis(df, target):
    instance = df.shape[0]
    types=df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques=df.T.apply(pd.Series.unique,1)
    nulls= df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(pd.Series.nunique)
    null_perc = (df.isnull().sum()/instance)*100
    skewness = df.skew()
    kurtosis = df.kurt()
    
    corr = df.corr()[target]
    str = pd.concat([types, counts,uniques, nulls,distincts, null_perc, skewness, kurtosis, corr], axis = 1, sort=False)
    corr_col = 'corr '  + target
    cols = ['types', 'counts','uniques', 'nulls','distincts', 'missing_ration', 'skewness', 'kurtosis', corr_col ]
    str.columns = cols
    return str
details = analysis(train,'SalePrice')
details.sort_values(by='corr SalePrice',ascending =False)
#visualizing features with high correlation

fig = plt.figure(figsize=(20,15))
sns.set(font_scale=1.5)

#box plot overallqual 
fig1 = plt.subplot(221);
sns.boxplot(x='OverallQual', y='SalePrice', data=train[['SalePrice', 'OverallQual']])

#scatter plot GrLivarea
fig2 = plt.subplot(222);
sns.scatterplot(x=train.GrLivArea,y=train.SalePrice, hue=train.OverallQual, palette= 'Spectral')

#scatter plot GarageCars
fig3 = plt.subplot(223);
sns.scatterplot(x=train.GarageCars,y=train.SalePrice, hue=train.OverallQual, palette= 'Spectral')

#scatter plot GarageArea
fig4 = plt.subplot(224);
sns.scatterplot(x=train.GarageArea,y=train.SalePrice, hue=train.OverallQual, palette= 'Spectral')


fig = plt.figure(figsize=(20,15))
sns.set(font_scale=1.5)

fig6 = plt.subplot(221); 
sns.scatterplot(y = train.SalePrice , x = train.TotalBsmtSF, hue=train.OverallQual, palette= 'BrBG')

fig7 = plt.subplot(222); 
sns.scatterplot(y = train.SalePrice, x = train['1stFlrSF'], hue=train.OverallQual, palette= 'BrBG')

fig8= plt.subplot(223);
sns.scatterplot(y = train.SalePrice, x = train.FullBath, hue=train.OverallQual, palette= 'BrBG')

fig9 =plt.subplot(224);
sns.scatterplot(y = train.SalePrice, x = train.YearBuilt, hue=train.OverallQual, palette= 'BrBG')
# removing outlier 2 from above list
train = train[train['GrLivArea']<4500]

#plotting again to find the difference

fig = plt.figure(figsize=(20,15))
sns.set(font_scale=1.5)

#scatter plot TotalBsmtSF
fig1 = plt.subplot(221); 
sns.scatterplot(y = train.SalePrice , x = train.TotalBsmtSF, hue=train.OverallQual, palette= 'BrBG')

#scatter plot GrLivarea
fig2 = plt.subplot(222);
sns.scatterplot(x=train.GrLivArea,y=train.SalePrice, hue=train.OverallQual, palette= 'Spectral')

#scatter plot 1stFlrSF
fig3 = plt.subplot(223); 
sns.scatterplot(y = train.SalePrice, x = train['1stFlrSF'], hue=train.OverallQual, palette= 'BrBG')

#scatter plot GarageArea
fig4 = plt.subplot(224);
sns.scatterplot(x=train.GarageArea,y=train.SalePrice, hue=train.OverallQual, palette= 'Spectral')


fig = plt.figure(figsize=(18,8))
sns.set(font_scale=1.5)

fig1 =plt.subplot(121);
sns.scatterplot(y = train.SalePrice, x = train.YearBuilt, hue=train.OverallQual, palette= 'Spectral')

fig2 =plt.subplot(122);
sns.scatterplot(y = train.SalePrice, x = train.YearRemodAdd, hue=train.OverallQual, palette= 'Spectral')

#scatter plot GarageCars
#fig3 = plt.subplot(223);
#sns.scatterplot(x=train.GarageCars,y=train.SalePrice, hue=train.OverallQual, palette= 'Spectral')

#scatter plot GarageArea
#fig4 = plt.subplot(224);
#sns.scatterplot(x=train.GarageArea,y=train.SalePrice, hue=train.OverallQual, palette= 'Spectral')


# let us findout all the features with year values
yr  = [x for x in train.columns if 'Year' in x or 'Yr' in x]
yr
data = train.copy()
data['Age'] = data['YrSold']-data['YearBuilt']
data['RemodAge']= data['YrSold']-data['YearRemodAdd']

plt.figure(figsize=(18,8))
sns.set(font_scale=1.5)
fig1=plt.subplot(121);
sns.scatterplot(y = data.SalePrice, x = data.Age, hue=data.OverallQual, palette= 'Spectral')

fig2 =plt.subplot(122);
sns.scatterplot(y = data.SalePrice, x = data.RemodAge, hue=data.OverallQual, palette= 'Spectral')
Corr_year = data[['RemodAge','Age','YrSold','SalePrice']]
co = Corr_year.corr()['SalePrice']
print(co)
#train['Age']=data['Age']
#train['RemodAge']=data['Age']
#train.head()
#Corr_year
# we find that RemodAge and Age are slightly better correlated to Saleprice then YrRemodAdd and YrBuilt==
sns.lineplot(x='YrSold',y='SalePrice',data=train)
# it is evident that with more age price is decreasing
#sns.scatterplot(x='Age',y='SalePrice',data=data) #probable outliers
sns.lineplot(x='Age',y='SalePrice',data=data)  
sns.lineplot(x='RemodAge',y='SalePrice',data=data) 
data=data[data['Age']>100]
data=data[data['SalePrice']>285000]
data
# remod age is very less and hence these are not considered as outliers
fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(131); sns.boxplot(train.GarageCars)
fig2 = fig.add_subplot(132); sns.boxplot(train.GarageArea)
fig3 = fig.add_subplot(133); sns.boxplot(train.GarageCars, train.GarageArea)
plt.show()
df = train[['SalePrice', 'GarageArea', 'GarageCars']]
df['GarageAreaByCar'] = train.GarageArea/train.GarageCars
df['GarageArea_x_Car'] = train.GarageArea*train.GarageCars

fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(121); sns.regplot(x='GarageAreaByCar', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.GarageAreaByCar.corr(df['SalePrice'])))

fig2 = fig.add_subplot(122); sns.regplot(x='GarageArea_x_Car', y='SalePrice', data=df); plt.legend(['Outliers'])
plt.text(x=-100, y=750000, s='Correlation with SalePrice: {:6.4f}'.format(df.GarageArea_x_Car.corr(df['SalePrice'])))
print('                                                                 Outliers:',(df.GarageArea_x_Car>=3700).sum())
df = df.loc[df.GarageArea_x_Car<3700]
sns.regplot(x='GarageArea_x_Car', y='SalePrice', data=df); plt.title('Garage Area Multiply By Cars is the best!')
plt.text(x=-100, y=700000, s='Correlation without Outliers: {:6.4f}'.format(df.GarageArea_x_Car.corr(df['SalePrice'])))
plt.show()
del df
             
train = train[train.GarageArea * train.GarageCars < 3700]

bath_feature= [x for x in train.columns if 'Bath' in x]
bath_feature
for feature in bath_feature:
    print(train[feature].unique())

fig = plt.figure(figsize=(20,12))
fig1= plt.subplot(2,2,1);
sns.regplot(x='FullBath',y='SalePrice',data=train)
plt.title('Correlation: {:6.4f} '.format(train.FullBath.corr(train['SalePrice'])))

fig2= plt.subplot(2,2,2);
sns.regplot(x='HalfBath',y='SalePrice',data=train)
plt.title('Correlation: {:6.4f}'.format(train.HalfBath.corr(train['SalePrice'])))


fig3= plt.subplot(2,2,3);
sns.regplot(x='BsmtFullBath',y='SalePrice',data=train)
plt.title('Correlation: {:6.4f}'.format(train.BsmtFullBath.corr(train['SalePrice'])))

fig4= plt.subplot(2,2,4);
sns.regplot(x='BsmtHalfBath',y='SalePrice',data=train)
plt.title('Correlation:{:6.4f} '.format(train.BsmtHalfBath.corr(train['SalePrice'])))

df = train[bath_feature]
df['SalePrice'] = train['SalePrice']

df['TotBath']= df.FullBath + df.BsmtFullBath + (df.HalfBath*0.5)+ (df.BsmtHalfBath*0.5)

fig = plt.figure(figsize=(15,8))
fig1= plt.subplot(121);
sns.regplot(x='TotBath',y='SalePrice',data=df)
plt.title('Correlation with outliers: {:6.4f} '.format(df.TotBath.corr(df['SalePrice'])))

fig2=plt.subplot(122)
sns.regplot(x='TotBath', y='SalePrice', data=df);
plt.title('Correlation with outliers: {:6.4f} '.format(df.TotBath.corr(df['SalePrice'])))
plt.legend('outliers')

df=df.loc[df['TotBath']<5]
sns.regplot(x='TotBath', y='SalePrice', data=df);
plt.text(x=1,y=700000,s='Correlation without outliers: {:6.4f} '.format(df.TotBath.corr(df['SalePrice'])))

#fig2= plt.subplot(1,2,2);
#sns.regplot(x='HalfBath',y='SalePrice',data=train)
#plt.title('Correlation: {:6.4f}'.format(train.HalfBath.corr(train['SalePrice'])))

train = train[(train.FullBath + (train.HalfBath*0.5) + train.BsmtFullBath + (train.BsmtHalfBath*0.5))<5]
SF_feature= [x for x in train.columns if 'SF' in x]
SF_feature=SF_feature[3:6]
SF_feature
df=train[SF_feature]
df['SalePrice']=train['SalePrice']
df['TotalSF']=train['1stFlrSF']+train['2ndFlrSF']
for i in df.columns:
    print(i,df[i].corr(df['SalePrice']))
fig = plt.figure(figsize=(20,6))
fig1= plt.subplot(1,3,1);
sns.regplot(x='1stFlrSF',y='SalePrice',data=train)
plt.title('Correlation: {:6.4f} '.format(train['1stFlrSF'].corr(train['SalePrice'])))

fig2= plt.subplot(1,3,2);
sns.regplot(x='2ndFlrSF',y='SalePrice',data=train)
plt.title('Correlation: {:6.4f}'.format(train['2ndFlrSF'].corr(train['SalePrice'])))


fig3= plt.subplot(1,3,3);
sns.regplot(x='TotalBsmtSF',y='SalePrice',data=train)
plt.title('Correlation: {:6.4f}'.format(train.TotalBsmtSF.corr(train['SalePrice'])))
num_feature= [x for x in train.columns if train[x].dtypes != 'O' ]
num_feature
lessunique = [x for x in num_feature if train[x].nunique() < 25]
lessunique


for feature in lessunique:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.title(feature)
    plt.ylabel('SalePrice')
    plt.show()

moreunique = [x for x in num_feature if x not in lessunique]
moreunique
for feature in moreunique:
    data=train.copy()
    data[feature].hist(bins=20)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
deatils = analysis(train,'SalePrice')
details.sort_values(by='corr SalePrice', ascending=False)
cat_feature= [x for x in train.columns if train[x].dtype=='O']
cat_feature
for feature in cat_feature:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
y_train = train.SalePrice.values
fig=plt.figure(figsize=(15,7))

fig1=plt.subplot(121);
sns.distplot(train['SalePrice'],norm_hist=True,kde=True)
plt.text(x=0,y=0.000009,s='Skewness: {:.3f} and Kurtosis: {:.3f}'.format(train['SalePrice'].skew(),train['SalePrice'].kurtosis()))

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


train.SalePrice = np.log1p(train.SalePrice)
fig=plt.figure(figsize=(15,7))
fig2= plt.subplot(121)
sns.distplot(train['SalePrice'],norm_hist=True,kde=True)
plt.text(x=10,y=1.2,s='Skewness: {:.3f} and Kurtosis: {:.3f}'.format(train['SalePrice'].skew(),train['SalePrice'].kurtosis()))

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
