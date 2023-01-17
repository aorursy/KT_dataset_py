import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport

from scipy.stats import zscore

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from yellowbrick.features.pca import PCADecomposition

from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler
df=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
profile_report = ProfileReport(df, title='Profile Report', html={'style':{'full_width':True}})
profile_report.to_notebook_iframe()
# basement features

df.filter(regex='\Bsmt').columns
# 5 higher square feet

dfbs=df.nlargest(5, 'TotalBsmtSF')

dfbs
# excluding the bigger basement and selecting some features

dfbs2=dfbs.loc[[332, 496, 523,440], ["LotArea", "YearBuilt", "TotalBsmtSF", "BsmtCond","SalePrice" ]]

dfbs2
# other options for selecting/slicing

print(dfbs2.equals(dfbs.loc[332:440, ["LotArea", "YearBuilt", "TotalBsmtSF", "BsmtCond","SalePrice" ]]))

print(dfbs2.equals(dfbs.iloc[[1,2,3,4],[4, 19, 38, 31, 80]]))

print(dfbs2.equals(dfbs.iloc[1:5,[4, 19, 38, 31, 80]]))
# checking about Basement Condition

print(df["BsmtCond"].value_counts())
# checking about Basement Condition

df.groupby(by="BsmtCond")['SalePrice'].mean().sort_values(ascending=False)
print("The dataset has {} rows and {} columns. {} duplicated rows".format(df.shape[0], df.shape[1],df.duplicated().sum()))
s_types= df.dtypes

s_head= df.apply(lambda x: x[0:3].tolist())



explo1 = pd.DataFrame({'Types': s_types,

                      'Head': s_head}).sort_values(by=['Types'],ascending=False)

explo1.transpose()
s_missing= df.isnull().sum()

s_missingper= (df.isnull().sum()/df.shape[0])*100



explo2 = pd.DataFrame({'Types': s_types,

                       'Missing': s_missing,

                      'Missing%': s_missingper,}).sort_values(by=['Missing%','Types'],ascending=False)

explo2.transpose()
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

           'PoolQC','Fence','MiscFeature'):

    df[col]=df[col].fillna('None')
for col in ('Electrical','MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):

    df[col]=df[col].fillna(df[col].mode()[0])
for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',

            'GarageYrBlt','GarageCars','GarageArea'):

    df[col]=df[col].fillna(0) 
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
print(df.isnull().sum().sum())
#npo= number of possible outliers

list_of_numerics=df.select_dtypes(include=['float','int']).columns

s_npo= df.apply(lambda x: sum(i>3 for i in np.abs(zscore(x)))if x.name in list_of_numerics else '') 

s_npo2= df.apply(lambda x: sum((x < (x.quantile(0.25) - 1.5 * (x.quantile(0.75)- x.quantile(0.25))))|

                               (x > (x.quantile(0.75) + 1.5 * (x.quantile(0.75)- x.quantile(0.25)))))

                 if x.name in list_of_numerics else '')



explo3 = pd.DataFrame({'NPO': s_npo,

                       'NPO2': s_npo2}).sort_values(by=['NPO', 'NPO2'])

explo3.transpose()
fig, axes = plt.subplots(1,2, figsize=(12,5))



ax1= sns.scatterplot(x='GrLivArea', y='SalePrice', data= df,ax=axes[0])

ax2= sns.boxplot(x='GrLivArea', data= df,ax=axes[1])
#removing outliers recomended by author

df= df[df['GrLivArea']<4000]
plt.style.use('classic')

fig, axes = plt.subplots(1,4, figsize=(22,5))



ax1= sns.distplot(df.LotFrontage, bins= 30, hist_kws={'edgecolor':'k'},ax=axes[0])

ax1.set_title('LotFrontage')



ax2= sns.distplot(MinMaxScaler().fit_transform(df[['LotFrontage']]), bins= 30, hist_kws={'edgecolor':'k'},ax=axes[1])

ax2.set_title('LotFrontage with normalization')



ax3= sns.distplot(StandardScaler().fit_transform(df[['LotFrontage']]), bins= 30, hist_kws={'edgecolor':'k'},ax=axes[2])

ax3.set_title('LotFrontage with standardization')



ax4= sns.distplot(np.log(df[['LotFrontage']]), bins= 30, hist_kws={'edgecolor':'k'},ax=axes[3])

ax4.set_title('LotFrontage with log transformation')
df_num=pd.get_dummies(df)

x= df_num.drop(['SalePrice'], axis=1)

y= df_num.SalePrice



visu= PCADecomposition(scale=True)

visu.fit_transform(x,y)

visu.show()
np.random.seed(1)

uni=SelectKBest( mutual_info_regression, k=5).fit(x,y)

print(x.columns[uni.get_support(indices=True)].tolist())
est= DecisionTreeRegressor(random_state=1)

eli=RFE(est, 5).fit(x,y)

print(x.columns[eli.support_].tolist())
RF= RandomForestRegressor(random_state=1)

RF.fit(x,y)

importances=RF.feature_importances_

s_importances=pd.Series(importances, index=x.columns).sort_values(ascending=False)
plt.style.use('dark_background')

fig, axes = plt.subplots(1,2, figsize=(13,5), sharey=True) 



ax1= sns.barplot(x=s_importances.index[0:5], y= s_importances[0:5], ax=axes[0])

ax1.set_title('Feature Importance')



trees=pd.DataFrame(data=[tree.feature_importances_ for tree in RF], columns=x.columns)[s_importances[0:5].index.tolist()]

ax2= sns.boxplot(data=trees, ax=axes[1])

ax2.set_title('Feature Importance Distributions')
df["LotFrontage_bin_interval"]= pd.cut(df.LotFrontage, 3, labels=["small","medium","large"])



df["LotFrontage_bin_frequency"]= pd.qcut(df.LotFrontage, 3, labels=["small","medium","large"])
plt.style.use('default')

fig, axes= plt.subplots(1,3, figsize=(13,4),sharey=True)



ax1= sns.distplot(df.LotFrontage, bins=40, hist_kws={'edgecolor':'k'}, color='darkorchid',kde=False,ax=axes[0])

ax1.set_title('Histogram')

ax1= sns.despine()



ax2= sns.countplot(x='LotFrontage_bin_frequency', data=df, palette="Purples", ax=axes[1])

ax2.set_title('Binning with equal frequency')

ax2= sns.despine()



ax3= sns.countplot(x='LotFrontage_bin_interval', data=df, palette="Purples", ax=axes[2])

ax3.set_title('Binning with equal interval')

ax3= sns.despine();