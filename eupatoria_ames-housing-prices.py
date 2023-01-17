# I don't use all of the libraries below as I have been exploring different strategies, but most have been used



import numpy as np 

import matplotlib.pyplot as plt

import numpy as np



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

from pandas.plotting import scatter_matrix # I didn't end up using it because the dataset has too many dimensions, but can keep it here for now



from datetime import datetime



# the below allows to see all fields and rows of the output when running analysis -- important when we are starting with 80+ fields

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.width', None)



import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')



from scipy import stats

from scipy.stats import norm, skew, boxcox



from IPython.display import FileLink # to export outcome to .csv for a Kaggle submission



import missingno as mn



from sklearn import preprocessing, decomposition, preprocessing, cluster, tree

from sklearn.preprocessing import LabelEncoder, power_transform, StandardScaler, PowerTransformer

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer



%matplotlib inline

plt.style.use('ggplot')



# Limit floats to 3 decimal points

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# training dataset from Kaggle

df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv') # saving a copy of the original dataset if we need to check against it when we do our work



df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv') 

y = df_train.SalePrice # target variable

# df_train = df.drop(['SalePrice'],axis=1)

df_train['source_df'] = 'train'



# testing dataset from Kaggle

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_test['source_df'] = 'test'
plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'], color="red", edgecolors="#000000", linewidths=0.5);

plt.xlabel('Gr Liv Area')

plt.ylabel('SalePrice')
df_test[df_test['GrLivArea'] > 4000]
df_train = df_train[df_train['GrLivArea'] < 4000]
df_train.index.values
df_test.index.values
df_all = pd.concat([df_test,df_train.drop(['SalePrice'],axis=1)],ignore_index=True)

df_all = df_all.reset_index(drop=True) # need to reset the index after deleting 4 rows
# checking that the concat function worked

print('Original dataset: ', df.shape) # 81 field originally, including the target variable SalePrice

print('Training dataset: ', df_train.shape) # + 'source_df' column -> 82 columns

print('Testing dataset: ', df_test.shape) # 80 originally + 'source_df' column -> 80+1 = 81

print('Joined dataset: ', df_all.shape) # 80 originally + 'source_df' column -> 80+1 = 81
print('Original dataset -- min ID: ', df.Id.min())

print('Original dataset -- max ID: ',df.Id.max())

print('Training dataset -- min ID: ', df_train.Id.min())

print('Training dataset -- max ID: ',df_train.Id.max())

print('Testing dataset -- min ID: ',df_test.Id.min())

print('Testing dataset -- max ID: ',df_test.Id.max())

print('Joined daset -- min ID: ', df_all['Id'].min())

print('Joined daset -- min ID: ', df_all['Id'].max())
print('Dimensions in the original dataset: ', df.ndim) 

print('Dimensions in the testing dataset: ', df_test.ndim) 
sns.relplot(x="GrLivArea", y="SalePrice", hue="CentralAir", style="CentralAir", data=df_train, height=8, aspect=2)
sns.relplot(x="YearBuilt", y="SalePrice", hue="CentralAir", style="CentralAir", data=df_train, height=8, aspect=2)
sns.relplot(x="YearBuilt", y="SalePrice", data=df_train, height=8, aspect=2)
df_train.columns
df_test.columns
(df_train.drop(['SalePrice'], axis=1).columns == df_test.columns).any()
len(set(df_train.columns))==len(df_train.columns)
len(set(df_test.columns))==len(df_test.columns)
df_train.head(10)
df_train.tail(10)
df_test.head(10)
df_train.dtypes
df_train.info()
df_train.drop('SalePrice',axis=1).info()==df_test.info()
df_train.describe(include=[np.number], percentiles=[.5]).transpose().drop("count", axis=1) 
df_train_desc = df_train.drop(['source_df'], axis=1).describe(include=[np.object]).transpose().drop("count", axis=1)

df_train_desc.rename(columns={'unique':'train_unique_count','top':'train_top','freq':'train_freq'},inplace=True)



df_test_desc = df_test.drop(['source_df'], axis=1).describe(include=[np.object]).transpose().drop("count", axis=1)

df_test_desc.rename(columns={'unique':'test_unique_count','top':'test_top','freq':'test_freq'},inplace=True)



df_total_desc = pd.concat([df_train_desc, df_test_desc], axis=1, sort=False)

df_total_desc['unique_same'] = (df_total_desc['train_unique_count'] ==df_total_desc['test_unique_count'])



df_total_desc['top_same'] = (df_total_desc['train_top'] ==df_total_desc['test_top'])

df_total_desc
len(df.describe(include=[np.object]).transpose().drop("count", axis=1))
# pandas_profiling.ProfileReport(df)
# pandas_profiling.ProfileReport(df_test)
print('* Dates of sale range from {} to {} in the training dataset.'.format(df_train['YrSold'].min(), df_train['YrSold'].max()))

print('* Dates of sale range from {} to {} in the testing dataset.'.format(df_test['YrSold'].min(), df_test['YrSold'].max()))
def check_year_assumptions(df):

    currentYear = datetime.now().year

    print('* All dates in question occurred during or earlier than current year: ',(datetime.now().year >= df.YearBuilt.max()) & (datetime.now().year >= df.GarageYrBlt.max()) & (datetime.now().year >= df.YearRemodAdd.max()) & (datetime.now().year >= df.YrSold.max()))

    print('* Earliest MonthSold is January or later:',df['MoSold'].min()>=1)

    print('* Latest MonthSold is December or earlier:',df['MoSold'].max()<=12)

    

print('Training dataset: ')

check_year_assumptions(df_train)

print()

print('---'*10)

print()

print('Testing dataset: ')

check_year_assumptions(df_test)
print(df_test.YearBuilt.max())

print(df_test.YearRemodAdd.max())

print(df_test.GarageYrBlt.max())

print(df_test.YrSold.max())
off_values = df_test[df_test.GarageYrBlt > 2010]

off_values[['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']]
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].replace(2207.0,2007.0) # df_train didn't have this issue

df_all['GarageYrBlt'] = df_all['GarageYrBlt'].replace(2207.0,2007.0) # need to do the same for the joined dataset
# using joined dataset

fig, ax = plt.subplots(figsize=(20, 10))

ax.scatter(x = df_all.YearBuilt, y = df_all.GarageYrBlt)

plt.xlabel('House Year Built', fontsize=15)

plt.ylabel('Garage Year Built', fontsize=15)
plt.hist(df_train.SalePrice, bins=15)

plt.xlabel('Sale price, $')

plt.ylabel('# Houses')

sns.distplot(df_train['SalePrice'], fit=norm, kde=False, color='red')
fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
sns.violinplot(x=df_train['SalePrice'], inner="quartile", color="red")
sns.boxplot(df_train['SalePrice'], whis=10, color="red")
df_train.YrSold.plot.hist()

plt.xticks(range(2006, 2010))
fig, ax = plt.subplots(figsize=(15, 9))

ax.hist(x = df_train.YearBuilt)

plt.xlabel('House Year Built', fontsize=15)
# checking distribution for the joined dataset

fig, ax = plt.subplots(figsize=(15, 9))

ax.hist(x = df_all.YearBuilt)

plt.xlabel('House Year Built', fontsize=15)
(df_train.groupby('YrSold')[['SalePrice']].mean()).plot()
plt.figure(figsize=(15,8))

sns.boxplot(x=df_train['YrSold'], y=df_train['SalePrice'])
plt.figure(figsize=(15,8))

sns.boxplot(x=df_train['YrSold'], y=df_train['OverallCond'])
plt.figure(figsize=(15,8))

sns.boxplot(x=df_train['YrSold'], y=df_train['OverallQual'])
(df_train.groupby('OverallCond')[['SalePrice']].mean()).plot()
(df_train.groupby('OverallQual')[['SalePrice']].mean()).plot()
(df_train.groupby('MoSold')[['SalePrice']].mean()).plot()
df_train.MoSold.plot.hist() 
#  checking that the joined dataset displays similar distribution

df_all.MoSold.plot.hist()
mean_sale_price = df_train.groupby('MoSold').mean()['SalePrice'].rename({'SalePrice': 'Average Sale Price'})

n_by_month = df_train.groupby('MoSold').count()['SalePrice'].rename({'SalePrice': 'N'})



f, ax = plt.subplots()

ax.plot(mean_sale_price.index, (mean_sale_price - mean_sale_price.mean())/mean_sale_price.std(), label="Average Sale Price")

ax.plot(mean_sale_price.index, (n_by_month - n_by_month.mean())/n_by_month.std(), label="Sold houses count")

ax.legend()
plt.figure(figsize=(12,6))

sns.boxplot(x=df_train['MoSold'], y=df_train['OverallQual'])
pd.pivot_table(df_train,index=["YrSold","MoSold"],values=["SalePrice"]).plot()
pd.pivot_table(df_train,index=["YrSold","MoSold"],values=["SalePrice"]).plot(kind='bar', figsize=(12,7))
pd.pivot_table(df_train,index=['MoSold'], columns=['YrSold'],values=["SalePrice"])
pd.pivot_table(df_train,index=['MoSold'], columns=['YrSold'],values=['SalePrice'],aggfunc='count')
pd.pivot_table(df_train,index=['MoSold'], columns=['YrSold'],values=["SalePrice"]).plot()
pd.pivot_table(df_train,index=['MoSold'], columns=['YrSold'],values=['SalePrice'],aggfunc='count').plot()
df_viz = pd.DataFrame(pd.pivot_table(df_train,index=['MoSold'], columns=['YrSold'],values=['SalePrice'],aggfunc='mean').to_records())

df_viz = df_viz.set_index(['MoSold'])

df_viz.columns = [hdr.replace("(", "").replace(")", "").replace("'","").replace(", ","") \

                     for hdr in df_viz.columns] # should really use regex here instead...

df_viz['mean_monthly_saleprice']=df_viz.mean(axis=1)

df_viz
df_viz[["SalePrice2006","SalePrice2007","SalePrice2008","SalePrice2009","SalePrice2010"]].plot(figsize=(12,6), color=['green', 'brown','yellow','blue','black'], use_index=False)

df_viz['mean_monthly_saleprice'].plot(figsize=(16,6), color=['gray'], kind='bar',use_index=True)
# another quick visualization

fig= plt.figure()

plt.figure(figsize=(16, 6))

sale_prices=df_viz[["SalePrice2006","SalePrice2007","SalePrice2008","SalePrice2009","SalePrice2010"]]

avg_monthly_prices=df_viz['mean_monthly_saleprice']

plt.plot(sale_prices, 'go-', label='average monthly prices')

plt.plot(avg_monthly_prices)

plt.figure(figsize=(16, 6))
# training dataset

mn.matrix(df_train) # can use the corresponding part of Pandas Profiling as well

# mn.matrix(df.iloc[:200,:40])
# testing dataset

mn.matrix(df_test)
# calculate for both train datataset (df) and test dataset (df_test)

# I am calculating these numbers separately instead of using df_all as I want to see how the data is broken down 

def calc_missing_data(df_train, df_test):

    total_train = df_train.isnull().sum()

    total_test = df_test.isnull().sum()

    

    percent_1_train = df_train.isnull().sum()/df_train.isnull().count()*100

    percent_2_train = (round(percent_1_train, 1))

    

    percent_1_test = df_test.isnull().sum()/df_test.isnull().count()*100

    percent_2_test = (round(percent_1_test, 1))

    

    missing_data = pd.concat([total_train, percent_2_train, total_test, percent_2_test], axis=1, keys=['Total Train Missing', '% Train Missing', 'Total Test Missing', '% Test Missing'])

    missing_data = missing_data[(missing_data.T != 0).any()].sort_values(by=['% Train Missing','% Test Missing'],ascending=False)

    return missing_data
calc_missing_data(df_train, df_test)
df_train.PoolQC.value_counts()
df_test.PoolQC.value_counts()
df_train.PoolArea.describe()
len(df_train[df_train.PoolArea > 0])
df_train.MiscFeature.value_counts()
df_test.MiscFeature.value_counts()
df_train.Alley.value_counts()
df_train.Fence.value_counts()
df_train.FireplaceQu.value_counts()
df_train.LotFrontage.hist()
sns.lmplot(x ='LotFrontage', y ='SalePrice', data = df_train)
plt.figure(figsize=(20,14))

sns.boxplot(y='LotFrontage', x='Neighborhood', data=df_all, width=1, palette="colorblind")
df_train.Electrical.value_counts()
# most of the zeroes below can be replaced with 'None' as well 

def impute_data(df):

    # Misc

    df.PoolQC.fillna(0, inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.MiscFeature.fillna('None',inplace=True)

    df.Alley.fillna('None',inplace=True)

    df.Fence.fillna(0, inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.FireplaceQu.fillna(0,inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.Functional.fillna(df.Functional.mode()[0],inplace=True) # true case of missing data, so we need the mode

    df.KitchenQual.fillna(df.KitchenQual.mode()[0],inplace=True) # true case of missing data, so we need the mode

    df.MSZoning.fillna(df.MSZoning.mode()[0],inplace=True) # true case of missing data, so we need the mode

    df.SaleType.fillna(df.SaleType.mode()[0],inplace=True) # true case of missing data, so we need the mode

    df.Utilities.fillna(df.Utilities.mode()[0],inplace=True) # true case of missing data, so we need the mode

    

    # Lot Frontage 

    # df.LotFrontage.fillna(df.LotFrontage.mean(),inplace=True)

    

    # Garage Data -- all the values below have 81 missing entries, which corresponds to the 81 houses without a garage. We can replace with 0.

    df.GarageQual.fillna(0,inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.GarageCond.fillna(0,inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.GarageFinish.fillna(0,inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.GarageType.fillna('None',inplace=True)

    df.GarageYrBlt.fillna(0,inplace=True) # keeping this as 0 because we will do math here

    df.GarageArea.fillna(df.GarageArea.mode()[0],inplace=True)

    df.GarageCars.fillna(df.GarageCars.mode()[0],inplace=True)

    

    # Basement Data -- all the values below have 37 mission entries in the train dataset, which corresponds to the 37 houses without a basement in the train dataset. We can replace with 0. 37 v. 38!!!

    df.BsmtExposure.fillna(0, inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.BsmtFinSF1.fillna(0, inplace=True) # Type 1 finished square feet -> area has to be a number

    df.BsmtFinSF2.fillna(0, inplace=True) # Type 2 finished square feet -> area has to be a number

    df.BsmtFullBath.fillna(df.BsmtFullBath.mode()[0], inplace=True) 

    df.BsmtHalfBath.fillna(df.BsmtHalfBath.mode()[0], inplace=True)   

    df.BsmtUnfSF.fillna(df.BsmtUnfSF.mean(), inplace=True)

    df.BsmtFinType1.fillna(0, inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.BsmtFinType2.fillna(0, inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.BsmtCond.fillna(0, inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.BsmtQual.fillna(0, inplace=True) # I am going to keep it as 0 instead of None because we will later map conditions to integers

    df.TotalBsmtSF.fillna(df.BsmtFullBath.mean(), inplace=True)   

    

    # Exterior

    df.Exterior1st.fillna(df.Exterior1st.mode()[0],inplace=True)

    df.Exterior2nd.fillna(df.Exterior2nd.mode()[0],inplace=True)

    

    # Masonry Data

    df.MasVnrType.fillna(df.MasVnrType.mode()[0], inplace=True) # replace with mean; there is no allowance for no Masonry Veneer

    df.MasVnrArea.fillna(df.MasVnrArea.mean(), inplace=True)  # replace with mean; there is no allowance for no Masonry Veneer

    

    # Electrical Data

    df.Electrical.fillna(df.Electrical.mode()[0],inplace=True)   

    return df
df_train = impute_data(df_train)

df_test = impute_data(df_test)

df_all = impute_data(df_all)
# find out average LotFrontage of each neighborhood using the joined dataset for a larger sample size

df_all.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count'])

df_all["LotAreaCut"] = pd.qcut(df_all.LotArea,10) # discretize

df_all.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])



# use the averages to fill missing variables in df_all

df_all['LotFrontage']=df_all.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

df_all['LotFrontage']=df_all.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# use the averages to fill missing variables in df_all

df_train['LotFrontage']=df_all.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

df_train['LotFrontage']=df_all.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# use the averages to fill missing variables in df_test

df_test['LotFrontage']=df_all.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

df_test['LotFrontage']=df_all.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# remove the interim variable

df_all.drop(['LotAreaCut'],axis=1,inplace=True)
calc_missing_data(df_train, df_test)
square_footage_data = df_train[['SalePrice','LotArea', 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','LowQualFinSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','PoolArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']]

square_footage_data.head(10)
square_footage_data.basement_test = square_footage_data.BsmtFinSF1 + square_footage_data.BsmtFinSF2 + square_footage_data.BsmtUnfSF

square_footage_data.basement_test.equals(square_footage_data.TotalBsmtSF)
square_footage_data['GrLivArea'].equals(square_footage_data['1stFlrSF'] + square_footage_data['2ndFlrSF']+square_footage_data['LowQualFinSF'])
cols = ['SalePrice','LotArea', 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','PoolArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']

square_footage_data = square_footage_data[cols]

# adding a prefix so we remember these fields are sums of other fields (even though we didn't derive them ourselves)

square_footage_data.rename(columns={"TotalBsmtSF": "sum_TotalBsmtSF", "GrLivArea": "sum_GrLivArea"}, inplace=True)

square_footage_data.head(5)
# adding a d_ prefix to indicate a derived feature

square_footage_data['d_TotalLiveArea'] = square_footage_data['sum_TotalBsmtSF']+square_footage_data['sum_GrLivArea'] # assuming these basements are liveable

square_footage_data['d_other_useful_SF'] = square_footage_data['GarageArea']+square_footage_data['PoolArea']+square_footage_data['WoodDeckSF']+square_footage_data['OpenPorchSF']+square_footage_data['ScreenPorch']+square_footage_data['EnclosedPorch']+square_footage_data['3SsnPorch']

square_footage_data['d_tot_avail_SF'] = square_footage_data['sum_TotalBsmtSF']+square_footage_data['sum_GrLivArea']+square_footage_data['d_other_useful_SF']

square_footage_data['d_NumStories'] = df['2ndFlrSF'].apply(lambda x: 2 if x > 0 else 1)

square_footage_data.head(5)
# correlation matrix

corr = square_footage_data.corr()

f, ax = plt.subplots(figsize=(30, 25))

sns.heatmap(corr, vmax=.8,annot_kws={'size': 10}, cmap='coolwarm', annot=True)
# let's check that the datasets have the number of rows and columns we expect

def check_df_shape(df_train,df_test,df_all):

    print('Training dataset shape is: ',df_train.shape) 

    print('Testing dataset shape is: ', df_test.shape)

    print('Joined dataset shape is: ', df_all.shape)   



check_df_shape(df_train,df_test,df_all)
def derive_features(df):

    # ages of property

    df['AgeWhenSold'] = df.YrSold-df.YearBuilt 

    df.drop(['YearBuilt'],axis=1, inplace=True)

    df['YrsSinceRemodel'] = df.YrSold-df.YearRemodAdd

    df.drop(['YearRemodAdd'],axis=1, inplace=True)

    df['GarageAgeWhenSold'] = df.YrSold-df.GarageYrBlt 

    # we want to make sure that if there is no Garage, its age when sold is simply marked as 0

    def clean_up_garage(val):

        if val < 2000:

            return val

        return 0

    df['GarageAgeWhenSold'] = df['GarageAgeWhenSold'].apply(clean_up_garage)

    df.drop(['GarageYrBlt'],axis=1, inplace=True)

    # <- +3 fields, -3 fields

    

    # renaming

    df['sum_TotalBsmtSF'] = df['TotalBsmtSF']

    df.drop(['TotalBsmtSF'], axis=1, inplace=True)

    df['sum_GrLivArea'] = df['GrLivArea']

    df.drop(['GrLivArea'], axis=1, inplace=True) 

    # -> +2, -2

    

    # square footage 

    df.drop(['LowQualFinSF'], axis=1, inplace=True) # -1

    df['pot_drop_1stFlrSF'] = df['1stFlrSF']

    df.drop(['1stFlrSF'], axis=1, inplace=True)

    df['pot_drop_2ndFlrSF'] = df['2ndFlrSF']

    df.drop(['2ndFlrSF'], axis=1, inplace=True)

    

    df.drop(['BsmtFinSF2'], axis=1, inplace=True) # -1

    df['pot_drop_BsmtFinSF1'] = df['BsmtFinSF1']

    df.drop(['BsmtFinSF1'], axis=1, inplace=True)        

    df['pot_drop_BsmtUnfSF'] = df['BsmtUnfSF']

    df.drop(['BsmtUnfSF'], axis=1, inplace=True)

    # -2 fields

    

    # df['d_TotalLiveArea']=df['sum_TotalBsmtSF']+df['sum_GrLivArea'] 

    df['d_other_useful_SF'] = df['GarageArea']+df['PoolArea']+df['WoodDeckSF']+df['OpenPorchSF']+df['ScreenPorch']+df['EnclosedPorch']+df['3SsnPorch']

    df['d_tot_avail_SF'] = df['sum_TotalBsmtSF']+df['sum_GrLivArea']+df['d_other_useful_SF'] # +1 field

    df.drop(['d_other_useful_SF'], axis=1, inplace=True)



    # NET impact +3-3+2-2-2+1 = -1 field

    

    return df
df_train = derive_features(df_train)

df_test = derive_features(df_test)

df_all = derive_features(df_all)
# let's check that the datasets have the number of fields we expect

check_df_shape(df_train,df_test,df_all)
print('Training function max garage age ', df_train['GarageAgeWhenSold'].max())

print('Testing function max garage age: ', df_test['GarageAgeWhenSold'].max())
ticks = np.arange(0, 180, 10)

sns.distplot(df_all['AgeWhenSold']).axes.set_xticks(ticks)
ticks = np.arange(0, 90, 10)

sns.distplot(df_all['YrsSinceRemodel']).axes.set_xticks(ticks)
df_train['YrsSinceRemodel'].corr(df_train['SalePrice'])
df_train['AgeWhenSold'].corr(df_train['SalePrice'])
df_train.hist(figsize=(20,20), xrot=-45)

plt.show()
sns.boxplot(y='SalePrice', x='BldgType', data=df_train, width=0.5, palette="colorblind")
sns.boxplot(y='SalePrice', x='BsmtCond', data=df_train, width=0.5, palette="colorblind")
sns.boxplot(y='SalePrice', x='BsmtExposure', data=df_train, width=0.5, palette="colorblind")
plt.figure(figsize=(20,14))

sns.boxplot(y='SalePrice', x='Neighborhood', data=df_train, width=0.5, palette="colorblind")
plt.figure(figsize=(20,14))

sns.swarmplot(x=df['Neighborhood'], y=df['SalePrice']);
df['Neighborhood'].nunique()
data = pd.concat([df_train.groupby('Neighborhood').mean()['SalePrice'],df_train.groupby('Neighborhood').count()['Id']], axis=1)

data.sort_values(by='SalePrice')
f, ax = plt.subplots()

sns.stripplot(data.sort_values(by='SalePrice').SalePrice, data.sort_values(by='SalePrice').index, orient='h', color='red')
# correlation between sale price (y) and OverallCond

fig, ax = plt.subplots()

ax.scatter(x = df_train.OverallQual, y = df_train.SalePrice)

plt.ylabel('Sale Price, $', fontsize=13)

plt.xlabel('Overall Quality', fontsize=13)

plt.xticks(range(0, 10))
df_train.OverallCond.corr(df_train.SalePrice)
# adding random noise for a clearer visualization of the values

sns.lmplot(x="OverallCond", y="SalePrice", data=df, x_jitter=0.1)
df.OverallCond.plot(kind='hist',color='red',edgecolor='black',figsize=(10,10))

plt.title('Distribution of scores', size=24)

plt.xlabel('Condition Score', size=18)

plt.ylabel('Frequency', size=18)
# confirming distribution for the testing dataset

df_test.OverallCond.plot(kind='hist',color='red',edgecolor='black',figsize=(10,10))

plt.title('Distribution of scores', size=24)

plt.xlabel('Condition Score', size=18)

plt.ylabel('Frequency', size=18)
plt.figure(figsize=(15,7))

sns.boxplot(y='SalePrice', x='Exterior1st', data=df_train, width=0.5, palette="colorblind")
data = pd.concat([df_train.groupby('Exterior1st').mean()['SalePrice'],df_train.groupby('Exterior1st').count()['Id']], axis=1)

f, ax = plt.subplots()

sns.stripplot(data.sort_values(by='SalePrice').SalePrice, data.sort_values(by='SalePrice').index, orient='h', color='red')
df_train.Exterior1st.unique()
df_train.Exterior1st.nunique()
df_test.Exterior1st.unique()
df_test.Exterior1st.nunique()
plt.figure(figsize=(15,7))

sns.boxplot(y='SalePrice', x='Exterior2nd', data=df_train, width=0.5, palette="colorblind")
data = pd.concat([df_train.groupby('Exterior2nd').mean()['SalePrice'],df_train.groupby('Exterior2nd').count()['Id']], axis=1)

f, ax = plt.subplots()

sns.stripplot(data.sort_values(by='SalePrice').SalePrice, data.sort_values(by='SalePrice').index, orient='h', color='red')
df_train.Exterior2nd.unique()
df_train.Exterior2nd.nunique()
df_test.Exterior2nd.unique()
df_test.Exterior2nd.nunique()
plt.figure(figsize=(15,7))

sns.boxplot(y='SalePrice', x='Heating', data=df_train, width=0.5, palette="colorblind")
df_train.Heating.unique()
df_test.Heating.unique()
plt.figure(figsize=(15,7))

sns.boxplot(y='SalePrice', x='MiscFeature', data=df_train, width=0.5, palette="colorblind")
df_train.MiscFeature.unique()
df_test.MiscFeature.unique()
df_train[df_train["MiscFeature"]=='TenC']
def clean_data(df):

    # df.GarageYrBlt = df.GarageYrBlt.astype(int) -- we already dropped this because we calculate Garage Age instead

    # df.BsmtFullBath = df.BsmtFullBath.astype(str)

    # df.BsmtHalfBath = df.BsmtHalfBath.astype(str)

        

    # addressing CentralAir

    lab_enc = preprocessing.LabelEncoder()

    lab_enc.fit(df['CentralAir'])

    var = lab_enc.transform(df['CentralAir'])

    df['CentralAir'] = var

    

    # addressing MiscFeature

    def fix_MiscFeature(val):

        if val in  {'TenC'}:

            return 'Othr'

        return val

    df.MiscFeature = df.MiscFeature.apply(fix_MiscFeature)



    # addressing BldgType

    def fix_BldgType(val):

        if val in  {'1Fam'}:

            return val

        return 'Other'

    df.BldgType = df.BldgType.apply(fix_BldgType)

    

    # addressing MsSubClass

    df.MSSubClass = df.MSSubClass.astype(str)

    def fix_MSSubClass(val):

        if val in  {'20', '60','50','120','30'}:

            return val

        return 'Other'

    df.MSSubClass = df.MSSubClass.apply(fix_MSSubClass)

    df.MSSubClass = df.MSSubClass.astype(str)



    # addressing Condition2

    def fix_Condition2(val):

        if val in  {'Norm'}:

            return val

        return 'Other'

    df.Condition2 = df.Condition2.apply(fix_Condition2)



    # addressing Electrical

    def fix_Electrical(val):

        if val in  {'Sbrkr','FuseA'}:

            return val

        return 'Other'

    df.Electrical = df.Electrical.apply(fix_Electrical)

    

    # addressing Exterior1st

    #Let's categorize Exterior1st as:

    #* Cheapest - 1 - BrkComm, AsbShng, CBlock, AsphShn

    #* Medium - 2 - all else

    #* Most expensive - 3 - CemntBd, Stucco, Stone

    def fix_Exterior1st(val):

        if val in {"BrkComm", "AsbShng", "CBlock", "AsphShn"}:

            return 1

        elif val in {"CemntBd", "Stucco", "Stone"}:

            return 3

        else:

            return 2

    df.Exterior1st = df.Exterior1st.apply(fix_Exterior1st)

    

    # Addressing Exterior2nd: 

    #* Cheapest -1 - CBlock, AsbShng, Brk Cmn, Wd Sdng, MetalSd, AsphShn

    #* Medium - 2 - all other

    #* Most expensive - 3 - ImStucc, BrkFace, VinylSd, CmentBD, Other

    def fix_Exterior2nd(val):

        if val in {"CBlock", "AsbShng", "Brk Cmn", "Wd Sdng", "MetalSd", "AsphShn"}:

            return 1

        elif val in {"ImStucc", "BrkFace", "VinylSd", "CmentBD", "Other"}:

            return 3

        else:

            return 2

    df.Exterior2nd = df.Exterior2nd.apply(fix_Exterior2nd)

    

    # addressing Neighborhood

    # Cheapest: MeadowV + IDOTRR + BrDale

    # 2nd cheapest:  BrkSide + Edwards + OldTown + Sawyer + Blueste + SWISU + NPkVill, NAmes, Mitchel

    # 2nd most expensive: SawyerW + NWAmes + Gilbert + Blmngtn + CollgCr + Crawfor + ClearCr + Somerst + Veenker + Timber

    # Most expensive: StoneBr + NridgHt + NoRidge 

    #def fix_Neighborhood(val):

    #    if val in  {'MeadowV','IDOTRR','BrDale'}:

    #        return 1

    #    elif val in {'BrkSide','Edwards','OldTown','Sawyer','Blueste','SWISU','NPkVill','NAmes','Mitchel'}:

    #        return 2

    #    elif val in {'SawyerW','NWAmes','Gilbert','Blmngtn','CollgCr','Crawfor','ClearCr','Somerst','Veenker','Timber'}:

    #        return 3

    #    elif val in {'StoneBr','NridgHt','NoRidge'}:

    #        return 4

    #    else:

    #        return 0

    #df.Neighborhood = df.Neighborhood.apply(fix_Neighborhood)



    # addressing ExterQual: Evaluates the quality of the material on the exterior 

    # addressing ExterCond: Evaluates the present condition of the material on the exterior

       #Ex-Excellent

       #Gd-Good

       #TA-Average/Typical

       #Fa-Fair

       #Po-Poor

    mapping = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}

    df['ExterQual'] = df['ExterQual'].map(mapping)

    df['ExterCond'] = df['ExterCond'].map(mapping)

    

    # addressing Fence: Fence quality

    # GdPrv-Good Privacy | 4

    # MnPrv-Minimum Privacy | 3

    # GdWo-Good Wood | 2

    # MnWw-Minimum Wood/Wire | 1

    # NA-No Fence | 0

    mapping_Fence = {'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1, 0:0}

    df['Fence'] = df['Fence'].map(mapping_Fence)



    # addressing FireplaceQu: Fireplace quality

    #   Ex-Excellent - Exceptional Masonry Fireplace

    #   Gd-Good - Masonry Fireplace in main level

    #   TA-Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement

    #   Fa-Fair - Prefabricated Fireplace in basement

    #   Po-Poor - Ben Franklin Stove

    #   NA-No Fireplace

    mapping_with_absent_value = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,0:0}

    df['FireplaceQu'] = df['FireplaceQu'].map(mapping_with_absent_value)



    # addressing Foundation

    def fix_Foundation(val):

        if val in  {'PConc','CBlock','BrkTil'}:

            return val

        return 'Other'

    df.Foundation = df.Foundation.apply(fix_Foundation)

    

    # addressing Functional: Home functionality (Assume typical unless deductions are warranted)

    # Typ-Typical Functionality | 7

    # Min1-Minor Deductions 1 | 6

    # Min2-Minor Deductions 2 | 5

    # Mod-Moderate Deductions | 4

    # Maj1-Major Deductions 1 | 3

    # Maj2-Major Deductions 2 | 2

    # Sev-Severely Damaged | 1

    # Sal-Salvage only | 0

    mapping_functional = {'Typ':7, 'Min1':6, 'Min2':5, 'Mod':4, 'Maj1':3, 'Maj2':2, 'Sev':1, 'Sal':0}

    df['Functional'] = df['Functional'].map(mapping_functional)



    # addressing GarageCond: Garage condition

    # Ex-Excellent

    # Gd-Good

    # TA-Typical/Average

    # Fa-Fair

    # Po-Poor

    # NA-No Garage

    # Same as above for GarageQual

    df['GarageCond'] = df['GarageCond'].map(mapping_with_absent_value)

    df['GarageQual'] = df['GarageQual'].map(mapping_with_absent_value)

    

    # addressing GarageFinish: Interior finish of the garage

    # Fin-Finished

    # RFn-Rough Finished

    # Unf-Unfinished

    # NA-No Garage

    mapping_GarageFinish = {'Fin':3, 'RFn':2, 'Unf':1, 0:0}

    df['GarageFinish'] = df['GarageFinish'].map(mapping_GarageFinish)

    

    # address PoolQC : Pool Quality

    # Ex-Excellent

    # Gd-Good

    # TA-Average/Typical

    # Fa-Fair

    # NA-No Pool

    mapping_PoolQC = {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 0:0}

    df['PoolQC'] = df['PoolQC'].map(mapping_PoolQC)

    

    # address HeatingQC -- Heating quality and condition

    #   Ex-Excellent

    #   Gd-Good

    #   TA-Average/Typical

    #   Fa-Fair

    #   Po-Poor

    df['HeatingQC'] = df['HeatingQC'].map(mapping)

    

    # address KitchenQual: Kitchen quality

    #   Ex-Excellent

    #   Gd-Good

    #   TA-Average/Typical

    #   Fa-Fair

    #   Po-Poor

    df['KitchenQual'] = df['KitchenQual'].map(mapping)

    

    # address BsmtQual: Evaluates the height of the basement

    # Ex-Excellent (100+ inches)	

    # Gd-Good (90-99 inches)

    # TA-Typical (80-89 inches)

    # Fa-Fair (70-79 inches)

    # Po-Poor (<70 inches

    # NA-No Basement

    # Same as above for BsmtCond: Evaluates the general condition of the basement

    df['BsmtQual'] = df['BsmtQual'].map(mapping_with_absent_value)

    df['BsmtCond'] = df['BsmtCond'].map(mapping_with_absent_value)

    

    # address BsmtExposure: Refers to walkout or garden level walls

    # Gd-Good Exposure

    # Av-Average Exposure (split levels or foyers typically score average or above)	

    # Mn-Mimimum Exposure

    # No-No Exposure

    # NA-No Basement

    mapping_bsmtexposure = {'Gd':3,'Av':2,'Mn':1,'No':0, 0:0}

    df['BsmtExposure'] = df['BsmtExposure'].map(mapping_bsmtexposure)

    

    # addressing BsmtFinType1: Rating of basement finished area

    # GLQ-Good Living Quarters | 6

    # ALQ-Average Living Quarters | 5

    # BLQ-Below Average Living Quarters | 4

    # Rec-Average Rec Room | 3

    # LwQ-Low Quality | 2

    # Unf-Unfinished | 1

    # NA-No Basement | 0

    # Same as above for BsmtFinType2: Rating of basement finished area (if multiple types)

    mapping_BsmtFinType = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3, 'LwQ':2, 'Unf':1, 0:0}

    df['BsmtFinType1'] = df['BsmtFinType1'].map(mapping_BsmtFinType)

    df['BsmtFinType2'] = df['BsmtFinType2'].map(mapping_BsmtFinType)

    

    # addressing HouseStyle:

    def fix_HouseStyle(val):

        if val in {'1Story', '2Story', '1.5Fin'}:

            return val

        return 'Other'

    df.HouseStyle = df.HouseStyle.apply(fix_HouseStyle)

    

    # addressing SaleType

    def fix_SaleType(val):

        if val in {'WD', 'New'}:

            return val

        return 'Other'

    df.SaleType = df.SaleType.apply(fix_SaleType)    

    

    # addressing PavedDrive

    def fix_PavedDrive(val):

        if val in {'Y'}:

            return val

        return 'N'

    df.PavedDrive = df.PavedDrive.apply(fix_PavedDrive) 

    lab_enc = preprocessing.LabelEncoder()

    lab_enc.fit(df['PavedDrive'])

    var = lab_enc.transform(df['PavedDrive'])

    df['PavedDrive'] = var

    

    # addressing Heating

    # bucket Wall, OthW, and Floor together

    def fix_Heating(val):

        if val in {'Wall', 'OthW','Floor'}:

            return 'Other'

        return val

    df.Heating = df.Heating.apply(fix_Heating)  

    

    # addressing RoofMatl

    def fix_RoofMatl(val):

        if val in {'CompShg'}:

            return val

        return 'Other'

    df.RoofMatl = df.RoofMatl.apply(fix_RoofMatl)

    

    # addressing RoofStyle

    def fix_RoofStyle(val):

        if val in {'Gable', 'Hip'}:

            return val

        return 'Other'

    df.RoofStyle = df.RoofStyle.apply(fix_RoofStyle)   

    

    # addressing SaleCond:

    def fix_SaleCondition(val):

        if val in {'Normal', 'Partial','Abnorml'}:

            return val

        return 'Other'

    df.SaleCondition = df.SaleCondition.apply(fix_SaleType) 

    

    return df
df_train = clean_data(df_train)

df_test = clean_data(df_test)

df_all = clean_data(df_all)
check_df_shape(df_train,df_test,df_all)
df_train.Neighborhood.unique()
df_train[df_train.Neighborhood=='Not a valid neighborhood']
df_train.Functional.value_counts()
df_train.OverallCond.value_counts()
df_train.MiscVal.corr(df_train.SalePrice)
def preliminary_drop_features(df):

    df.drop(['MiscVal'],axis=1,inplace=True)

    df.drop(['Utilities'],axis=1,inplace=True)

    df.drop(['Street'],axis=1,inplace=True) # -3 fields

    return df
df_train = preliminary_drop_features(df_train)

df_test = preliminary_drop_features(df_test)

df_all = preliminary_drop_features(df_all)
# let's check that the datasets have the number of fields we expect

check_df_shape(df_train,df_test,df_all)
skewed_cols = df_train.skew(axis = 0).sort_values(ascending=False)

skewed_cols
plt.hist(df_train.SalePrice, bins=15)

plt.xlabel('Sale price, $')

plt.ylabel('# Houses')

sns.distplot(df_train['SalePrice'], fit=norm, kde=False, color='red')



fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
# let's choose the columns to transform

# there are MUCH easier ways to do this, but I wanted to go through each column to ensure I understood what was happening:

cols_to_keep = ['PoolQC'

,'PoolArea' 

,'LotArea'

,'3SsnPorch'

,'KitchenAbvGr'

,'BsmtHalfBath'

,'ScreenPorch'

,'BsmtFinType2'

,'EnclosedPorch'

,'MasVnrArea'

,'OpenPorchSF'

,'Fence'

,'Exterior1st'

#,'SalePrice'

,'WoodDeckSF'

,'ExterCond'

,'BsmtExposure'

,'pot_drop_BsmtUnfSF'

,'pot_drop_1stFlrSF'

,'sum_GrLivArea'

,'ExterQual'

,'pot_drop_2ndFlrSF'

,'pot_drop_BsmtFinSF1'

,'GarageAgeWhenSold'

,'OverallCond'

,'HalfBath'

,'TotRmsAbvGrd'

,'d_tot_avail_SF'

,'Fireplaces'

,'AgeWhenSold'

,'BsmtFullBath'

,'LotFrontage'

,'YrsSinceRemodel'

,'sum_TotalBsmtSF'

,'KitchenQual'

,'MoSold'

,'BedroomAbvGr'

,'Neighborhood'

,'OverallQual'

,'GarageArea'

,'GarageFinish'

,'FireplaceQu'

,'YrSold'

,'FullBath'

,'Id'

,'Exterior2nd'

,'BsmtFinType1'

,'GarageCars'

,'HeatingQC'

,'BsmtQual'

,'PavedDrive'

,'GarageQual'

,'GarageCond'

,'CentralAir'

,'BsmtCond'

,'Functional']
cols_to_drop = ['Id','YrSold','MoSold','PoolQC','PoolArea','LotArea','3SsnPorch']

skewed_df = df_train[cols_to_keep].drop(cols_to_drop, axis=1)

skewed_df.columns
pt = PowerTransformer() # the default method is yeo-johnson, which works with both positive and negatives variables; box-cox would only work with positive variables.

cols_to_transform = skewed_df.columns

#df_train[cols_to_transform] = pd.DataFrame(pt.fit_transform(df_train[cols_to_transform]), columns=cols_to_transform)

df_train.head(5)
df_train.SalePrice = np.log1p(df_train.SalePrice)
calc_missing_data(df_train, df_test)
# df_train=df_train.dropna()
plt.hist(df_train.SalePrice, bins=15)

plt.xlabel('Sale price, $')

plt.ylabel('# Houses')

sns.distplot(df_train['SalePrice'], fit=norm, kde=False, color='red')



fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
# df_train[cols_to_keep].drop(cols_to_drop, axis=1).skew(axis = 0).sort_values(ascending=False) 
corr = df_train.corr()

f, ax = plt.subplots(figsize=(40, 20))

sns.heatmap(corr, vmax=.8,annot_kws={'size': 10}, cmap='coolwarm', annot=True)
s = corr.unstack()

s[(abs(s)>0.6) & (abs(s) < 1)]
s = corr.unstack()

s[(abs(s)>0) & (abs(s) < 0.10)]["SalePrice"]
def drop_features(df):

    cols_to_drop = ['sum_GrLivArea','sum_TotalBsmtSF','pot_drop_1stFlrSF','GarageArea','ExterQual','Fireplaces','GarageCond','BsmtFinType2','BsmtHalfBath','OverallCond',

                   'ExterCond','3SsnPorch','PoolArea','PoolQC','LotFrontage','pot_drop_BsmtFinSF1','BedroomAbvGr', 'Exterior2nd']

    df.drop(cols_to_drop, axis=1, inplace=True)

    return df
#df_train = drop_features(df_train)

#df_test = drop_features(df_test)

#df_all = drop_features(df_all)
# let's check that the datasets have the number of fields we expect

check_df_shape(df_train,df_test,df_all)
# saleprice correlation matrix

plt.figure(figsize=(20,10))

corrmat = df_train.corr()

# picking the top N correlated features

cols = corrmat.nlargest(20, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
fig, ax = plt.subplots()

ax.scatter(x = df_train.OverallQual, y = df_train.SalePrice)

plt.ylabel('Sale Price, $', fontsize=13)

plt.xlabel('Overall Quality of the House', fontsize=13)

plt.xticks(range(0, 10))
# adding random noise for a more clear visualization of the values

sns.lmplot(x="OverallQual", y="SalePrice", data=df_train, x_jitter=0.1)
# confidence interval

sns.lmplot(x="OverallQual", y="SalePrice", data=df_train, x_estimator=np.mean)
fig, ax = plt.subplots()

ax.scatter(x = df_train.GarageCars, y = df_train.SalePrice)

plt.ylabel('Sale Price, $', fontsize=13)

plt.xlabel('Number of cars in a garage', fontsize=13)

plt.xticks(range(0, 5))

plt.show()
sns.lmplot(x="GarageCars", y="SalePrice", data=df_train, x_jitter=.05);
with sns.color_palette('Dark2'):

    with sns.plotting_context('poster'):

        with sns.axes_style('dark',{'axes.facecolor':'pink'}):

            fig, ax = plt.subplots(figsize=(20, 8))

            sns.boxplot(x='GarageCars',y='SalePrice',data=df_train,ax=ax)

            ax.tick_params(axis='x', labelrotation=45)

#fig.savefig('/tmp/box.png', dpi=300)
df_train.GarageCars.corr(df_train.d_tot_avail_SF )

# numbers of cars that fit in a garage does correlate with size of the house, although the fact that it correlates strongly with the price as we saw separately above indicates its independent value to buyers
sns.lmplot(x='d_tot_avail_SF',y='SalePrice',data=df_train)
fig, ax = plt.subplots()

ax.scatter(x = df_train.AgeWhenSold, y = df_train.SalePrice)

plt.ylabel('Sale Price, $', fontsize=13)

plt.xlabel('Age When Sold', fontsize=13)
plt.figure(figsize=(20,10))

sns.boxplot(df_train.AgeWhenSold, df_train.SalePrice)

plt.ylabel('Sale Price, $', fontsize=13)

plt.xlabel('Age When Sold', fontsize=13)
df_train.AgeWhenSold.corr(df_train.SalePrice)
fig, ax = plt.subplots()

ax.scatter(x = df_train.AgeWhenSold, y = df_train.GarageAgeWhenSold)

plt.ylabel('Garage Age When Sold', fontsize=13)

plt.xlabel('Age When Sold', fontsize=13)
fig, ax = plt.subplots()

ax.scatter(x = df_train.YrsSinceRemodel, y = df_train.SalePrice)

plt.ylabel('Sale Price, $', fontsize=13)

plt.xlabel('Years Since Remodel', fontsize=13)

plt.show()
check_df_shape(df_train,df_test,df_all)
df_train = pd.get_dummies(df_train, prefix_sep='_',drop_first=True)

df_test = pd.get_dummies(df_test, prefix_sep='_',drop_first=True)
check_df_shape(df_train,df_test,df_all)
df_train.head()
df_train_to_check = df_train.drop('SalePrice', axis=1)

set(df_train_to_check.columns)==set(df_test.columns)
# saleprice correlation matrix

plt.figure(figsize=(20,10))

corrmat = df_train.corr()

# identify top N correlated features

cols = corrmat.nlargest(25, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
corr = df_train.corr()

s = corr.unstack()

s[(abs(s)>0.6) & (abs(s) < 1)]
def drop_features(df):

    cols_to_drop = ['HouseStyle_1Story','HouseStyle_2Story']

    df.drop(cols_to_drop, axis=1, inplace=True)

    return df
#df_train = drop_features(df_train)

#df_test = drop_features(df_test)
df_train.drop('Id', axis=1, inplace=True)

#df_test.drop('Id', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('SalePrice', axis=1), df_train['SalePrice'], test_size=0.25, random_state=42)
scaler = StandardScaler()

df_train_c = df_train.copy()
df_train_c = pd.DataFrame(scaler.fit_transform(df_train_c), columns=df_train_c.columns) 



#X_train.loc[:, df_train.columns != 'SalePrice'] = scaler.fit_transform(X_train.loc[:, df_train.columns != 'SalePrice'])

# df_train.drop('SalePrice', axis=1) = pd.DataFrame(scaler.fit_transform(df_train),columns = df_train.columns) # have to convert into DataFrame as it returns a numpy array otherwise

#X_test = pd.DataFrame(scaler.transform(dX_test),columns = df_test.columns)
df_train_c.describe(include=[np.number], percentiles=[.5]).transpose().drop("count", axis=1)
pca = decomposition.PCA()
# pca = PCA(0.95) choose principal components such that 95% of variance is retained

pca_X = pd.DataFrame(pca.fit_transform(df_train_c), columns=[f'PC{i+1}' for i in range(len(df_train_c.columns))])
pca.explained_variance_ratio_
pca.components_[0]
(pd.DataFrame(pca.components_, columns=df_train_c.columns).iloc[:2].plot.bar().legend(bbox_to_anchor=(1,1)))
(pd.DataFrame(pca.components_, columns=df_train_c.columns).iloc[2:4].plot.bar().legend(bbox_to_anchor=(1,1)))
# Cumulative explained variance

cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)



# Plot cumulative explained variance

plt.plot(range(len(cumulative_explained_variance)), cumulative_explained_variance)
scorer = make_scorer(mean_squared_error, greater_is_better = False) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html



def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 5))

    return rmse



def rmse_cv_test(model):

    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 5))

    return rmse
# Fit a linear regression model

linreg = LinearRegression()

linreg.fit(X_train, y_train)
# Look at predictions on training and validation set

print("RMSE training dataset: ", rmse_cv_train(linreg).mean())

print("RMSE testing dataset: ", rmse_cv_test(linreg).mean())
# Create Predictions

y_train_pred = linreg.predict(X_train)

y_test_pred = linreg.predict(X_test)
# Predictions

plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear Regression -- No Penalty")

plt.xlabel("Predicted values")

plt.ylabel("Actual")

plt.legend(loc = "lower right")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
# Residuals

plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear Regression -- No Penalty")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "best")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
# Fit an L1 regularization linear regression model

lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 

                          0.3, 0.6, 1], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)
alpha = lasso.alpha_

print("Best alpha :", alpha)
print("Try again for more precision with alphas centered around " + str(alpha))

lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 

                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 

                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 

                          alpha * 1.4], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)
print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())

print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())

y_train_las = lasso.predict(X_train)

y_test_las = lasso.predict(X_test)
# Predictions

plt.scatter(y_train_las, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_las, y_test, c = "lightgreen", marker = "s", label = "Validation data")



plt.title("Linear regression | Lasso L1")

plt.xlabel("predicted values")

plt.ylabel("Actuals")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
# Plot residuals

plt.scatter(y_train_las, y_train_las - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_las, y_test_las - y_test, c = "lightgreen", marker = "s", label = "Validation data")



plt.title("Linear regression | Lasso L1")

plt.xlabel("predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper right")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
# Coefs

coefs = pd.Series(lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Lasso Coefficients")
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(X_train, y_train)
alpha = ridge.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 

                cv = 10)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)



print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())

print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())





y_train_rdg = ridge.predict(X_train)

y_test_rdg = ridge.predict(X_test)



# Predictions

plt.scatter(y_train_rdg, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_rdg, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression | Ridge L2")

plt.xlabel("Predicted values")

plt.ylabel("Actuals")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")



# Residuals

plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression | Ridge L2")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")





# Coefficients

coefs = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated other " +  \

      str(sum(coefs == 0)) + " features.")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Ridge Coefficients")
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 

                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)
alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Try again for more precision with l1_ratio centered around " + str(ratio))

elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 

      " and alpha centered around " + str(alpha))

elasticNet = ElasticNetCV(l1_ratio = ratio,

                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 

                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 

                                    alpha * 1.35, alpha * 1.4], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())

print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())

y_train_ela = elasticNet.predict(X_train)

y_test_ela = elasticNet.predict(X_test)



# Plot residuals

plt.scatter(y_train_ela, y_train_ela - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_ela, y_test_ela - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with ElasticNet regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train, y_train_ela, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test, y_test_ela, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with ElasticNet regularization")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()



# Plot important coefficients

coefs = pd.Series(elasticNet.coef_, index = X_train.columns)

print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the ElasticNet Model")

plt.show()
df_test.shape
df_train.head(5)
#ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

df_test_c = df_test.copy()

df_test.drop('Id',axis=1, inplace=True)

prediction = ridge.predict(df_test)

prediction = pd.DataFrame(prediction)

prediction.head()
# Export to a .csv file:

os.chdir('/kaggle/working')

prediction.to_csv('ames_prediction.csv')

FileLink('ames_prediction.csv')