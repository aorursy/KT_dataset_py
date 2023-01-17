import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import os
print(os.listdir('../input'))
df_train = pd.read_csv('../input/train.csv')
df_train.info()
df_train.head(20)
df_train.columns
#check if there any zero in minimal of the price
df_train['SalePrice'].describe()
#https://seaborn.pydata.org/tutorial/distributions.html#plotting-univariate-distributions
sns.distplot(df_train['SalePrice'], bins=20, rug=True);
#Skew = ambience distributions data (0=evenly distributed)
#Kurt = to check the outlier data (3=standart value)
print("Skewness: %f" %df_train['SalePrice'].skew())
print("Kurtosis: %f" %df_train['SalePrice'].kurt())
#Annotation Heatmap https://seaborn.pydata.org/examples/heatmap_annotation.html

Heatmap_Annotation = df_train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(Heatmap_Annotation, 
            vmax=.8, square=True)
#Diagonal Correlation https://seaborn.pydata.org/examples/many_pairwise_correlations.html

sns.set(style='white')
Diagonal_Corr = df_train.corr()
mask = np.zeros_like(Diagonal_Corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(Diagonal_Corr, mask=mask, cmap=cmap,
           vmax=.8, center=0,square=True,
           linewidths=5, cbar_kws={"shrink":.5})
#SalePrice Correlation Matrix
k=10
sns.set(font_scale=1.25)
corrmat=df_train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.heatmap(cm, annot=True, square=True,
           fmt='.2f', annot_kws={'size':10},
           yticklabels=cols.values,
           xticklabels=cols.values)

#Scatterplot https://seaborn.pydata.org/tutorial/regression.html
cols = ['SalePrice', 'OverallQual','GrLivArea',
       'GarageCars', 'GarageArea', 'TotalBsmtSF',
       '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
       'YearBuilt']
sns.pairplot(df_train[cols], size =5)
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = ((df_train.isnull().sum()/df_train.isnull().count())*100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
f, ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation='90')
sns.barplot(x=total.index, y=total)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values',fontsize=15)
plt.title('Percent missing data by feature',fontsize=15)
#drop columns/keys that have more than 50% of null values
df_train = df_train.drop((missing_data[missing_data['Percent'] > 50 ]).index,1)
df_train.isnull().sum().sort_values(ascending=False) #check
#FireplaceQu : data description says Null means "no fireplace"
df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 
            'BsmtFinType1', 'BsmtFinType2',
            'GarageType', 'GarageFinish', 'GarageQual', 
            'GarageCond'):
    df_train[col] = df_train[col].fillna('None')
#GarageYrBlt replacing missing data with 0
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna(0)
df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")
df_train["MasVnrArea"] = df_train["MasVnrArea"].fillna(0)
#Electrical : It has one NA value. 
#Since this feature has mostly 'SBrkr', we can set that for the missing value.
df_train['Electrical'] = df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])
df_train.isnull().sum().sort_values(ascending=False) #check
#SalePrice Correlation Matrix
k=10
sns.set(font_scale=1.5)
corrmat=df_train.corr()
cols = corrmat.nsmallest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.heatmap(cm, annot=True, square=True,
           fmt='.2f', annot_kws={'size':10},
           yticklabels=cols.values,
           xticklabels=cols.values)



#deleting uncorrelate colomns
Uncor = ['EnclosedPorch', 'OverallCond', 
        'YrSold', 'LowQualFinSF', 'Id', 
         'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2']
df_train.drop(Uncor, axis=1, inplace=True)
df_train.info()
#More features engineering
#Transforming some numerical variables that are really
df_train['MSSubClass'] = df_train['MSSubClass'].astype(str)
df_train['MoSold'] = df_train['MoSold'].astype(str)
# Adding total sqfootage feature 
df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
#Univariate analysis
#Detect and exclude outlier in numeric dtype
#low 0.05 and high 0.90 quantile
from pandas.api.types import is_numeric_dtype
def remove_outlier(df_train):
    low = .05
    high = .90
    quant_df = df_train.quantile([low, high])
    for name in list(df_train.columns):
        if is_numeric_dtype(df_train[name]):
            df_train = df_train[(df_train[name] > quant_df.loc[low, name]) & (df_train[name] < quant_df.loc[high, name])]
    return df_train

remove_outlier(df_train).head()
#check the standardizing data
for name in list(df_train.columns):
    if is_numeric_dtype(df_train[name]):
        saleprice_scaled = StandardScaler().fit_transform(df_train[name][:,np.newaxis]);
        low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:5]
        high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-5:]
        print('outer range (low) of the distribution:',name)
        print(low_range)
        print('\nouter range (high) of the distribution:',name)
        print(high_range)
#Bivariate/Multivariate outlier checking with scatter plot
for name in list(df_train.columns):
    if is_numeric_dtype(df_train[name]):
        data = pd.concat([df_train['SalePrice'], df_train[name]], axis=1)
        data.plot.scatter(x=name, y='SalePrice', ylim=(0,800000))
#Dropping the outlier
#Only on the Feature that perform linear regression dot in the scatter plot
df_train = df_train.drop(df_train[df_train['LotFrontage'] > 300].index)
df_train = df_train.drop(df_train[df_train['LotArea'] > 60000].index)
df_train = df_train.drop(df_train[(df_train['OverallQual'] > 9) & (df_train['SalePrice'] < 200000)].index)
df_train = df_train.drop(df_train[df_train['MasVnrArea'] > 1500].index)
df_train = df_train.drop(df_train[df_train['TotalBsmtSF'] > 3000].index)
df_train = df_train.drop(df_train[df_train['1stFlrSF'] > 2500].index)
df_train = df_train.drop(df_train[df_train['BsmtFullBath'] > 2.5].index)
df_train = df_train.drop(df_train[df_train['HalfBath'] > 1.5].index)
df_train = df_train.drop(df_train[df_train['BedroomAbvGr'] > 4].index)
df_train = df_train.drop(df_train[df_train['TotRmsAbvGrd'] > 13].index)
df_train = df_train.drop(df_train[df_train['Fireplaces'] > 2.5].index)
df_train = df_train.drop(df_train[df_train['GarageCars'] > 3].index)
df_train = df_train.drop(df_train[df_train['GarageArea'] >= 1250].index)

#skewed features
numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index
# Check the skew of all numerical features
skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(20)
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #df_train[feat] += 1
    df_train [feat] = boxcox1p(df_train[feat], lam)
    
#df_train[skewed_features] = np.log1p(df_train[skewed_features])
#check
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
from sklearn.preprocessing import LabelEncoder
colomns = df_train.dtypes[df_train.dtypes == "object"].index
# process columns, apply LabelEncoder to categorical features
for name in colomns:
    lbl = LabelEncoder() 
    lbl.fit(list(df_train[name].values)) 
    df_train[name] = lbl.transform(list(df_train[name].values))

# shape        
print('Shape of df_train: {}'.format(df_train.shape))
#Dummy categorical features
df_train = pd.get_dummies(df_train)
print(df_train.shape)
df_train.head(20) #please compare the data after engineering and before engineering