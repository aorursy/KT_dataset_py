import numpy as np

import pandas as pd

import statsmodels.api as sm

from scipy import stats

from scipy.stats import skew 

from scipy.special import boxcox1p

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn import metrics

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge, Lasso



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

%matplotlib inline

sns.set(style='darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn 

warnings.filterwarnings('ignore')

print ('All relevant packages imported successfully')
#df_train = pd.read_csv('train.csv')

#df_test = pd.read_csv('test.csv')

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

print('Data imported')





print('Checking shape of datasets')

print("The are {} rows of data in train dataset and {} columns in training dataset". format(str(df_train.shape[0]), str(df_train.shape[1])))

print("The are {} rows of data in train dataset and {} columns in training dataset". format(str(df_test.shape[0]), str(df_test.shape[1])))



# Lets create ID variables for safe-keeping

train_ID = df_train.iloc[:,0]

test_ID = df_test.iloc[:,0]
df_train.head()
unique_ID = len(set(df_train.Id))

total_ID = df_train.shape[0]

duplicate_ID = total_ID - unique_ID

print("There are " + str(duplicate_ID) + " duplicate IDs for " + str(total_ID) + " total rows")
df_train['SalePrice'].describe().to_frame()
plt.figure(figsize=(16, 9))

plt.title('SalePrice Distribution', fontsize=24)

sns.distplot(df_train['SalePrice'], color='darkviolet')

plt.ylabel('Frequency', fontsize=18)

plt.xlabel('SalePrice (USD)', fontsize=18)

# plt.savefig('figures/sale_price_distribution_skewed.png')
print("Skewness of SalePrice: %f" % df_train['SalePrice'].skew())

print("Kurtosis of SalePrice: %f" % df_train['SalePrice'].kurt())
df_train['GrLivArea'].describe().to_frame()
plt.figure(figsize=(16, 9))

plt.title('Above Ground Living Area Distribution', fontsize=24)

sns.distplot(df_train['GrLivArea'], color='darkviolet')

plt.xlabel('Above Ground Living Area (square feet)', fontsize=18)

plt.ylabel('Frequency', fontsize=18)

# plt.savefig('figures/grade_living_area_distribution_skewed.png')
print("Skewness: %f" % df_train['GrLivArea'].skew())

print("Kurtosis: %f" % df_train['GrLivArea'].kurt())
plt.figure(figsize=(16, 9))

sns.regplot(df_train['GrLivArea'], df_train['SalePrice'], color = 'darkviolet' , fit_reg= False)

plt.title('Above Ground Living Area vs. Sale Price', fontsize=24)

plt.xlabel("Above Ground Living Area (Sqaure Feet", fontsize=18)

plt.ylabel("Sale Price (USD)", fontsize=18)

# plt.savefig('figures/GrLivArea_vs_SalePrice_scatter.png')
df_train['TotalBsmtSF'].describe().to_frame()
plt.figure(figsize=(16, 9))

plt.title('Total Square Feet of Basement Area Distribution', fontsize=24)

plt.xlabel('Total Square Feet of Basement Area', fontsize=18)

plt.ylabel('Frequency', fontsize=18)

sns.distplot(df_train['TotalBsmtSF'], color='darkviolet')

# plt.savefig('figures/TotalBsmtSF_distribution_skewed.png')
print("Skewness: %f" % df_train['TotalBsmtSF'].skew())

print("Kurtosis: %f" % df_train['TotalBsmtSF'].kurt())
plt.figure(figsize=(16, 9))

sns.regplot(df_train['TotalBsmtSF'], df_train['SalePrice'], color = 'darkviolet' , fit_reg= False)

plt.title('Total Basement Area (SF) vs. Sale Price', fontsize=24)

plt.xlabel('Total Basement Area (Sqaure Feet)', fontsize=18)

plt.ylabel("Sale Price (USD)", fontsize=18)

# plt.savefig('figures/TotalBsmtSF_vs_SalePrice_scatter.png')
#First off lets create a total bathrooms variable

df_train["TotalBathroom"] = df_train["BsmtFullBath"] + (0.5 * df_train["BsmtHalfBath"]) + df_train["FullBath"] + (0.5 * df_train["HalfBath"])

df_train["TotalBathroom"].describe()
plt.figure(figsize=(16, 9))

plt.title('Sale Price vs. Total Number of Bathrooms', fontsize=24)

sns.boxplot(df_train["TotalBathroom"], df_train['SalePrice'], linewidth=1.0, palette='Purples')

plt.xlabel('Number of Bathrooms', fontsize=18)

plt.ylabel('Sale Price (USD)', fontsize=18)

plt.tight_layout()

# plt.savefig('figures/TotalBathrooms_vs_SalePrice_boxplot.png')
df_train["BedroomAbvGr"].describe()
plt.figure(figsize=(16, 9))

plt.title('Sale Price vs. Number of Bedrooms Above Ground', fontsize=24)

fig = sns.boxplot(df_train['BedroomAbvGr'], df_train['SalePrice'], palette='Purples', linewidth=1.0)

plt.xlabel('Bedrooms Above Ground', fontsize=18)

plt.ylabel('Sale Price (USD)', fontsize=18)

plt.tight_layout()

# plt.savefig('figures/BedroomAbvGr_vs_SalePrice_boxplot.png')
plt.subplots(figsize=(16, 9))

median_prices=pd.DataFrame(df_train.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False))

sns.barplot(x=median_prices.index, y=median_prices['SalePrice'], palette='Purples_r', linewidth=1.0, edgecolor=".2")

plt.title('Median Sale Price In Different Neighbourhoods', fontsize=24)

plt.xlabel('Neighbourhood', fontsize=18)

plt.ylabel('Sale Price (USD)', fontsize=18)

plt.xticks(rotation=90)

plt.tight_layout()

# plt.savefig('figures/Neighborhood_vs_SalePrice_boxplot.png')
plt.figure(figsize=(16, 9))

plt.title('Zone Classification vs. Sale Price', fontsize=24)

sns.boxplot(df_train["MSZoning"], df_train['SalePrice'], palette='Purples', linewidth=1.0)

plt.xlabel('Zone Classification', fontsize=18)

plt.ylabel('Sale Price (USD)', fontsize=18)

# plt.savefig('figures/MSZoning_vs_SalePrice_boxplot.png')
plt.subplots(figsize=(16, 9))

sns.boxplot(df_train['YearBuilt'], df_train['SalePrice'], palette='Purples', linewidth=1.0)

plt.title('Sale Price vs Construction Year of House Built', fontsize=24)

plt.xlabel('Year Built', fontsize=18)

plt.ylabel('Sale Price (USD)', fontsize=18)

plt.xticks(rotation=90)

plt.tick_params(labelsize=9)

# plt.savefig('figures/YearBuilt_vs_SalePrice_boxplot.png')
plt.subplots(figsize=(16, 9))

sns.boxplot(df_train['YrSold'], df_train['SalePrice'], palette='Purples', linewidth=0.8)

plt.title('Sale Price vs. Year Sold', fontsize=24)

plt.xlabel('Year Sold', fontsize=18)

plt.ylabel('Sale Price (USD)', fontsize=18)

# plt.savefig('figures/YrSold_vs_SalePrice_boxplot.png')
plt.subplots(figsize=(16, 9))

sns.boxplot(df_train['OverallQual'], df_train['SalePrice'], palette='Purples', linewidth=0.8)

plt.title('Overall Quality of House vs. Sale Price', fontsize=24)

plt.xlabel('Overall Quality of House', fontsize=18)

plt.ylabel('Sale Price', fontsize=18)

# plt.savefig('figures/OverallQual_vs_SalePrice_boxplot.png')
plt.subplots(figsize=(16, 9))

sns.boxplot(df_train['OverallCond'], df_train['SalePrice'], palette='Purples', linewidth=1.0)

plt.title('Overall Condition of House vs. Sale Price', fontsize=24)

plt.xlabel('Overall Condition of House', fontsize=18)

plt.ylabel('Sale Price (USD)', fontsize=18)

# plt.savefig('figures/OverallCond_vs_SalePrice_boxplot.png')
corr = df_train.corr()['SalePrice'].sort_values(ascending=False).head(20).to_frame()

plt.subplots(figsize=(16, 9))

sns.barplot(x=corr.index, y=corr['SalePrice'], palette='Purples_r', linewidth=1.0, edgecolor=".2")

plt.title('Top 20 Features Most Correlated With Sale Price', fontsize=24)

plt.xlabel('Feature', fontsize=18)

plt.ylabel('Coefficint of Correlation with Sale Price (USD)', fontsize=18)

plt.xticks(rotation=90)

plt.tight_layout()

# plt.savefig('figures/Neighborhood_vs_SalePrice_boxplot.png')
# Displays A Scatter Plot with Regression Line + Outliers in Red

x_axis = 'GrLivArea'

y_axis = 'SalePrice'



value=(df_train[x_axis]>4000) | (df_train[y_axis]>621000)



plt.figure(figsize=(16, 9))

sns.regplot(df_train['GrLivArea'], df_train['SalePrice'], fit_reg=True,

            scatter_kws={'facecolors':np.where( value==True , "red", 'darkviolet'),

           'edgecolor':np.where( value==True , "red", 'darkviolet')},

           line_kws = {'color': 'darkblue'})

plt.title('Above Ground Living Area (SF) vs. Sale Price', fontsize=24)

plt.xlabel('Above Ground Living Area (Square Feet)', fontsize=18)

plt.ylabel('Sale Price (USD)', fontsize=18)

plt.tight_layout()

# plt.savefig('figures/SalePrice_vs_GrLivArea_regline.png')
df_train['Above Ground Standardised'] = np.abs(stats.zscore(df_train.GrLivArea))

df_train['Above Ground Standardised'].nlargest(10).to_frame()
df_train['Selling Price Standardised'] = np.abs(stats.zscore(df_train.SalePrice))

df_train[df_train['Above Ground Standardised']>4]
# Displays A Scatter Plot with Regression Line + Outliers in Red

x_axis = 'Above Ground Standardised'

y_axis = 'Selling Price Standardised'

value=(df_train[x_axis]>4)



plt.figure(figsize=(16, 9))

sns.regplot(df_train['Above Ground Standardised'], df_train['Selling Price Standardised'], fit_reg=True,

            scatter_kws={'facecolors':np.where( value==True , "red", 'darkviolet'),

           'edgecolor':np.where(value==True , "red", 'darkviolet')},

           line_kws = {'color': 'darkblue'})

plt.title('Above Ground Living Area (Standardised) vs. Sale Price (Standardised)', fontsize=24)

plt.xlabel('Standardised Above Ground Living Area', fontsize=18)

plt.ylabel('Standardised Sale Price (USD)', fontsize=18)

# plt.savefig('figures/Standardised_SalePrice_GrLivArea.png')
df_train = df_train.drop(df_train[(df_train['Above Ground Standardised']> 4)].index)

print('Outliers dropped successfully')
# Displays A Scatter Plot with Regression Line + Outliers in Red

plt.figure(figsize=(16, 9))

sns.regplot(df_train["GrLivArea"], df_train["SalePrice"], fit_reg=True,

            scatter_kws={'facecolors':'darkviolet',

           'edgecolor':'darkviolet'},

           line_kws = {'color': 'darkblue'})

plt.title('Above Ground Living Area vs. Sale Price (After Dropping Outliers)', fontsize=24)

plt.xlabel('Above Ground Living Area', fontsize=18)

plt.ylabel('Sale Price (USD)', fontsize=18)
plt.figure(figsize=(16, 9))



# plot

sns.regplot(df_train['TotalBsmtSF'], df_train['SalePrice'], fit_reg=True, 

            scatter_kws={'facecolors':'darkviolet',

           'edgecolor':'darkviolet'},

           line_kws = {'color': 'darkblue'})

plt.title('Total Basement Area (SF) vs. Sale Price ', fontsize=24)

plt.xlabel('Total Basement Area (in Square Feet)', fontsize=18)

plt.ylabel('Sale Price (USD)', fontsize=18)

# plt.savefig('figures/Basement_SalePrice_scatter.png')
#Drop all constructed variables for now

df_train = df_train.drop(['Above Ground Standardised','Selling Price Standardised', "TotalBathroom"], axis = 1)
#Create a place holder for where the train dataset ends and the test set begins

df_train_ends = df_train.shape[0]



#Duplicate our train and test sets to ensure we don't mistakenly overwrite them

df_train1 = df_train.copy()

df_test1 = df_test.copy()

df_total = pd.concat((df_train1, df_test1))

df_total = df_total.drop('Id', axis = 1 )
print("There are {} anomalies in the YearBuilt variable". format(df_total[(df_total['YearBuilt'] > 2010) | (df_total['YearBuilt'] < 1858)]['YearBuilt'].count()))

print("There are {} anomalies in the YearRemodAdd variable". format(df_total[(df_total['YearRemodAdd'] > 2010) | (df_total['YearRemodAdd'] < 1858)]['YearRemodAdd'].count()))

print("There are {} anomalies in the GarageYrBlt variable". format(df_total[(df_total['GarageYrBlt'] > 2010) | (df_total['GarageYrBlt'] < 1858)]['GarageYrBlt'].count()))

print("There are {} anomalies in the YrSold variable". format(df_total[(df_total['YrSold'] > 2010) | (df_total['YrSold'] < 2006)]['YrSold'].count()))
df_total[(df_total['GarageYrBlt'] > 2010) | (df_total['GarageYrBlt'] < 1858)][['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']]
#It is clear that 2207 is supposed to be 2007

df_total.loc[1132,'GarageYrBlt'] = '2007'
df_total_na = (df_total.isnull().sum() / len(df_total)) * 100

df_total_na = df_total_na.drop(df_total_na[df_total_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Percentage of data missing' :df_total_na})

missing_data.head(20)
# Horizontal bar graph

plt.figure(figsize=[16, 9])

plt.title('Percentage of Missing Values Per Feature', fontsize=24)

sns.barplot(missing_data['Percentage of data missing'], y=missing_data.index, data=missing_data, 

            palette='Purples_r', linewidth=1.0, edgecolor=".2")

plt.xlabel('Percentage of Total Missing ', fontsize=18)

plt.ylabel('Feature', fontsize=18)
df_total['LotFrontage'] = df_total.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df_total['MSZoning'] = df_total['MSZoning'].fillna(df_total['MSZoning'].mode()[0])
df_total['MSSubClass'] = df_total.groupby('Neighborhood')['MSSubClass'].transform(lambda x: x.fillna(x.mode()))
df_total['GarageYrBlt'] = df_total['GarageYrBlt'].fillna(df_total['YearBuilt'])
modes = ["Functional", 'Electrical', "Utilities",'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']

for col in modes:

    df_total[col] = df_total[col].fillna(df_total[col].mode()[0])
cats = ["PoolQC", "Fence", "MiscFeature", "Alley", "FireplaceQu", "MasVnrType", 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for col in cats:

    df_total[col] = df_total[col].fillna('None')     
nums = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']

for col in nums:

    df_total[col] = df_total[col].fillna(0)

print('Finished imputing')
df_total_na = (df_total.isnull().sum() / len(df_total)) * 100

df_total_na = df_total_na.drop(df_total_na[df_total_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Percentage missing' :df_total_na})

missing_data.head()
df_total['MSSubClass'] = df_total['MSSubClass'].apply(str)



#Year and month sold are transformed into categorical features.

df_total['YrSold'] = df_total['YrSold'].astype(str)

df_total['MoSold'] = df_total['MoSold'].astype(str)
df_total = df_total.replace({"BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"None" : 0, "No" : 1, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 1, "Rec" : 2, "BLQ" : 3, 

                                         "ALQ" : 4, "GLQ" : 5},

                       "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 1, "Rec" : 2, "BLQ" : 3, 

                                         "ALQ" : 4, "GLQ" : 5},

                       "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                        'GarageFinish' : {"None" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                        "PoolQC" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}, 

                        "CentralAir" : {"N" : 0, "Y" : 1}})
df_total = df_total.replace({"Alley" : {"None" :0, "Grvl" : 1, "Pave" : 2}, "PavedDrive" : {"N" : 1, "P" : 2, "Y" : 3},

                             "Street" : {"Grvl" : 1, "Pave" : 2}})
df_total = df_total.replace({"LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3} , 

                             "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4}})
df_total = df_total.replace({"Functional" : {"Sal" : 0, "Sev" : 1, "Maj2" : 2, "Maj1" : 2, "Mod":3, 

                                       "Min2" : 4, "Min1" : 4, "Typ" : 5}})
df_total = df_total.replace({"Electrical" : {"FuseP" : 1, "FuseF" : 2, "FuseA" : 3, "Mix" : 4, "SBrkr" : 5},

                            'Utilities': {'NoSeWa':0, "AllPub": 1}})
df_total['OverallQual_simple'] = df_total.OverallQual.replace({1:1, 2:1, 3:1, # bad quality

                                                        4:2, 5:2, 6:2, # mediocre quality

                                                        7:3, 8:3, 9:3, 10:3 # good quality

                                                       })

df_total['OverallCond_simple'] = df_total.OverallCond.replace({1:1, 2:1, 3:1, # bad quality

                                                        4:2, 5:2, 6:2, # mediocre quality

                                                        7:3, 8:3, 9:3, 10:3 # good quality

                                                       })

# convert from categorical to ordinal with smaller groups

df_total['ExterQual_simple'] = df_total.ExterQual.replace({5:3, 4:3, 3:2, 2:2, 1:1})

df_total['ExterCond_simple'] = df_total.ExterCond.replace({5:3, 4:3, 3:2, 2:2, 1:1})

df_total['BsmtQual_simple'] = df_total.BsmtQual.replace({5:3, 4:3, 3:2, 2:2, 1:1})

df_total['BsmtCond_simple'] = df_total.BsmtCond.replace({5:3, 4:3, 3:2, 2:2, 1:1})

df_total['BsmtFinType1_simple'] = df_total.BsmtFinType1.replace({6:3, 5:3, 4:2, 3:2, 2:1, 1:1})

df_total['BsmtFinType2_simple'] = df_total.BsmtFinType2.replace({6:3, 5:3, 4:2, 3:2, 2:1, 1:1})

df_total['HeatingQC_simple'] = df_total.HeatingQC.replace({5:3, 4:3, 3:2, 2:2, 1:1})

df_total['KitchenQual_simple'] = df_total.KitchenQual.replace({5:3, 4:3, 3:2, 2:2, 1:1})

df_total['Functional_simple'] = df_total.Functional.replace({8:4, 7:3, 7:3, 6:3, 5:2,4:2, 3:1, 2:1})

df_total['GarageQual_simple'] = df_total.GarageQual.replace({5:3, 4:3, 3:2, 2:2, 1:1})

df_total['GarageCond_simple'] = df_total.GarageCond.replace({5:3, 4:3, 3:2, 2:2, 1:1})

df_total['PoolQC_simple'] = df_total.PoolQC.replace({5:3, 4:3, 3:2, 2:2, 1:1})

df_total['FireplaceQu_simple'] = df_total.FireplaceQu.replace({5:3, 4:3, 3:2, 2:2, 1:1})
corr = df_total.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize=(15, 11))

with sns.axes_style('dark'):

    ax = sns.heatmap(corr, mask=mask, cmap="Purples")
garage = df_total[['GarageArea','GarageCars','GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']]

corr = garage.corr()

corr[corr>0.8]
SF = df_total[['TotRmsAbvGrd','GrLivArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'TotalBsmtSF']]

corr = SF.corr()

corr[corr>0.8]
df_total['AboveSF'] = df_total['1stFlrSF'] + df_total['2ndFlrSF'] + df_total['LowQualFinSF']

df_total[['AboveSF', 'GrLivArea']].corr()
year = df_total[df_total['GarageYrBlt']!='None'][['YearBuilt','YearRemodAdd', 'GarageYrBlt']]

year['GarageYrBlt'] = year['GarageYrBlt'].astype(int)

year.corr()
basement = df_total[["BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", 'BsmtFinSF1', 'BsmtFinSF2', "BsmtQual", 'TotalBsmtSF', '1stFlrSF']]

corr = basement.corr()

corr[corr>0.8]
quality = df_total[["ExterCond", "ExterQual", "FireplaceQu", "GarageQual", "HeatingQC", "KitchenQual", "OverallQual", 'OverallCond']]

corr = quality.corr()

corr[corr>0.8]
plt.figure(figsize=(16, 9))

sns.distplot(df_train['SalePrice'], color = 'darkviolet')

plt.title('Distribution of House Sale Price (USD) ', fontsize=24)

plt.xlabel('Sale Price (USD)', fontsize=18)

plt.ylabel('Frequency', fontsize=18)

fig = plt.figure()



#Plotting the Q-Q plot

plt.figure(figsize=(16, 9))

res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

y = df_train.SalePrice



plt.figure(figsize=(16, 9))

sns.distplot(y, color = 'darkviolet')

plt.title('Distribution of Log-Transformed Sale Price ', fontsize=24)

plt.xlabel('Log of Sale Price', fontsize=18)

plt.ylabel('Frequency', fontsize=18)

fig = plt.figure()



#Plotting the Q-Q plot

plt.figure(figsize=(16, 9))

res = stats.probplot(y, plot=plt)
#Partiton the dataset into its catergorical and numerical subsets

categorical_features = df_total.select_dtypes(include = ["object"]).columns

numerical_features = df_total.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

price = df_total['SalePrice']



print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))



df_total_num = df_total[numerical_features]

df_total_cat = df_total[categorical_features]
#Create a skewness measure of all variables in numerical set

skewness = df_total_num.apply(lambda x: skew(x))



#We assume that any variable that exhibits skewness greater than 0.5 is likely to pose problems and needs to transformed

skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + " skewed numerical features need to be transform")

skewed_features = skewness.index



# The optimal lambda used here was taken from the calculations used by other notebooks.

df_total_num[skewed_features] = boxcox1p(df_total_num[skewed_features], 0.15)
df_total = pd.concat([df_total_num, df_total_cat, price], axis = 1)
print('There are currently {} total variables'.format(df_total.shape[1]))
#Total living area - above and below ground living area available

df_total['HouseSF'] = df_total["GrLivArea"] + df_total["TotalBsmtSF"]
#The total porch area available in the house as some houses may have more than type of porch

df_total["AllPorchSF"] = df_total["OpenPorchSF"] + df_total["EnclosedPorch"]  + df_total["ScreenPorch"]
#Total number of bathrooms in the house regardless of where they are (you dont care where they are when you need one)

df_total["TotalBathroom"] = df_total["BsmtFullBath"] + (0.5 * df_total["BsmtHalfBath"]) + df_total["FullBath"] + (0.5 * df_total["HalfBath"])
#Creating a measure of quality per square foot

df_total['QualperSF'] = df_total['HouseSF'] / (1/df_total['OverallQual'])



# Quality per bathroom

df_total['BathroomGrade'] = df_total["TotalBathroom"] * df_total['OverallQual']



# Quality per SF of porch

df_total['PorchGrade'] = df_total['ExterQual'] * df_total["AllPorchSF"]



#Quality per car garage

df_total['GarageSFQual'] = df_total['GarageCars'] * df_total['GarageQual'] * df_total['GarageFinish']
# Size of house might have a quadratic, cubic  or relationship with SalePrice

df_total['HouseSFSq'] = df_total['HouseSF']**2

df_total['HouseSFCb'] = df_total['HouseSF']**3
# Noted earlier that quality appeared to have an exponential relationship with SalePrice

# The use of quadratic term should capture this relationship

df_total['OverQualSq'] = df_total['OverallQual']**2 



#Same process for condition

df_total['OverCondSq'] = df_total['OverallCond']**2 
#Does the house have a pool present?

df_total['HasPool'] = df_total['PoolArea'].apply(lambda x: 1 if x > 0 else 0)



#Is the house more than 1 storey?

df_total['Has2ndFloor'] = df_total['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)



#Does the house have a garage of any type?

df_total['HasGarage'] = df_total['GarageCars'].apply(lambda x: 1 if x > 0 else 0)



#Does the house have a basement of any size?

df_total['HasBsmt'] = df_total['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)



#Does the house have any sort of fireplace present?

df_total['HasFireplace'] = df_total['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



# Does the house have a shed?

df_total["HasShed"] = (df_total["MiscFeature"] == "Shed").apply(lambda x: 1 if x > 0 else 0)
neighborhood_map = {"MeadowV" : 1, "IDOTRR" : 1, "BrDale" : 1, "OldTown" : 2, "Edwards" : 2, "BrkSide" : 2, "Sawyer" : 2, 

                    "Blueste" : 2, "SWISU" : 2, "NAmes" : 2, "NPkVill" : 2, "Mitchel" : 2, "SawyerW" : 3,  "Gilbert" : 3, 

                    "NWAmes" : 3,  "Blmngtn" : 3,  "CollgCr" : 3, "ClearCr" : 3, "Crawfor" : 3,  "Veenker" : 4, 

                    "Somerst" : 4, "Timber" : 4, "StoneBr" : 5, "NoRidge" : 5, "NridgHt" : 5,}



df_total["NeighborhoodCluster"] = df_total["Neighborhood"].map(neighborhood_map)
monthly_sales = df_train.groupby('MoSold')['SalePrice'].count()



plt.figure(figsize=(16, 9))

sns.barplot(x=monthly_sales.index, y=monthly_sales, palette='Purples', linewidth=1.0, edgecolor=".2")

plt.title('Incidence of House Sales by the Month of the Year', fontsize=24)

plt.xlabel('Month of the Year', fontsize=18)

plt.ylabel('Frequency', fontsize=18)

fig = plt.figure()
df_total["HighSeason"] = df_total["MoSold"].replace( 

        {'1': 0, '2': 0, '3': 0, '4': 1, '5': 1, '6': 1, '7': 1, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0})
# If YearRemodAdd != YearBuilt, then a remodeling took place at some point.

df_total["Remodeled"] = (df_total["YearRemodAdd"] != df_total["YearBuilt"]).apply(lambda x: 1 if x > 0 else 0)



# Did a remodeling happen in the year the house was sold?

df_total["RecentRemodel"] = (df_total["YearRemodAdd"] == df_total["YrSold"]).apply(lambda x: 1 if x > 0 else 0)
# A sale in which it is highly unlikely that the seller was not looking to maximise price

df_total["Unusual_sale"] = df_total.SaleCondition.replace(

        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})



# House completed before sale or not

df_total["BoughtOffPlan"] = df_total.SaleCondition.replace(

   {"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
corr = df_total.corr()['SalePrice'].sort_values(ascending=False).head(20).to_frame()

plt.subplots(figsize=(16, 9))

sns.barplot(x=corr.index, y=corr['SalePrice'], palette='Purples_r', linewidth=1.0, edgecolor=".2")

plt.title('Top 20 Features Most Correlated With Sale Price', fontsize=24)

plt.xlabel('Feature', fontsize=18)

plt.ylabel('Coefficint of Correlation with Log-Transformed Sale Price', fontsize=18)

plt.xticks(rotation=90)

plt.tight_layout()

# plt.savefig('figures/Neighborhood_vs_SalePrice_boxplot.png')
#Firstly, we need to drop the features we decided were likely to cause issues of multicollineraity for our models

df_total.drop(['GarageArea', 'GarageCond', 'TotRmsAbvGrd', 'AboveSF', '2ndFlrSF', 'GarageYrBlt', 'BsmtFinSF2'], 

              axis = 1, inplace = True)



#Then drop variable with little to no variation across the whole dataset

df_total = df_total.drop(['Utilities', 'Street', 'PoolQC', 'PoolArea'], axis=1)
#Select the numercial features in our dataset

numerical_features = df_total.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")
print("Pre - dummy number of features : " + str(df_total.shape[1]))
df_total = pd.get_dummies(df_total)

print("New number of features : " + str(df_total.shape[1]))
#We need to break up our total dataset back into our training and test set

df_total = df_total.drop(['SalePrice', "MSZoning_C (all)"], axis = 1)

train = df_total.iloc[:df_train_ends,:]

test = df_total.iloc[df_train_ends:,:]

print(train.shape)
# Partition the dataset in train + validation sets

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state = 0)

print("X_train : " + str(X_train.shape))

print("X_test : " + str(X_test.shape))

print("y_train : " + str(y_train.shape))

print("y_test : " + str(y_test.shape))
from sklearn.preprocessing import RobustScaler

rbstSc = RobustScaler()

X_train.loc[:, numerical_features] = rbstSc.fit_transform(X_train.loc[:, numerical_features])

X_test.loc[:, numerical_features] = rbstSc.transform(X_test.loc[:, numerical_features])
train.loc[:, numerical_features] = rbstSc.fit_transform(train.loc[:, numerical_features])

test.loc[:, numerical_features] = rbstSc.transform(test.loc[:, numerical_features])
#Create a function that will return RMSLE with a prediciton and an actual as inputs

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
from sklearn.ensemble import RandomForestRegressor

randomforest = RandomForestRegressor(n_estimators=1000, random_state=42)

randomforest.fit(X_train, y_train)



# Lets have a look at which features that the Random Forest regressor identifies as being significant 

coef = pd.Series(randomforest.feature_importances_, index = X_train.columns).sort_values(ascending=False)

coef = coef.head(10).to_frame()

coef.columns = ['Feature Significance']



plt.subplots(figsize=(16, 9))

plt.title('Random Forest: Feature Significance',fontsize=24)

plt.xlabel('Significance', fontsize=18)

plt.ylabel('Feature', fontsize=18)

plt.xticks(rotation=90)

plt.tight_layout()

sns.barplot(x='Feature Significance', y=coef.index, data=coef, palette='Purples_r', linewidth=1.0, edgecolor=".2")
# Predict the values of SalePrice for training and validation datasets

y_rf_train = randomforest.predict(X_train)

y_rf_test = randomforest.predict(X_test)



print("Random Forest RMSE on Training set :", rmsle(y_train, y_rf_train).mean())

print("Random Forest RMSE on Validation set :", rmsle(y_test, y_rf_test).mean())
plt.figure(figsize=(16, 9))



sns.regplot(x=y_train, y=y_rf_train, fit_reg=False, color = 'darkblue', label = "Training data")

sns.regplot(x=y_test, y=y_rf_test, fit_reg=False, color = 'darkviolet', label = "Validation data")



plt.title('Random Forest: Predicted vs. Actual Sale Price', fontsize=24)

plt.xlabel('Actual Sale Price', fontsize=18)

plt.ylabel('Predicted Sale Price', fontsize=18)

plt.legend(loc = "upper left")



plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'darkred')

plt.tight_layout()
#Fit the Random Forest model to full training set

randomforest = RandomForestRegressor(n_estimators=1000, random_state=42)

randomforest.fit(train, y)



# Predict the SalePrice values for the test set

y_rf = randomforest.predict(train)

print("Random Forest RMSE on Full Training set :", rmsle(y, y_rf).mean())

randforest_submission_price = randomforest.predict(test)
# Fit the OLS regression to the X_train set

reg = LinearRegression()

reg.fit(X_train.values, y_train.values)



#Create series of coefficient values 

coef = pd.Series(reg.coef_, index = X_train.columns).sort_values(ascending=False)

big_coef = coef.head(10).to_frame()

small_coef = coef.tail(10).to_frame()

coef = pd.concat([big_coef, small_coef])

coef.columns = ['Coefficient']



#Create the graph of coefficient size

plt.subplots(figsize=(16, 9))

plt.title('OLS Regression Beta Coefficients',fontsize=24)

plt.xlabel('Size of Coefficient', fontsize=18)

plt.ylabel('Feature', fontsize=18)

plt.xticks(rotation=90)

plt.tight_layout()

sns.barplot(x='Coefficient', y=coef.index, data=coef, palette='Purples_r', linewidth=1.0, edgecolor=".2")
# Predict SalePrice for both training and validation data 

y_train_pred = reg.predict(X_train)

y_test_pred = reg.predict(X_test)

print("RMSE on Training set :", rmsle(y_train.values, y_train_pred).mean())

print("RMSE on Validation set :", rmsle(y_test.values, y_test_pred).mean())
plt.figure(figsize=(16, 9))



# Overlay the scatter plots

sns.regplot(x=y_train, y=y_train_pred, fit_reg=False, color = 'darkblue', label = "Training data")

sns.regplot(x=y_test, y=y_test_pred, fit_reg=False, color = 'darkviolet', label = "Validation data")



# Create the titles

plt.title('OLS Regression: Predicted vs. Actual Sale Price', fontsize=24)

plt.xlabel('Log of Actual Sale Price', fontsize=18)

plt.ylabel('Log of Predicted Sale Price', fontsize=18)

plt.legend(loc = "upper left")



# Create the line of prediction

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'darkred')

plt.tight_layout()
# Fit the OLS onto train data

reg = LinearRegression()

reg.fit(train, y)



# Predict SalePrice of training and test dataset

y_train_OLS = reg.predict(train)

OLS_predictions = reg.predict(test)

print("RMSE on Full Training set :", rmsle(y, y_train_OLS).mean())
# Create a range of alphas that the model can use to penalise the model

# The model will run every one of the options and then take one that gives the lowest score

alphas = np.linspace(1,50,50)



# Fit the model on training using the alphas created

ridge = RidgeCV(alphas = alphas, cv=10)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)



# Fit a second model looking for greater precision in our alpha value

print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 

                cv = 10)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)



# Determine if any values were able to be set to 0

print("Ridge picked " + str(sum(ridge.coef_ != 0)) + " features and eliminated the other " +  \

      str(sum(ridge.coef_ == 0)) + " features")
# Locate the largest positive and negative coefficients

coef = pd.Series(ridge.coef_, index = X_train.columns).sort_values(ascending=False)

big_coef = coef.head(10).to_frame()

small_coef = coef.tail(10).to_frame()

coef = pd.concat([big_coef, small_coef])

coef.columns = ['Coefficient']



plt.figure(figsize=(16, 9))

plt.title('Ridge Regression Coefficient', fontsize=24)

plt.xlabel('Size of Coefficent', fontsize=18)

plt.ylabel('Feature', fontsize=18)



plt.tight_layout()

sns.barplot(x='Coefficient', y=coef.index, data=coef, palette='Purples_r', linewidth=1.0, edgecolor=".2")
# Predicting the SalePrice for our training and test sets

y_train_rdg = ridge.predict(X_train)

y_test_rdg = ridge.predict(X_test)



print("RMSE on Training set :", rmsle(y_train.values, y_train_rdg).mean())

print("RMSE on Validation set :", rmsle(y_test.values, y_test_rdg).mean())
plt.figure(figsize=(14, 7))

sns.regplot(x=y_train, y=y_train_rdg, fit_reg=False, color = 'darkblue', label = "Training data")

sns.regplot(x=y_test, y=y_test_rdg, fit_reg=False, color = 'darkviolet', label = "Test data")



plt.title('Ridge Regression: Predicted vs. Actual Sale Price', fontsize=24)

plt.xlabel('Log of Actual Sale Price', fontsize=18)

plt.ylabel('Log of Predicted Sale Price', fontsize=18)

plt.legend(loc = "upper left")



plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'darkred')

plt.tight_layout()
# Training the Ridge model on the full training dataset

alphas = np.linspace(1,50,50)

ridge = RidgeCV(alphas = alphas)

ridge.fit(train, y)

alpha = ridge.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 

                cv = 10)

ridge.fit(train, y)

alpha = ridge.alpha_

print("Best alpha :", alpha)



print("Ridge picked " + str(sum(ridge.coef_ != 0)) + " features and eliminated the other " +  \

      str(sum(ridge.coef_ == 0)) + " features")



y_ridge = ridge.predict(train)

print("Ridge RMSE on Full Training set :", rmsle(y, y_ridge).mean())

ridge_submission_price = ridge.predict(test)
#Create the list of possible values that alpha can take on (alpha in this case denotes the L1 penalty)

alphas = np.linspace(0.00001,1,1000)



#Fit the model to our training dataset using the 

lasso = LassoCV(alphas = alphas, max_iter = 50000, cv = 10)

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



print("LASSO picked " + str(sum(lasso.coef_ != 0)) + " features and eliminated the other " +  \

      str(sum(lasso.coef_ == 0)) + " features")
coef = pd.Series(lasso.coef_, index = X_train.columns).sort_values(ascending=False)

big_coef = coef.head(10).to_frame()

small_coef = coef.tail(10).to_frame()

coef = pd.concat([big_coef, small_coef])

coef.columns = ['Coefficient']



plt.figure(figsize=(16, 9))

plt.title('Lasso Regression Coefficient', fontsize=24)

plt.xlabel('Size of Coefficent', fontsize=18)

plt.ylabel('Feature', fontsize=18)



plt.tight_layout()

sns.barplot(x='Coefficient', y=coef.index, data=coef, palette='Purples_r', linewidth=1.0, edgecolor=".2")
y_train_las = lasso.predict(X_train)

y_test_las = lasso.predict(X_test)



print("Lasso RMSE on Training set :", rmsle(y_train, y_train_las).mean())

print("Lasso RMSE on Validation set :", rmsle(y_test, y_test_las).mean())
plt.figure(figsize=(16, 9))

sns.regplot(x=y_train, y=y_train_las, fit_reg=False, color = 'darkblue', label = "Training data")

sns.regplot(x=y_test, y=y_test_las, fit_reg=False, color = 'darkviolet', label = "Test data")

 

plt.title('Lasso Regression: Predicted vs. Actual Sale Price', fontsize=24)

plt.xlabel('Log of Actual Sale Price ', fontsize=18)

plt.ylabel('Log of Predicted Sale Price', fontsize=18)

plt.legend(loc = "upper left")



plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'darkred')

plt.tight_layout()
#Create the list of possible alphas

alphas = np.linspace(0.00001,1,1000)



#Fit the model to the full training dataset and alphas

lasso = LassoCV(alphas = alphas, max_iter = 50000, cv = 10)

lasso.fit(train, y)

alpha = lasso.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 

                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 

                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 

                          alpha * 1.4], 

                max_iter = 50000, cv = 10)

lasso.fit(train, y)

alpha = lasso.alpha_

print("Best alpha :", alpha)



print("LASSO picked " + str(sum(lasso.coef_ != 0)) + " features and eliminated the other " +  \

      str(sum(lasso.coef_ == 0)) + " features")



y_lasso = lasso.predict(train)

print("Lasso RMSE on Training set :", rmsle(y, y_lasso).mean())

lasso_submission_price = lasso.predict(test)
plt.figure(figsize=(16, 9))

sns.regplot(x=y_test, y=y_rf_test, fit_reg=False, color = 'darkcyan', label = 'Random Forest regression validation data')

sns.regplot(x=y_test, y=y_test_rdg, fit_reg=False, color = 'darkslategrey', label = 'Ridge regression validation data')

sns.regplot(x=y_test, y=y_test_las, fit_reg=False, color = 'darkblue', label = 'Lasso regression validation data')

sns.regplot(x=y_test, y=y_test_pred, fit_reg=False, color = 'darkviolet', label = 'OLS regression validation data')



    

plt.title('All models: Predicted vs. Actual Sale Price', fontsize=24)

plt.xlabel('Log of Actual Sale Price ', fontsize=18)

plt.ylabel('Log of Predicted Sale Price', fontsize=18)

plt.legend(loc = "upper left")



plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'darkred')

plt.tight_layout()
ridge_submission_price = np.expm1(ridge_submission_price)
plt.figure(figsize=(16, 9))

plt.title('Distribution of Predicted Sale Price for Test Set', fontsize=24)

sns.distplot(ridge_submission_price, color='darkviolet')

plt.xlabel('Predicted Sale Price (USD)', fontsize=18)

plt.ylabel('Frequency', fontsize=18)

plt.tight_layout()
solution = pd.DataFrame({"Id":test_ID, "SalePrice":ridge_submission_price})

solution.to_csv("ridge_sol.csv", index = False)