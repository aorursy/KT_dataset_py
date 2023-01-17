import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

%matplotlib inline
# read data set
df_train = pd.read_csv("../input/train.csv")
df_train.head()
display(df_train.dtypes.value_counts())
#missing data
#df_train.info()
# present null value
null_percentage = ((df_train.isnull().sum().div(df_train['Id'].count()))).sort_values(ascending=False)
#pd.options.display.float_format = '{:.1%}'.format
null_percentage.head(10)

df_train_numeric = df_train.select_dtypes(include = ['int64', 'float64'])
display(df_train_numeric.head(3))
#display(df_train_numeric.columns.values)
df_train.describe().T
#general stats adescriptive statistics summary
#pd.options.display.float_format = '{0:,.2f}'.format
df_train['SalePrice'].describe()
# Distribution
sns.distplot(df_train['SalePrice'],color='green')
plt.show()
#skewness and kurtosis
#skewness is a measure of the asymmetry of the probability distribution about its mean.
print("Skewness: %f" % df_train['SalePrice'].skew())


log_SalePrice = np.log(df_train['SalePrice'])
sns.distplot(log_SalePrice,color='green')
plt.show()
print("Skewness: %f" % log_SalePrice.skew())
    
fig, axes = plt.subplots(nrows = 19, ncols = 2, figsize = (40, 200))
for ax, column in zip(axes.flatten(), df_train_numeric.columns):
    sns.distplot(df_train_numeric[column].dropna(), kde = False, ax = ax, color = 'green')
    ax.set_title(column, fontsize = 43)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 35)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 35)
    ax.set_xlabel('')
fig.tight_layout(rect = [0, 0.03, 1, 0.95])
#correlation 
corr= df_train.corr()
corr.style.background_gradient().set_precision(2)
#Top correlations
ex=  df_train_numeric
ex.head()

def get_redundant_pairs(df):
    #Get diagonal and lower triangular pairs of correlation matrix
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(ex, 20))

corr = df_train.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, 
            linewidths=0.01, annot_kws={'size':30})

## Main Variables ##
#Categorll variable: 
    #OverallQual- Rates the overall material and finish of the house    
#Continuous variable: 
    #GrLivArea- Above grade (ground) living area square feet 
    #TotalBsmtSF- Total square feet of basement area
    #GarageCars- Size of garage in car capacity
#Need to check: Neighborhood, 1stFlrSF
fig,axs = plt.subplots(ncols=4,figsize=(16,5))
sns.regplot(x='OverallQual', y='SalePrice', data=df_train, ax=axs[0])
sns.regplot(x='GrLivArea', y='SalePrice', data=df_train, ax=axs[1])
sns.regplot(x='GarageCars', y='SalePrice', data=df_train, ax=axs[2])
sns.regplot(x='TotalBsmtSF',y='SalePrice', data=df_train, ax=axs[3])
plt.show()
#box plot overallqual/saleprice- categorical features
#OverallQual- Overall material and finish quality
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#box plot GarageCars/saleprice- categorical features
#GarageCars-  Size of garage in car capacity
var = 'GarageCars'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#Convert MSSubClass, OverallQual, OverallCond, MoSold, YrSold into categorical variables
df_train.loc[:,['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']] = df_train.loc[:,['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']].astype('object')
display(df_train.dtypes.value_counts())
#final_data = pd.df_train.columns['OverallQual','GarageCars' 'GrLivArea', 'TotalBsmtSF', 'Neighborhood']

#GarageType-  Garage location
#MasVnrType-  Masonry veneer type
#MoSold-      Month Sold
fig, axs = plt.subplots(ncols=4,figsize=(16,5))
#plt.rcParams['figure.figsize'] = (10.0, 8.0)
sns.countplot(df_train["GarageType"],ax=axs[0])
sns.countplot(df_train["MasVnrType"],ax=axs[1])
sns.countplot(df_train["MoSold"],ax=axs[2])
sns.countplot(df_train["Neighborhood"],ax=axs[3])
plt.show()
df_train["Neighborhood"].unique()
#Convert MSSubClass, OverallQual, OverallCond, MoSold, YrSold into categorical variables
df_train.loc[:,['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']] = df_train.loc[:,['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']].astype('object')
display(df_train.dtypes.value_counts())
#Determine pivot table
impute_grps = data.pivot_table(values=["SalePrice"], index=[df_train["ExterCond"]]
                               , aggfunc=[np.mean, min, max])
print (impute_grps)
#Determine pivot table
impute_grps = data.pivot_table(values=["SalePrice"], index= [df_train["Condition1"]]
                               , aggfunc={np.mean, min, max,np.count_nonzero})
print (impute_grps)

print  (type(impute_grps))

impute_grps.head()
impute_grps = data.pivot_table(values=["SalePrice"], index= [df_train["Condition1"]]
                               , aggfunc=[np.mean, min, max,np.count_nonzero])
print (impute_grps)
impute_grps.head()