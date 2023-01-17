import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data visulisation

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant



from sklearn.impute import SimpleImputer

import datetime

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from scipy.special import boxcox1p



import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Check the shape of the file

print('Shape of train dataset is : {}'.format(train.shape))

print('Shape of test dataset is : {}\n'.format(test.shape))

### Drop Id from the data

#-------------------------------------------------------

y_train = train['SalePrice']

train_ = train.drop('SalePrice', axis=1)



data = pd.concat([train_, test], axis=0, ignore_index=True)

data_Id = data['Id']

data = data.drop('Id', axis=1)

data.shape
### Segregate numeric and categorical data

#-------------------------------------------------------



# Select numerical columns 

num_cols = data.select_dtypes(include=[np.number])

# Select categorical columns 

cat_cols = data.select_dtypes(include=[object])
num_cols.head()
# Sort the numeric variables

num_vars = ['LotFrontage','LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 

            'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'YearRemodAdd',

           'YrSold', 'YearBuilt', 'GarageYrBlt', 'BsmtUnfSF']



# filter numeric data 

num_data = num_cols[num_vars]



# Filter out non-numeric from the numeric dataset

cat_vars = list(set(list(num_cols.columns))- set(num_vars))



# Concat non-numeric with categorical data

cat_data = pd.concat([cat_cols, num_cols[cat_vars]], axis=1)
# Map ordinal labels to the data

ordinal_code_mapping = {'Ex': 9, 'Gd': 7, 'TA': 5, 'Fa': 3, 'Po':1,'GLQ':7, 'Rec':5, 'ALQ':5, 'BLQ':4, 'LwQ':2,'Unf':1, 'Gd':7, 'Av':5, 'Mn':3, 'No':1}

cat_data = cat_data.replace(ordinal_code_mapping)



# Seperate ordinal data from nominal data

cat_ordinal_data = cat_data.select_dtypes(include=[np.number])

cat_nominal_data = cat_data.select_dtypes(include=object)
#Convert the year columns to datetime format

num_data['YearBuilt'] = pd.to_datetime(num_data['YearBuilt'], format='%Y', errors='ignore').dt.year

num_data['YearRemodAdd'] = pd.to_datetime(num_data['YearRemodAdd'], format='%Y', errors='ignore').dt.year

num_data['YrSold'] = pd.to_datetime(num_data['YrSold'], format='%Y', errors='ignore').dt.year

num_data['GarageYrBlt'] = pd.to_datetime(num_data['YrSold'], format='%Y', errors='ignore').dt.year





# Construct new variables for years since - Take the difference between the current year and variable year

num_data["Yrs_Since_YearBuilt"] = datetime.datetime.now().year - num_data['YearBuilt']  # substract to get the year delta

num_data["Yrs_Since_YearRemodAdd"] = datetime.datetime.now().year- num_data['YearRemodAdd']  # substract to get the year delta

num_data["Yrs_Since_YrSold"] = datetime.datetime.now().year - num_data['YrSold']  # substract to get the year delta

num_data["Yrs_Since_GarageYrBlt"] = datetime.datetime.now().year - num_data['GarageYrBlt']  # substract to get the year delta



# Delete date columns

num_data = num_data.drop(['YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt'], axis=1)

num_data_cols = num_data.columns
imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer_num = imputer_num.fit(num_data)

num_data = pd.DataFrame(imputer_num.transform(num_data), columns = num_data_cols)
# Check for missing values

nan_num_cols = num_data.columns[num_data.isnull().any()].tolist()

print('Missing Values(%) in numeric variables after imputing NaNs with mean:\n------------\n')

print(num_data[nan_num_cols].isnull().sum()/len(num_data)*100)
num_data.shape
cat_ordinal_data = cat_ordinal_data.fillna(0)



# Check for missing values

nan_cat_ord_cols = cat_ordinal_data.columns[cat_ordinal_data.isnull().any()].tolist()

print('Missing Values(%) after replacing NaNs in the ordinal variables with 0 :\n------------\n')

print(cat_ordinal_data[nan_cat_ord_cols].isnull().sum()/len(cat_cols)*100)
cat_ordinal_data = cat_ordinal_data.drop('MSSubClass', axis=1)

cat_ordinal_data.shape
#Add MSSubClass into nominal data

cat_nominal_data['MSSubClass'] = data['MSSubClass']

cat_nominal_data_cols = cat_nominal_data.columns

# lets replace the NaNs in the nominal variables for the NaNs explanation given in the data description

cat_nominal_data["Alley"].fillna("No alley access", inplace = True) 

cat_nominal_data["GarageType"].fillna("No Garage", inplace = True) 

cat_nominal_data["GarageFinish"].fillna("No Garage", inplace = True) 

cat_nominal_data["Fence"].fillna("No Fence", inplace = True) 

cat_nominal_data["MiscFeature"].fillna("No Misc", inplace = True) 
# Check for missing values

nan_cat_nom_cols = cat_nominal_data.columns[cat_nominal_data.isnull().any()].tolist()

print('Missing Values(%) after replacing nominal variables with data description:\n------------\n')

print(cat_nominal_data[nan_cat_nom_cols].isnull().sum()/len(cat_cols)*100)

# Missing variables

cat_nominal_missingvar = list(cat_nominal_data[nan_cat_nom_cols].isnull().sum().index)



# Lets impute Electrical and MasVnrType for mode

imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer_cat = imputer_cat.fit(cat_nominal_data[cat_nominal_missingvar])

cat_nominal_data[cat_nominal_missingvar] = imputer_cat.transform(cat_nominal_data[cat_nominal_missingvar])
# Check for missing values

nan_cat_nom_cols = cat_nominal_data.columns[cat_nominal_data.isnull().any()].tolist()

print('Missing Values(%) after replacing nominal variables with mode:\n------------\n')

print(cat_nominal_data[nan_cat_nom_cols].isnull().sum()/len(cat_cols)*100)
#One-Hot encoding

cat_nominal_encoded = pd.DataFrame()

for variables in cat_nominal_data:

    dummy_var = pd.get_dummies(cat_nominal_data[variables],prefix=variables)

    cat_nominal_encoded = pd.concat([cat_nominal_encoded,dummy_var], axis=1)



cat_nominal_encoded.head()
# Lets combine both numerical and ordinal variables for our numerical analysis

num_ord_data = pd.concat([num_data, cat_ordinal_data], axis=1)

num_ord_data.shape
# Plot correlation

num_corr = num_ord_data.corr()

#num_corr.style.background_gradient(cmap='coolwarm')
# Compute the correlation and filter for high correlation

corr_mat = num_corr.unstack(level=0).sort_values(kind="quicksort")

corr_mat_high = corr_mat[(corr_mat>=0.80) & (corr_mat<1)].drop_duplicates()

corr_mat_high
a = add_constant(num_ord_data)

vif_mat = pd.Series([variance_inflation_factor(a.values, i) for i in range(a.shape[1])],index=a.columns)

vif_mat[vif_mat>3]
# Drop highly correlated numeric variables

#num_data_ = num_ord_data.drop(['TotalBsmtSF', 'GrLivArea', 'Fireplaces', 'GarageArea', 'PoolArea', 'GarageCond','Yrs_Since_GarageYrBlt'], axis=1)

num_data_ = num_ord_data



a = add_constant(num_data_)

vif_mat = pd.Series([variance_inflation_factor(a.values, i) for i in range(a.shape[1])],index=a.columns)

vif_vars = vif_mat[vif_mat>3]

vif_vars
num_data_.head()


# Compute the skewness of the numeric variables

skew_mat = num_data_.skew(axis = 0, skipna = True).sort_values(kind="quicksort")

skewed_vars = skew_mat[(skew_mat>1)]





# Box-Cox transformation

skewed_vars_ = list(skewed_vars.index)

skew_mat = boxcox1p(num_data_[skewed_vars_],  0.15).skew(axis = 0, skipna = True).sort_values(kind="quicksort")

skewed_vars_boxcox = skew_mat[(skew_mat>=1.5)]

skewed_vars_boxcox
#Lets drop PoolArea

#num_data_ = num_data_.drop('PoolQC', axis=1)
num_data_.head()
plt.scatter(train['LotArea'] , train['SalePrice'])

plt.xlabel('Lot Area')

plt.ylabel('Sale Price')
dfs = [data_Id,num_data_,cat_nominal_encoded]

data_ = pd.concat(dfs, axis=1, sort=False)
colnames = list(data_.columns.str.replace(r'[^a-zA-Z\d]', "_"))

data_.columns = colnames

data_.head()
train_clean = data_.iloc[:1460,:]

test_clean = data_.iloc[1460:,:]



print(train_clean.shape)

print(test_clean.shape)
train_clean_ = train_clean[train_clean['LotArea']<100000]

train_clean_.shape
train_clean_['SalePrice'] = y_train

train_clean_.shape
from pandas import DataFrame

df_train = DataFrame(train_clean_, columns= train_clean_.columns)

df_train.to_csv('train_.csv', index = None, header=True)





df_test = DataFrame(test_clean, columns= test_clean.columns)

df_test.to_csv('test_.csv', index = None, header=True)
