##Import Libraries
import pandas as pd

import numpy as np

import sklearn as sc

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as mnso



from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', None)

np.set_printoptions(threshold=np.inf)
df_train = pd.read_csv("/kaggle/input/train.csv")
df_test = pd.read_csv("/kaggle/input/test.csv")
df_train.head()
#Check all unique column data types

df_train.dtypes.unique()
## 'O' in above list signifies the Object i.e categorical column and 'int64','float64' are numeric columns

## Lets get the list of the numeric and categorical columns



l_num_col = df_train.select_dtypes(exclude='O').columns

l_cat_col = df_train.select_dtypes(include='O').columns

print("Numeric Columns : ", l_num_col)

print("Non Numeric Columns : ", l_cat_col)
# Numeric columns stats 

df_train.select_dtypes(exclude='O').describe()



## Observations from Describe : 

# 1. Outliers present in some columns like LotArea, BsmtFinSF1 etc. - Outliers doesnot mean that its a junk value, it can be junk but it can be useful as well.

# So before dropping off or modify these outliers. It need more analysis

# 2. Featrue scaling is required - Some larger value columns  can override the effect of small value column.

# 3. There are missing values - From count we can easily observe that.

#####Understanding Sale Pattern

sns.distplot(df_train['SalePrice'])

##Skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
###Observation : 

# 1. Positive Skewness = Means trail on the right side

# 2. High Kurtosis = Peak on the top

#sns.histplot(df_train[l_num_col])

df_train[l_num_col].hist(figsize=(25,25))

plt.suptitle("Numeric feature distribution")

plt.show()


# Compare with dependent varaible i.e sale price



#Check correlation

correlation = df_train.corr()

correlation['SalePrice']
plt.figure(figsize=(25,12))

sns.heatmap(correlation, annot = True)

plt.plot
### Check the correlation of sales with independent variable

# 1. OveGrallQual, TotalBsmtSF, 1stFlrSF  are highly correlated with sale price.

# 2. GarageCars and Garage area seems to interlinked with each other that is the case of multicollinearity i.e. check 88% correlation between both





####Analyse collinearity of these variables with sale prize 
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']

sns.pairplot(df_train[columns],height = 2 ,kind ='scatter',diag_kind='kde')

plt.show()
## Observation

# We can see the correlation of these shortlisted independent varaibles with sale price.
## Let check for outliers 



fig, axes = plt.subplots(len(columns), figsize=(12,len(columns)*5))

count = 0



for i in columns:

    df_train.boxplot(column=i, ax=axes[count]) 

    count=count+1



plt.show()
## Observation

# There are some outliers which doesnot lie between Q1-1.5*IQR and Q3+1.5*IQR
#Missing number matrix

# mnso is the good tool to get the missing numbers

# In below graph, white line shows missing values

mnso.matrix(df_train[l_num_col])
# Check on shortlisted columns

mnso.matrix(df_train[columns])
#Missing number bar graph

mnso.bar(df_train[l_num_col])
#Missing number bar graph on shortlisted values

mnso.bar(df_train[columns])
#Skewness Graph for numeric variables

sns.distplot(df_train[l_num_col].skew(),color = 'blue', axlabel= 'Skewness')
# Graph for numeric variables

plt.figure(figsize = (12,8))

sns.distplot(df_train[l_num_col].kurt(),color = 'orange', axlabel= 'Kurtosis')
l_cat_col
##Check number of unique values in each feature

df_train[l_cat_col].nunique()
## Categorical Variable analysis



fig, axes = plt.subplots(len(l_cat_col), figsize=(8,len(l_cat_col)*5))

count = 0



for i in l_cat_col:

    sns.countplot(df_train[i], ax=axes[count])

    count=count+1



plt.show()
sns.barplot(x= "MSZoning",y="SalePrice", data = df_train)



## Categorical Variable analysis



fig, axes = plt.subplots(len(l_cat_col), figsize=(8,len(l_cat_col)*5))

count = 0



for i in l_cat_col:

    sns.barplot(x= i,y="SalePrice", data = df_train, ax = axes[count])

    count=count+1



plt.show()