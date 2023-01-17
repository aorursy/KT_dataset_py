# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import the required library 



import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm
# Reading the file

home_data=pd.read_csv("../input/home-data-for-ml-course/train.csv",index_col="Id")



# Observations

home_data.head()
home_data.shape



# There are total 80 variables in the data
# Dataset consist of numerical as well as categorical variable

# Numerical attributes

home_data.select_dtypes(exclude=["object"]).columns
# Total numerical attributes in the dataset

len(home_data.select_dtypes(exclude=["object"]).columns)
### Categorical attributes

cat_var=home_data.select_dtypes(include=["object"])

cat_var.columns

# Total categorical variable in the dataset

len(home_data.select_dtypes(include=["object"]).columns)
## General features of the Numerical attributes

home_data.select_dtypes(exclude=['object']).describe().round(2)
## Missing value in numerical attributes

num_var=home_data.select_dtypes(exclude=["object"])

num_var.isna().sum().sort_values(ascending=False).head(5)
# missing value in categorical attribute

cat_var.isna().sum().sort_values(ascending=False).head(18)
# Handling of missing values in categorical attribute

for i in cat_var.columns:

    home_data[i]=home_data[i].fillna("None")



    

    

# checking for missing data in categorical attribute after handling missing data    

cat_var=home_data.select_dtypes(include=["object"])

cat_var.isna().sum().max() 
price=home_data.SalePrice

price.describe()
home_data.MSZoning
### Relationship between sales price and different numerical variable





num_var=home_data.select_dtypes(exclude=["object"])

num_var.drop("SalePrice",axis=1,inplace=True)



fig=plt.figure(figsize=(14,20))



for i in range(len(num_var.columns)):

    fig.add_subplot(9,4,i+1)

    sns.scatterplot(x=num_var.iloc[:,i],y=price)



plt.tight_layout()

plt.show()
## scatterplot between sales price and "TotalBsmtSF"



sns.scatterplot(x="TotalBsmtSF",y="SalePrice",data=home_data)
## Scatterplot between sales price and "GRLivArea"



sns.scatterplot(x="GrLivArea",y="SalePrice",data=home_data)
## Relationship between sales price and "OverallQual"

plt.figure(figsize=(12,8))

sns.boxplot(x="OverallQual",y="SalePrice",data=home_data)
### how sale prices are varying in different year.

def ggplots(a):

    plt.style.use('ggplot')

    sns.boxplot(y="SalePrice",x="YearBuilt",data=home_data,orient="v")

    plt.xticks(rotation=90)

    plt.tight_layout()

    

    

    

    

plt.figure(figsize=(20,10))

ggplots(home_data)
## Correlation between attributes

num_var=home_data.select_dtypes(exclude=["object"])

num_corr=num_var.corr()

num_corr.head()
plt.figure(figsize=(16,10))

sns.set(style="darkgrid")

sns.heatmap(num_corr)

plt.title("Correlation Between Attributes",size=20);
## Top 10 attributes which are strongly correlated with sales price 



num_corr["SalePrice"].sort_values(ascending=False).head(10)
### sales price correlational matrix



top10_attribute=home_data[["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt","SalePrice"]]

top10_corr=top10_attribute.corr()

fig=plt.figure(figsize=(14,8))

sns.set(style="darkgrid")



fig=sns.heatmap(top10_corr,annot=True,cbar=True)
cat_var=home_data.select_dtypes(include=["object"])

cat_var.head()
cat_var.columns
### category in kitchen quality

home_data.KitchenQual.value_counts()
fig, ax=plt.subplots(figsize=(10,8))

sns.boxplot(x="KitchenQual",y="SalePrice",data=home_data)

plt.title("Sales Price Based on Kitchen Quality",size=(18),color="Red")

plt.show()
### Type of House style mostly liked by buyers



home_data.HouseStyle.value_counts()
fig,ax=plt.subplots(2,1,figsize=(14,14))



sns.barplot(y="HouseStyle",x="SalePrice",data=home_data,orient="h",ax=ax[0])

ax[0].set_title("House style Preffered By cutomers",color="Blue")

sns.boxplot(y="Neighborhood",x="SalePrice",data=home_data,ax=ax[1])

ax[1].set_title("Neighborhood VS SalePrice",color="Blue");

### Count of categories within Foundation

plt.figure(figsize=(12,6))

sns.countplot(x="Foundation",data=home_data)

plt.title("Foundation counts",color="Blue");
## Explore data with sales price





plt.figure(figsize=(10,6))

sns.distplot(home_data["SalePrice"],fit=norm)

plt.title("Distribution of the sales price",color="Red",size=15);
## Probability Plot of Sales Price

plt.figure(figsize=(10,6))

stats.probplot(home_data["SalePrice"],plot=plt);
## skewness  and kurtosis of sales price 

print("Skewness of the sales price : {:.2f}".format(price.skew()))

print("Kurtosis of the sales price : {:.2f}".format(price.kurt()))
### tansformed data and histogram and probability plot



saleprice_log=np.log(home_data["SalePrice"])



fig,ax=plt.subplots(2,1,figsize=(12,14))

sns.distplot(saleprice_log,fit=norm,ax=ax[0])

ax[0].set_title("Distribution of Log_Transformed SalePrice ",size=15,color="Green")

ax[1]=stats.probplot(saleprice_log,plot=plt)

# Distribution of GrLivArea



fig,ax=plt.subplots(1,2,figsize=(14,6))

sns.distplot(home_data["GrLivArea"],fit=norm,ax=ax[0])

ax[0].set_title("Distribution of GrLivArea ",size=15,color="Green")

ax[1]=stats.probplot(home_data["GrLivArea"],plot=plt)

home_data["GrLivArea"].skew()
## Data Transformation

Livarea=np.log(home_data["GrLivArea"])

fig,ax=plt.subplots(1,2,figsize=(14,6))

sns.distplot(Livarea,fit=norm,ax=ax[0])

ax[0].set_title("Distribution of Log_transformed GrLivArea ",size=15,color="Red")

ax[1]=stats.probplot(Livarea,plot=plt)
