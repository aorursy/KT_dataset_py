#Import libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore") # ignore warnings
wine = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")

df = wine.copy()
#The data set has 150930 observations and 11 variables.

df.shape 
#First 10 records in the dataset.

df.head(10)
df.info()
#Unnecessary variable deletion.

df.drop("Unnamed: 0", axis = 1, inplace = True)
#Check, delete successful.

df.head(3)
#missing observation visualization tool..

import missingno as msno 
df.head()
#total missing values in the variables.

df.isna().sum() 
msno.matrix(df);
msno.heatmap(df);
# Missing Value Table

def missing_value_table(df):

    missing_value = df.isna().sum().sort_values(ascending=False)

    missing_value_percent = 100 * df.isna().sum()//len(df)

    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)

    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})

    cm = sns.light_palette("lightgreen", as_cmap=True)

    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)

    return missing_value_table_return

  

missing_value_table(df)
#5 countries are missing and 5 provinces are missing. 

df[df["province"].isna()]
#As a result of this review, we understand that the last 3 observations were produced in Chile.

df[df["winery"]=="Chilcas"].head(7)
#This missing value is in Greece.

df[df["winery"]=="Tsililis"].head()
#There is a result.In research on the Internet, we see that Turkey.

df[df["winery"]=="Büyülübağ"].head()
df['country'][df.winery=='Chilcas'] = "Chile"
df['country'][df.winery=='Tsililis'] = "Greece"
df['country'][df.winery=='Büyülübağ'] = "Turkey"
df[df["province"].isna()]
# Missing Value Table

def missing_value_table(df):

    missing_value = df.isna().sum().sort_values(ascending=False)

    missing_value_percent = 100 * df.isna().sum()//len(df)

    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)

    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})

    cm = sns.light_palette("lightgreen", as_cmap=True)

    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)

    return missing_value_table_return

  

missing_value_table(df)
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="most_frequent",missing_values=np.nan)
df_cat = df.drop(["country","description","points","variety","winery","price"],axis=1)
df_notnull = imp.fit_transform(df_cat)
var_name = list(df_cat)
df_notnull = pd.DataFrame(df_notnull, columns=var_name)
df_notnull.isna().sum()
#Filling the Price variable with the mean.

df['price'] = df['price'].fillna(df.groupby(['country','province'])['price'].transform('mean'))
#the minimum value is 80, the maximum value is 100. This is actually a variable that is between certain values, so it would be unreasonable to study outliers..

df.points.describe() 
#As we can see with the help of Boxplot, there are actually no outliers. Values range from 80 to 100.

sns.boxplot(df.points);
sns.boxplot(x = df.price);
df.price.describe()
Q1 = df.price.quantile(0.25)

Q3 = df.price.quantile(0.75)

IQR = Q3 - Q1
upper_value = Q3 + 1.5*IQR

lower_value = Q1 - 1.5*IQR
outlier_values = (df.price < lower_value) | (df.price > upper_value)
#total outliers..

df.price[outlier_values].value_counts().sum() 
df.price[outlier_values]
upper_outlier = df.price> upper_value
upper_outlier.sum()
df.price[upper_outlier] = upper_value
df.price[upper_outlier]
sns.boxplot(df.price);
from sklearn import preprocessing
df_points = df.select_dtypes(include=["int64"])
preprocessing.scale(df_points) #mean 0, std 1.
scaler = preprocessing.MinMaxScaler(feature_range=(10,30))
df_price = df[["price"]]
scaler.fit_transform(df_price)
df["country_01"] = np.where(df.country.str.contains("US"),1,0)

df.head()