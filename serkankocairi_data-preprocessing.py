import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns # visualization

import matplotlib.pyplot as plt # visualization

import missingno as msno # visualizatin for missing values



import warnings

warnings.filterwarnings("ignore") # ignore warnings
wine = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")

df = wine.copy()
df.head()
df.info()
df.columns
df.drop("Unnamed: 0", axis = 1, inplace = True)
df.columns
round(df.describe(),3) 
plt.figure(figsize=(12,5))

sns.heatmap(df.corr(),annot=True,linewidth=2.5,fmt='.3F',linecolor='black');
sns.pairplot(df);
df.isnull().values.any()
df.isnull().sum()
msno.bar(df,color = sns.color_palette('viridis'));
msno.matrix(df, color = (0.2, 0.4, 0.4));
msno.heatmap(df);
# Missing Value Table

def missing_value_table(df):

    missing_value = df.isna().sum().sort_values(ascending=False)

    missing_value_percent = 100 * df.isna().sum()//len(df)

    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)

    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})

    cm = sns.light_palette("red", as_cmap=True)

    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)

    return missing_value_table_return

  

missing_value_table(df)
df.drop("region_2", axis = 1, inplace = True)
df['price'] = df['price'].fillna(df.groupby(['country','province'])['price'].transform('mean'))
df.isnull().sum()
df = df[~(df.price.isnull())]
df=df.sort_values(by=["country","province","region_1"],ascending=True)

df.head()
df=df.fillna(method="ffill")
plt.subplot(2,1,1)

df.points.plot(kind='hist',color='pink',bins=50,figsize=(10,10))

plt.title("Points Variable Histogram Chart");





plt.subplot(2,1,2)

df.price.plot(kind='hist',color='pink',bins=50,figsize=(10,10))

plt.title("Price Variable Histogram Chart");
sns.boxplot(df.points);
sns.boxplot(df.price);
Q1 = df.price.quantile(0.25)

Q3 = df.price.quantile(0.75)

IQR = Q3 - Q1
print("Q1:",Q1)

print("Q3:",Q3)

print("IQR:",IQR)
upper_value = Q3 + 1.5*IQR

lower_value = Q1 - 1.5*IQR
print("upper_value:",upper_value)

print("lower_value:",lower_value)
outlier_values = (df.price < lower_value) | (df.price > upper_value)
df.price[outlier_values].value_counts().sum() 
upper_outlier = df.price> upper_value

upper_outlier.sum()
df.price[upper_outlier] = upper_value
sns.boxplot(df.price);