import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Load dataset

df_train = pd.read_csv('../input/black-friday/blackFriday_train.csv')

df_train.head()
df_train.shape
df_test = pd.read_csv('../input/black-friday/blackFriday_test.csv')

df_test.head()
df_test.shape
# Concat two datasets

df = df_train.append(df_test, sort=False)

df.shape
df.head()
df.describe()
df.isnull().sum()/df.shape[0]*100
df.drop(["User_ID"], axis=1, inplace=True)

df.head()
# Convert categorical data into numerical

df["Gender"] = df["Gender"].map({'M': 1, 'F':0})

df.head()
df.Gender.unique()
# Mapping age variable

sorted(df.Age.unique())
age_map = {'0-17':1, '18-25':2, '26-35':3, '36-45':4, '46-50':5, '51-55':6, '55+':7}
df["Age"] = df["Age"].map(age_map)

df.head()
# Map city variable

df.City_Category.unique()
city = pd.get_dummies(df.City_Category, drop_first=True)

city.head()
# Concat dummy variables to df

df = pd.concat([df, city], axis=1)

df.head()
# Feature Analysis

df.groupby(["Product_ID"])["Product_ID"].count().sort_values(ascending=False)
df.isnull().sum()
np.where(df.Product_Category_2.isnull())
df.Product_Category_2
def impute_nan(df,variable):

    

    df[variable+"_random"]=df[variable]

    ##It will have the random sample to fill the na

    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0, replace=True)

    ##pandas need to have same index in order to merge the dataset

    random_sample.index=df[df[variable].isnull()].index

    df.loc[df[variable].isnull(),variable+'_random']=random_sample
## Impute nan values with random samples



impute_nan(df, 'Product_Category_2')
impute_nan(df, 'Product_Category_3')
df.head()
# Impute NaN values in purchase by the average purchase
avg_purchase = df["Purchase"].mean()

df["Purchase"].fillna(avg_purchase, inplace=True)
df.isnull().sum()
df.drop(["Product_Category_2", "Product_Category_3"], axis=1, inplace=True)
df.rename(columns={'Product_Category_1':'cat1', 'Product_Category_2_random': 'cat2', 'Product_Category_3_random': 'cat3'}, inplace=True)

df.columns
df = df[['Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',

       'Stay_In_Current_City_Years', 'Marital_Status',  'B',

       'C', 'cat1','cat2', 'cat3', 'Purchase']]

df.head()
# Confirm no nan values 

df.isnull().sum()
df.drop(["City_Category"], axis=1, inplace=True)
df.head(2)
#Replace symbols



df["Stay_In_Current_City_Years"].unique()
df["Stay_In_Current_City_Years"] = df["Stay_In_Current_City_Years"].str.replace('+', ' ')

df[df["Stay_In_Current_City_Years"]=='4 '].head(2)
df.info()
df["Stay_In_Current_City_Years"] = df["Stay_In_Current_City_Years"].astype(int)

df["B"] = df["B"].astype(int)

df["C"] = df["C"].astype(int)
df.dtypes
df_new = df.copy()

df_new.head(3)
df_new.groupby(["Age"])["Purchase"].mean().plot(kind = 'line');
sns.scatterplot(x="Age", y="Purchase", data=df_new);
sns.lineplot(x="Age", y="Purchase", data=df_new);
sns.barplot(x="Gender", y="Purchase", data=df_new);
sns.barplot(x ="cat1", y = "Purchase",  data=df_new);
sns.barplot(x ="cat2", y = "Purchase",  data=df_new);
sns.barplot(x ="cat3", y = "Purchase",  data=df_new);
df_new.head(2)
sns.barplot(x="Marital_Status", y="Purchase", data=df_new);
sns.relplot(x="Marital_Status", y="Purchase", data=df_new, hue="Gender")
df_new.Occupation.unique()
occ_purchase = df_new.groupby(["Occupation"])['Purchase'].mean().sort_values()

plt.figure(figsize=(10,8))

occ_purchase.plot(kind="barh", color="green");
sns.set()

plt.figure(figsize=(15,10))

ax = sns.heatmap(df_new.corr(),annot=True,linewidths=.5)
df_new.corr()
df_new.drop(["Product_ID"], axis=1, inplace=True)

df_new.head(2)