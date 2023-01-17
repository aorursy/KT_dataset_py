#import required libraries

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#reading data files

store_df=pd.read_csv("../input/rossmann-store-sales/store.csv")

train_df=pd.read_csv("../input/rossmann-store-sales/train.csv")
store_df.head()
store_df.describe()
#Checking the no. of NaN vales

store_df.isna().sum()
train_df.head()
train_df.describe()
#Checking the no. of NaN values

train_df.isna().sum()
#Merging both the Dataframes into one based on the "Store" ID

df=store_df.merge(train_df,on=["Store"],how="inner")

df.head()
#(rowsxcolumns) of the merged DataFrame

df.shape
#Checking the no. of NaN values

df.isna().sum()
#Dropping columns

df=df.drop(columns=["PromoInterval","Promo2SinceWeek","Promo2SinceYear","CompetitionOpenSinceMonth","CompetitionOpenSinceYear"])
#Handling NaN

df["CompetitionDistance"]=df["CompetitionDistance"].fillna(df["CompetitionDistance"].mode()[0])
#Find the range of data

plt.figure(figsize=(5,10))

sns.set(style="whitegrid")

sns.distplot(df["Sales"])
#Find the range of the data

plt.figure(figsize=(5,10))

sns.set(style="whitegrid")

sns.distplot(df["Customers"])
plt.figure(figsize=(10,10))

sns.set(style="whitegrid")

sns.boxenplot(data=df,scale="linear",x="DayOfWeek",y="Sales",color="orange")
plt.figure(figsize=(10,10))

sns.set(style="whitegrid")

sns.boxenplot(y="Customers", x="DayOfWeek",data=df, scale="linear",color="orange")
df["Sales"]=df["Sales"].apply(lambda x: 20000 if x>20000 else x)

df["Customers"]=df["Customers"].apply(lambda y: 3000 if y>3000 else y)

print(max(df["Sales"]))

print(max(df["Customers"]))
df["Date"]=pd.to_datetime(df["Date"])

df["Year"]=df["Date"].dt.year

df["Month"]=df["Date"].dt.month

df["Day"]=df["Date"].dt.day

df["Week"]=df["Date"].dt.week%4

df["Season"] = np.where(df["Month"].isin([3,4]),"Spring",np.where(df["Month"].isin([5,6,7,8]), "Summer",np.where(df["Month"].isin ([9,10,11]),"Fall",np.where(df["Month"].isin ([12,1,2]),"Winter","None"))))

df
Holiday_Year_Month_Week_df=pd.DataFrame({"Holiday per week":df["SchoolHoliday"],"Week":df["Week"],"Month":df["Month"],"Year":df["Year"],"Date":df["Date"]})

Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.drop_duplicates(subset=['Date'])

Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.groupby(["Year","Month","Week"]).sum()

Holiday_Year_Month_Week_df
df=df.merge(Holiday_Year_Month_Week_df, on=["Year","Month","Week"],how="inner")
customer_time_df=pd.DataFrame({"Avg CustomersPerMonth":df["Customers"],"Month":df["Month"]})

AvgCustomerperMonth=customer_time_df.groupby("Month").mean()

AvgCustomerperMonth
customer_time_df=pd.DataFrame({"Avg CustomersPerWeek":df["Customers"],"Week":df["Week"],"Year":df["Year"],"Month":df["Month"]})

AvgCustomerperWeek=customer_time_df.groupby(["Year","Month","Week"]).mean()

AvgCustomerperWeek
df=df.merge(AvgCustomerperMonth,on="Month",how="inner")

df=df.merge(AvgCustomerperWeek,on=["Year","Month","Week"],how="inner")
promo_time_df=pd.DataFrame({"PromoCountperWeek":df["Promo"],"Year":df["Year"],"Month":df["Month"],"Week":df["Week"],"Date":df["Date"]})

promo_time_df=promo_time_df.drop_duplicates(subset=['Date'])

promo_time_df=promo_time_df.groupby(["Year","Month","Week"]).sum()

promo_time_df
df=df.merge(promo_time_df,on=["Year","Month","Week"], how="inner")
numerical_data_col=["Store","Competition Distance","Promo2","DayOfWeek","Sales","Customers","Open","SchoolHoliday","Year","Month","Day","Week"]

categorical_data_col=["StoreType","Assortment","Season"]
for i in categorical_data_col:

    p=0

    for j in df[i].unique():

        df[i]=np.where(df[i]==j,p,df[i])

        p=p+1



    df[i]=df[i].astype(int)
#The column StateHoliday contains 0,'0',a and b. This needs to be conerted to a pure numerical data column

df["StateHoliday"].unique()
df["StateHoliday"]=np.where(df["StateHoliday"] == '0' ,0,1)

df["StateHoliday"]=df["StateHoliday"].astype(int)
plt.figure(figsize=(10,10))

sns.set(style="whitegrid",palette="pastel",color_codes=True)

sns.violinplot(x="DayOfWeek",y="Sales",hue="Promo",split=True, data=df)
plt.figure(figsize=(10,10))

sns.set(style="whitegrid",palette="pastel",color_codes=True)

sns.violinplot(x="DayOfWeek",y="Customers",hue="Promo",split=True, data=df)
sns.set(style="whitegrid")

g=sns.relplot(x="CompetitionDistance", y="Sales", hue="Promo", data=df)

g.fig.set_size_inches(15,15)
sns.set(style="whitegrid")

g=sns.relplot(y="Avg CustomersPerWeek", x="Week", hue="Holiday per week", data=df)

g.fig.set_size_inches(10,10)
sns.set(style="whitegrid")

g=sns.relplot(y="Holiday per week", x="Week", hue="PromoCountperWeek", data=df)

g.fig.set_size_inches(10,10)
df.head()
#Find Correlation between the data columns

plt.figure(figsize=(15,15))

sns.heatmap((df.drop(columns=["Date"]).corr()))
df=df.drop(columns=["Date"])

df.shape
#Splitting of data

features=df[["Customers","Open","Promo","Assortment","PromoCountperWeek","SchoolHoliday","StoreType","Week","Month"]]

features=preprocessing.scale(features)

target=df["Sales"]

X_train,X_test,Y_train,Y_test=train_test_split(features,target)
model1=LinearRegression()

model1.fit(X_train,Y_train)

print(model1.score(X_test,Y_test))