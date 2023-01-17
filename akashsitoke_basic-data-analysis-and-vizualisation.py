#Loading file data
import pandas as pd
import numpy as np
df_data=pd.read_csv("/kaggle/input/nifty-indices-dataset/NIFTY 50.csv")
df_data.head()

df_data.columns
#Data types of the Features
df_data.dtypes
#Identifying the missing data
missing_data = df_data.isnull()
missing_data.head(5)

#Identifying the number of missing data in all columns 
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 
#Averaging the volume
avg_vol = df_data["Volume"].astype("float").mean(axis=0)
print("Average Volume:", avg_vol)
#Averaging the Turnover
avg_tur = df_data["Turnover"].astype("float").mean(axis=0)
print("Average Turnover:", avg_tur)
#Replacing the missing or nan value by the average value in volume column
df_data["Volume"].replace(np.nan, avg_vol, inplace=True)
#Replacing the missing or nan value by the average value in turnover column
df_data["Turnover"].replace(np.nan, avg_tur, inplace=True)
#Rechecking the missing values in the columns
missing_data = df_data.isnull()
missing_data.head(5) 
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 
    
#So there is no missing values now
#converting the data type of datetime from object to timestamp
df_data["Date"]= pd.to_datetime(df_data["Date"])
df_data.dtypes
df_data['month'] = df_data['Date'].dt.month
df_data['year'] = df_data['Date'].dt.year
df_data.head()
df_data.info()
df_data.describe()
df_data.corr()
# Importing all the Vizualization libraries
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
import matplotlib.pyplot as plt

df_data[["Volume", "Turnover"]].corr()
sns.regplot(x="Volume", y="Turnover",color='green', data=df_data,truncate=False)
plt.ylim(0,)
df_data[["Volume", "Div Yield"]].corr()
sns.regplot(x="Volume", y="Div Yield", color='r',data=df_data,truncate=False)
plt.ylim(0,)
df_data[["Turnover", "Div Yield"]].corr()
sns.regplot(x="Turnover", y="Div Yield",data=df_data,truncate=False)
plt.ylim(0,)
sns.relplot(x="month", y="Volume", kind="line", data=df_data)
sns.relplot(x="month", y="Turnover", kind="line", data=df_data)
sns.set_style("darkgrid")
plt.figure(figsize=(15,6))
sns.barplot(x='year', y='Volume', data=df_data)
plt.figure(figsize=(15,6))
df= sns.barplot(x="year", y="Turnover", data=df_data,ci="sd")
plt.figure(figsize=(18,5))
sns.pointplot(x='year', y='Div Yield', data=df_data)
plt.figure(figsize=(15,6))
df = sns.boxplot(x="year", y="P/E", data=df_data)
plt.figure(figsize=(15,6))
df = sns.boxplot(x="year", y="P/B", data=df_data)
df_ohlc=df_data[['Date','Open', 'High','Low', 'Close']]
df_ohlc.head()
import seaborn as sns; sns.set(style="ticks", color_codes=True)
df = sns.pairplot(df_ohlc, corner=True)
df_ohlc.corr()
df = sns.jointplot(x="High", y="Low", data=df_data,kind='hex',height=5, ratio=3)
sns.set_style("darkgrid")
plt.figure(figsize=(18,5))
sns.pointplot(x='year', y='High', data=df_data)
