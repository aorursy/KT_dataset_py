# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading in the data 
df = pd.read_csv("/kaggle/input/business-transaction-data/Business_Data.csv")
df_R = pd.read_csv("/kaggle/input/rand-and-jse-top40-data/Rand_data.csv", delimiter=';', low_memory=False)
df_J = pd.read_csv("/kaggle/input/rand-and-jse-top40-data/JSE_40.csv", delimiter=';', low_memory=False)
#Drop redundant columns
df = df.drop(columns=['Unnamed: 0'])

print(df.columns)
print()
print(df_R.columns)
print()
print(df_J.columns)

#Convert columns to special datatypes
df.Tran_Date = pd.to_datetime(df.Tran_Date)
#df.Military_Time = pd.to_datetime(df.Military_Time)

#Add week day column
df['Week_day'] = df.Tran_Date.dt.day_name()

cols = list(df.columns.values)
#Reorder columns
df = df[cols[0:2] + [cols[-1]] + cols[2:16]]

print(df.columns)
#Sort according to Merchant_Name and Trans_Date

df = df.sort_values(['Merchant_Name', 'Tran_Date'])

df
df.info()
df.describe()
df.columns
plt.style.use('dark_background')

plt.plot_date(df.Tran_Date, df.Amount, linestyle='solid')
#plt.plot_date(df.Tran_Date, df.Card_Amount_Paid, linestyle='solid')

#plt.tight_layout()
plt.show()

#print df.isna().any().any()
#print df.isna().sum().sum()
#print df['Amount'].isnull()

#df[df.isnull().any(axis=1)]

bool1 = pd.isnull(df['Amount'])
bool2 = pd.isnull(df['Card_Amount_Paid'])

print (bool1)
print (bool2)
#print df.Card_Amount_Paid.isna().any().any()
#print df.Card_Amount_Paid.isna().sum().sum()
#print df['Card_Amount_Paid'].isnull()
#print df['NUM_BEDROOMS'].isnull()

df.groupby(['Week_day'])
df.loc[df.Avg_Income_3M == "a.No inflows", "Avg_Income_3M"] = "A"
df.loc[df.Avg_Income_3M == "b", "Avg_Income_3M"] = "B"
df.loc[df.Avg_Income_3M == "c.R3,000 - R7,499", "Avg_Income_3M"] = "C"
df.loc[df.Avg_Income_3M == "d.R7,500 - R14,999", "Avg_Income_3M"] = "D"
df.loc[df.Avg_Income_3M == "e.R15,000 - R29,999", "Avg_Income_3M"] = "E"
df.loc[df.Avg_Income_3M == "e", "Avg_Income_3M"] = "E"
df.loc[df.Avg_Income_3M == "f.R30,000+", "Avg_Income_3M"] = "F"

df
df_R.columns
plt.style.use('dark_background')

plt.plot_date(df_R.Date, df_R.Price, linestyle='solid')
#plt.plot_date(df.Tran_Date, df.Card_Amount_Paid, linestyle='solid')

#plt.tight_layout()
plt.show()
#plt.style.use('dark_background')

#plt.plot_date(df_J.Date, df_J.Price, linestyle='solid')
#plt.plot_date(df.Tran_Date, df.Card_Amount_Paid, linestyle='solid')

#plt.tight_layout()
#plt.show()