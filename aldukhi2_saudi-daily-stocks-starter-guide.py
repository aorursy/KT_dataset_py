import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

%config InlineBackend.figure_format = 'retina'

plt.style.use('ggplot')
# load on data set

df_stc = pd.read_csv('/kaggle/input/saudi-daily-stocks-history-test/STC_7010.csv')
# look at shape

df_stc.shape
df_stc.dtypes
# convert date to datetime

df_stc.date = pd.to_datetime(df_stc.date)
# plot closing prices

plt.figure(figsize=(8,6))

plt.plot(df_stc.date, df_stc.close)

plt.xlabel("Year")

plt.ylabel('Price (SAR)');

plt.title("STC Stock Price");

# plot moving-average

plt.figure(figsize=(8,6))

plt.plot(df_stc.date, df_stc.close.rolling(window=30).mean())

plt.xlabel("Year")

plt.ylabel('Price (SAR)');

plt.title("STC Stock Price with 30-day Moving-average");
df_stc.value_traded_SAR = df_stc.value_traded_SAR.str.replace(',','').astype(float)
# plot closing prices

plt.figure(figsize=(8,6))

plt.plot(df_stc.date, df_stc.value_traded_SAR)

plt.xlabel("Year")

plt.ylabel('Value Traded (SAR)');

plt.title("STC Stock Value Traded");



# plot moving-average

plt.figure(figsize=(8,6))

plt.plot(df_stc.date, df_stc.value_traded_SAR.rolling(window=30).mean())

plt.xlabel("Year")

plt.ylabel('Value Traded (SAR)');

plt.title("STC Stock Value Traded with 30-day Moving-average");