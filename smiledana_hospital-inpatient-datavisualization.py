import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../input/inpatient-hospital-charges/inpatientCharges.csv')
df.head(3)
df.describe()
# missing data check
df.isna().sum()
df.columns
# change to numerical variables
vars = [' Average Covered Charges ',' Average Total Payments ', 'Average Medicare Payments']
df[vars] = df[vars].replace({'\$': ''}, regex=True).astype(float)
df["Average Patients Payments"] = df[' Average Total Payments '] - df['Average Medicare Payments']
df.info()
# Biggest Payment Difference Treatment
minmax = df[['DRG Definition',' Average Total Payments ']].groupby(by='DRG Definition').agg(['max','min'])
minmax['Difference'] = minmax[(' Average Total Payments ', 'max')] - minmax[(' Average Total Payments ',   'min')]
difference5 = minmax.sort_values(by='Difference',ascending=False).head(5)
sns.set_context("paper")
ax = sns.barplot(difference5["Difference"],difference5.index, color="tomato",alpha=0.8)
ax.set(ylabel=None)
# Subset data with DRG Definition of 207(order by median)
df_207 = df[df["DRG Definition"] == "207 - RESPIRATORY SYSTEM DIAGNOSIS W VENTILATOR SUPPORT 96+ HOURS"]
sorted_index = df_207[['Provider State', ' Average Total Payments ']].groupby(by='Provider State').median().sort_values(by=' Average Total Payments ',ascending=False).index
plt.figure(figsize=(13,5))
sns.boxplot(x="Provider State", y=" Average Total Payments ",
            order = sorted_index, data=df_207)
plt.title('Avg. Total Payments for DRG = 207',fontsize=20)
sorted_index = df_207[['Provider State', 'Average Patients Payments']].groupby(by='Provider State').median().sort_values(by='Average Patients Payments',ascending=False).index
plt.figure(figsize=(13,5))
sns.boxplot(x="Provider State", y="Average Patients Payments",
            order = sorted_index, data=df_207)
plt.title('Avg. Patients Payment for DRG = 207',fontsize=20)
# Top 10 most expensive Average Covered Charges
df_table = df[['DRG Definition', ' Average Covered Charges ']].groupby(by='DRG Definition').agg(['mean','std','count'])
df_table = df_table.sort_values((' Average Covered Charges ',  'mean'), ascending=False).head(5)

fig,ax= plt.subplots()
fig = sns.barplot( df_table.iloc[:,0],df_table.index, color="steelblue",alpha=0.6)
ax.set(ylabel=None)
plt.xlabel("Average Covered Charges ($)")