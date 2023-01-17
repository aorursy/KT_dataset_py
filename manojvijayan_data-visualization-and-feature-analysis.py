# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as st
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
plt.style.use('fivethirtyeight')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/dengue-dataset.csv')
df.info()
df.head(5)
cols = ['Date', 'Confirmed_cases', 'Rain', 'Mean_temp', 'Min_temp', 'Max_temp']
df.columns = cols
# Extract the year and month from the date and separate them as different columns. 
df = pd.concat([pd.DataFrame([each[:2] for each in df['Date'].str.split('-').values.tolist()],
                             columns=['Year', 'Month']),df],axis=1)

# Convert the Date column to datetime
df.Date = df.Date.apply(lambda x : pd.to_datetime(x))

# Set the Date column as index
df.set_index('Date', inplace=True)

# Set the frequency of time series as Monthly
df = df.asfreq('MS')
df.info()
df.loc[df.Rain.isnull(),'Rain']
missing_df = df[(df.Month == '06') | (df.Month == '07') | (df.Month == '08')]
missing_df.dropna(inplace=True)
sns.boxplot(x = missing_df.Month, 
            y = missing_df.Rain)
missing_df.groupby('Month')['Rain'].describe()
fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(missing_df[['Mean_temp', 'Min_temp', 'Max_temp', 'Rain']].corr(method='spearman'),
                       ax=ax,cmap='coolwarm',
                       annot=True)
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df['Rain'].dropna(), lags=24)
df['Rain'].fillna(method='ffill',inplace=True)
df_yearly = df.groupby('Year')['Confirmed_cases'].agg('sum')
ax = df_yearly.plot(kind='bar',
                    title = 'Cases by Year',figsize=(15,8) )
ax.set_xlabel("Year")
ax.set_ylabel("Cases")
ax.axhline(df.Confirmed_cases.mean(), color= 'r', ls = 'dotted', lw =1.5)
plo = (df_yearly.pct_change())*100
for (i, v) , val in zip(enumerate(plo), df_yearly):
    ax.text(i-.2, val + 500, val, color='black', fontsize=11)
    if i != 0:
        if v > 0:
            colr = 'green'
        else:
            colr = 'red'
        ax.text(i-.3, val + 2000, str(int(v)) + '%', color=colr, fontsize=11)
df_year_month = df.groupby(['Year','Month'], as_index=False).agg('sum')
cross= pd.crosstab(df_year_month['Year'], df_year_month['Month'], df_year_month['Confirmed_cases'],aggfunc='sum')
fig, ax = plt.subplots(figsize=(10,5))
plt.title("Cases - Yearly %")
sns.heatmap((cross.div(cross.sum(axis=1), axis=0)*100),annot=True)
cross= pd.crosstab(df_year_month['Year'], df_year_month['Month'], df_year_month['Mean_temp'],aggfunc='sum')
fig, ax = plt.subplots(figsize=(10,5))
plt.title("Mean Temperature - Yearly")
sns.heatmap(cross,annot=True)
cross= pd.crosstab(df_year_month['Year'], df_year_month['Month'], df_year_month['Rain'],aggfunc='sum')
fig, ax = plt.subplots(figsize=(10,5))
plt.title("Rain - Yearly %")
sns.heatmap((cross.div(cross.sum(axis=1), axis=0)*100),annot=True)
for i in range(0,5):
    print('lag ' + str(i) + ' = ' + str(df['Confirmed_cases'].corr(df['Rain'].shift(i))))
for i in range(0,5):
    print('lag ' + str(i) + ' = ' + str(df['Confirmed_cases'].corr(df['Mean_temp'].shift(i))))
df_lag = df.copy()
df_lag['Rain'] = df_lag.Rain.shift(2)
df_lag['Mean_temp'] = df_lag.Mean_temp.shift(3)
df_lag.dropna(axis=0,inplace=True)
df_lag.info()
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
least_square1 = smf.ols('Confirmed_cases ~ Rain -1', data=df_lag).fit()
least_square2 = smf.ols('Confirmed_cases ~ Mean_temp -1', data=df_lag).fit()
least_square3 = smf.ols('Confirmed_cases ~ Rain + Mean_temp -1', data=df_lag).fit()
least_square4 = smf.ols('Confirmed_cases ~ Rain*Mean_temp -1', data=df_lag).fit()
anova_lm(least_square1, least_square2, least_square3, least_square4)
least_square4.summary()
least_square5 = smf.ols('Confirmed_cases ~ Month -1', data=df_lag).fit()
least_square5.summary()
anova_lm(least_square4, least_square5)
least_square6 = smf.ols('Confirmed_cases ~ Rain*Mean_temp + Month-1', data=df_lag).fit()
least_square6.summary()
anova_lm(least_square5, least_square6)
