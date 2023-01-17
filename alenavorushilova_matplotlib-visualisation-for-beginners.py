import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

%matplotlib inline

import random

import seaborn as sns

from fbprophet import Prophet

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
df_org = pd.read_csv('../input/world-unemployment-rate-from-oecd/DP_LIVE_10012020131736460 Unemployment Rate OECD.csv', index_col = 'TIME', parse_dates = True)

df_org.head()
df_rus = df_org.loc[df_org['LOCATION']==('RUS')]

df_rus = df_rus[['SUBJECT', 'Value']]

df_rus.head()
df_rus.describe()
df_rus.info()
#Looking for the unique values in the column 'SUBJECT'

df_rus.SUBJECT.unique()
#preparing data by pivoting the table

df = df_rus.pivot_table(values='Value',index=['TIME'],columns=['SUBJECT'])

df.head()
#removing NAs

df = df.dropna()

df.head()
#renaming columns with .rename

df.rename(columns={"MEN":"MEN", "TOT":"TOTAL", "WOMEN":"WOMEN"}, inplace = True)

df.head()
#rearranging columns

df = df[["MEN", "WOMEN", "TOTAL"]]

df.head()
#coloring table

cm = sns.light_palette("blue", as_cmap=True)



s = df.style.background_gradient(cmap=cm)

s
#plotting the data by markers and different colors for each time series, specifying the size of the graph

#labeling y-axis and the title, adding the grid

plt.figure(figsize=(12,6), dpi= 80)

plt.plot(df["TOTAL"],'g^',df["WOMEN"],'ro',df["MEN"],'bs')

plt.ylabel('Unemployment Rate')

plt.title("Russian Unemployment", fontsize=22)

plt.grid(axis='both', alpha=.3)

plt.show()
#plotting the time series with the lines instead of markers, adding the legend with specific labels

plt.figure(figsize=(10,6), dpi= 80)

plt.plot(df.index, df["TOTAL"],'g',df["WOMEN"],'r',df["MEN"],'b')

plt.legend(labels=('Average', 'Women', 'Men'))

plt.ylabel('Unemployment')

plt.title("Russian Unemployment", fontsize=22)

plt.grid(axis='both', alpha=.3)

# Remove borders

plt.gca().spines["top"].set_alpha(0.0)    

plt.gca().spines["bottom"].set_alpha(0.3)

plt.gca().spines["right"].set_alpha(0.0)    

plt.gca().spines["left"].set_alpha(0.3)   

plt.show()
#plotting only "WOMEN" values, changing color, marker and '-' line style

plt.figure(figsize=(10,6), dpi= 80)

plt.plot_date(df.index, "WOMEN", data=df, color='r', ls = '-', marker = '')

plt.legend(loc = 'upper center')

#to fit in the date labels

plt.tight_layout()

plt.show()
#another way of doing it

df['WOMEN'].plot(figsize=(10,6), color='purple')

plt.legend(loc = 'upper center')

#to fit in the date labels

#fig.autofmt_xdate()

plt.tight_layout()

plt.show()
#changing the interval or slising the date

fig, ax = plt.subplots(figsize=(10,6))

ax.plot(df.loc['2010-01':'2020-02', 'TOTAL'], color = 'g',marker='o', linestyle='-')

ax.set_title('Unemployment')

plt.grid(axis='both', alpha=.2)



#removing the borders

plt.gca().spines["top"].set_alpha(0.0)    

plt.gca().spines["bottom"].set_alpha(0.3)

plt.gca().spines["right"].set_alpha(0.0)    

plt.gca().spines["left"].set_alpha(0.3)   

plt.show()
#creating subplots

fig, axes = plt.subplots(2,1, figsize=(10,7), sharex=True)

df[["WOMEN", "MEN"]].plot(subplots=True, ax=axes)
#plotting the stackable graphs and changing its colors

plt.figure(figsize=(25,16), dpi= 100)

df.plot.area(color = sns.color_palette("Set2"))

plt.legend(loc='best')

plt.title("Russian Unemployment", fontsize = 22)

plt.show()
#another way to change the colors

plt.figure(figsize=(25,16), dpi= 100)

df.plot.area(color = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"])

plt.legend(loc='best')

plt.title("Russian Unemployment", fontsize = 22)

plt.show()
#percentage stacked area chart

data = df[["MEN", "WOMEN"]]

data_perc = data.divide(data.sum(axis=1), axis=0)

 

#create the plot

data_perc.plot.area(color = sns.color_palette("Set1"))

plt.legend(loc='best')

plt.margins(0,0)

plt.title("Russian Unemployment", fontsize = 22)

plt.show()
#autocorrelation and partial autocorrelation

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6), dpi= 80)

plot_acf(df["TOTAL"].values.tolist(), ax=ax1, lags=50, color = "purple", marker = "^")

plot_pacf(df["TOTAL"].values.tolist(), ax=ax2, lags=20, color = "yellow", marker = "o")



# Decorate

# lighten the borders

ax1.spines["top"].set_alpha(.3); ax2.spines["top"].set_alpha(.3)

ax1.spines["bottom"].set_alpha(.3); ax2.spines["bottom"].set_alpha(.3)

ax1.spines["right"].set_alpha(.3); ax2.spines["right"].set_alpha(.3)

ax1.spines["left"].set_alpha(.3); ax2.spines["left"].set_alpha(.3)



# font size of tick labels

ax1.tick_params(axis='both', labelsize=14)

ax2.tick_params(axis='both', labelsize=14)

plt.show()