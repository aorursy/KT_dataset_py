# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# I can't remember why I have this setting, but it solved something a while ago.

# (Stackoverflow solved this)



pd.options.mode.chained_assignment = None  # default='warn'
# import libraries



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from matplotlib.ticker import FormatStrFormatter

import matplotlib.ticker as ticker



import datetime



import seaborn as sns
# Import workbook as pandas dataframe

deliveries_df = pd.read_excel('/kaggle/input/deliveroo-data/Delivery History.xlsx', sheet_name=1)



# Drop rows with blank values - this is required due to some housekeeping I do in Excel

deliveries_df = deliveries_df.dropna()



deliveries_df
# Calculate base fee - the price for the delivery without a fee multiplier in place

deliveries_df["Base Fee"] = deliveries_df['Earnings'] / deliveries_df['Fee Multiplier']



# Calculate additional revenue from the fee multiplier

deliveries_df["Multiplier Bonus"] = deliveries_df['Earnings'] - deliveries_df['Base Fee']



# Calculate price per delivery for a stacked order

deliveries_df["Stack Order Adjust"] = deliveries_df["Base Fee"] / deliveries_df["Drops on Order"]



# Round values because currency

deliveries_df["Base Fee"] = deliveries_df["Base Fee"].round(2)

deliveries_df["Multiplier Bonus"] = deliveries_df["Multiplier Bonus"].round(2)

deliveries_df["Stack Order Adjust"] = deliveries_df["Stack Order Adjust"].round(2)



deliveries_df
deliveries_df.corr()
# heatmap of correlation plot



f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(deliveries_df.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)

plt.show()
# Donut chart plot of revenue breakdown



Base_fee_sum = deliveries_df["Base Fee"].sum().round(2)

Multiplier_sum = deliveries_df["Multiplier Bonus"].sum().round(2)

Tip_sum = deliveries_df["Tip"].sum().round(2)



Total_revenue = Base_fee_sum + Multiplier_sum + Tip_sum



Base_fee_pct = ((Base_fee_sum / Total_revenue) * 100).round(2)

Multiplier_pct = ((Multiplier_sum / Total_revenue) * 100).round(2)

Tip_pct = ((Tip_sum / Total_revenue) * 100).round(2)



Base_fee_label = "Base Fee {}%".format(Base_fee_pct)

Multiplier_label = "Fee Boost {}%".format(Multiplier_pct)

Tip_label = "Tips {}%".format(Tip_pct)



# create data groups

names = Base_fee_label, Multiplier_label, Tip_label



size_of_groups = [Base_fee_sum, Multiplier_sum, Tip_sum]



# Create a pieplot

# from palettable.colorbrewer.qualitative import Pastel1_7

plt.pie(size_of_groups, labels=names, wedgeprops = {'linewidth' : 6, 'edgecolor' : 'white' }, colors = ['#5f00cc', '#cc9600', '#cc006d'], textprops={'fontsize': 14}, radius = 2.5)

        

# add a circle at the center

my_circle = plt.Circle((0,0), 1.2, color='white')

p = plt.gcf()

p.gca().add_artist(my_circle)

 

plt.show()
# convert Start Time to a usable format



deliveries_df["Start Time"] = pd.to_datetime(deliveries_df["Start Time"], format="%H:%M:%S")
# Define maximum recieved fee to plot scatter graphs better

Fee_max = deliveries_df["Stack Order Adjust"].max()+1
# Scatter plot of Fee by Time of Day



x = deliveries_df["Start Time"]

y = deliveries_df["Revenue"]



fig, ax = plt.subplots(figsize=(15, 10))



plt.title('Delivery Fee By Time Order Was Recieved', fontsize= 18)

plt.xlabel("Time", fontsize= 16)

plt.ylabel("Adjusted Delivery Price", fontsize= 16)



myFmt = mdates.DateFormatter('%H:%M')



# plot graph

plt.plot_date(x, y, linestyle='none', marker='h', markersize=5)

ax.xaxis.set_major_formatter(myFmt)

ax.yaxis.set_major_formatter(FormatStrFormatter('£%.2f'))

fig.autofmt_xdate()

#ax.set_ylim(0, Fee_max)



plt.xticks(fontsize=14)

plt.yticks(fontsize=14)





plt.show()
# Scatter plot of Fee by Distance



x = deliveries_df["Distance (km)"]

y = deliveries_df["Stack Order Adjust"]



z = np.polyfit(x, y, 1)

p = np.poly1d(z)



fig, ax = plt.subplots(figsize=(15, 10))



plt.plot(x, y, linestyle='none', marker='o', markersize=8)

plt.plot(x,p(x), 'r')



plt.title('Delivery Fee Variance by Distance Travelled', fontsize= 18)

plt.xlabel("Distance (km)", fontsize= 16)

plt.ylabel("Delivery Price", fontsize= 16)



ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.yaxis.set_major_formatter(FormatStrFormatter('£%.2f'))

ax.set_ylim(0, Fee_max)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)





plt.show()
import seaborn as sns

import matplotlib.pyplot as plt



df = deliveries_df

sns.set_style("ticks")

#sns.set(font_scale=1.3)



fig = sns.lmplot(x="Distance (km)", y="Stack Order Adjust", data=df, fit_reg=True, hue='Fee Multiplier',

           markers=["d", "v", "x"], palette = ['#5f00cc', '#cc006d', '#cc9600'], height=20, aspect=1, scatter_kws={"s":250})

import seaborn as sns

import matplotlib.pyplot as plt



df = deliveries_df

sns.set_style("ticks")

#sns.set(font_scale=1.3)



fig = sns.lmplot(x="Distance (km)", y="Stack Order Adjust", data=df, fit_reg=True, hue='Drops on Order',

           legend=False, markers=["d", "v"], palette = ['#5f00cc', '#cc006d'],height=20, aspect=1, scatter_kws={"s":500})



#50



fig.ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

fig.ax.set_xlim(0, 5)

fig.ax.yaxis.set_major_formatter(FormatStrFormatter('£%.2f'))

fig.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))



fig.ax.set_xlabel("Distance (km)",fontsize=5)

fig.ax.set_ylabel("Adjusted Delivery Price",fontsize=5)



# Move the legend to an empty part of the plot

mylabels = ["1 Drop", "2 Drops"]

plt.legend(labels=mylabels, loc='lower right', prop={'size': 10})



plt.show()
shifts_df = pd.read_excel('/kaggle/input/deliveroo-data/Delivery History.xlsx', sheet_name=0)

shifts_df = shifts_df.dropna()



shifts_df['Riding Distance Efficiency'] = shifts_df['Riding Distance Efficiency']*100



shifts_df
# Box & Scatter plot of deliveries



plt.figure(figsize=(15, 3))



sns.boxplot(x = 'Avg Deliveries/hr', data = shifts_df,

            palette=["#5f00cc"],

            fliersize = 0, orient = 'h')



# Add a scatterplot for each category.

sns.stripplot(x = 'Avg Deliveries/hr', data = shifts_df, s=7,

     linewidth = 2.5, 

              palette=["#ffffff"],

              orient = 'h')



plt.xlabel('Deliveries/hr', fontsize= 16)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)



plt.tight_layout()
# Box & Scatter plot of deliveries



plt.figure(figsize=(15, 3))



sns.boxplot(x = 'Riding Distance Efficiency', data = shifts_df,

            palette=['#cc006d'],

            fliersize = 0, orient = 'h')



# Add a scatterplot for each category.

sns.stripplot(x = 'Riding Distance Efficiency', data = shifts_df, s=7,

     linewidth = 2.5, 

              palette=["#ffffff"],

              orient = 'h')



plt.xlabel('Riding Efficiency', fontsize= 16)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)



plt.tight_layout()
# Box & Scatter plot of deliveries



plt.figure(figsize=(15, 3))



sns.boxplot(x = 'Avg Earnings/hr', data = shifts_df,

            palette=["#cc9600"],

            fliersize = 0, orient = 'h')



# Add a scatterplot for each category.

sns.stripplot(x = 'Avg Earnings/hr', data = shifts_df, s=7,

     linewidth = 2.5, 

              palette=["#ffffff"],

              orient = 'h')



plt.xlabel('Equivalent Hourly Wage', fontsize= 16)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)



plt.tight_layout()
# Box & Scatter plot of earning variation



plt.figure(figsize=(15, 3))



sns.boxplot(y = 'DOTW', x = 'Avg Earnings/hr', data = shifts_df,

     #palette=["#a985c6", "#3882d6"],

            fliersize = 0, orient = 'h')



# Add a scatterplot for each category.

sns.stripplot(y = 'DOTW', x = 'Avg Earnings/hr', data = shifts_df,

     linewidth = 0.6, 

              #palette=["#a985c6", "#3882d6"],

              orient = 'h')



plt.title('Earnings Variation by Day of The Week', fontsize= 18)

plt.xlabel('Effective £/hr', fontsize= 16)

plt.ylabel('Day', fontsize= 16)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)



plt.tight_layout()
# Box & Scatter plot of number of drops/hr



plt.figure(figsize=(15, 3))



sns.boxplot(y = 'DOTW', x = 'Avg Deliveries/hr', data = shifts_df,

     #palette=["#a985c6", "#3882d6"],

            fliersize = 0, orient = 'h')



# Add a scatterplot for each category.

sns.stripplot(y = 'DOTW', x = 'Avg Deliveries/hr', data = shifts_df,

     linewidth = 0.6, 

              #palette=["#a985c6", "#3882d6"],

              orient = 'h')



plt.title('Variation of delivery throughput by Day of The Week', fontsize= 18)

plt.xlabel('Deliveries/hr', fontsize= 16)

plt.ylabel('Day', fontsize= 16)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)



plt.tight_layout()
# Box & Scatter plot of number of riding efficiency



# Riding efficiency defined as the % amount of distance that was spent delivering items.



plt.figure(figsize=(15, 3))



sns.boxplot(y = 'DOTW', x = 'Riding Distance Efficiency', data = shifts_df,

     #palette=["#a985c6", "#3882d6"],

            fliersize = 0, orient = 'h')



# Add a scatterplot for each category.

sns.stripplot(y = 'DOTW', x = 'Riding Distance Efficiency', data = shifts_df,

     linewidth = 0.6, 

              #palette=["#a985c6", "#3882d6"],

              orient = 'h')



plt.title('Delivery Efficiency Variation by Day of The Week', fontsize= 18)

plt.xlabel('Riding Efficiency (%)', fontsize= 16)

plt.ylabel('Day', fontsize= 16)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)



plt.tight_layout()