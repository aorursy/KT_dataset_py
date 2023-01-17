import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set()
data = pd.read_csv("../input/BreadBasket_DMS.csv")
data.index = pd.to_datetime(data.Date + " " + data.Time, format = "%Y/%m/%d %H:%M:%S") #format and index
data.drop(['Date', 'Time'], axis = 1, inplace = True) #drop the date and time columns as no longer needed
print("Total Record Period = {}".format(data.last_valid_index() - data.first_valid_index()))
#Plot top 10 items
fig = plt.figure(figsize = (20,5))
ax = plt.subplot(1,1,1)
ax.set_ylabel('Total Sold')
data.groupby('Item')['Item'].count().sort_values(ascending = False)[:10].plot.bar(color = 'b');
print("There are {} transactions".format(len(np.unique(data.Transaction))))
print("There are {} total sold items".format(data.Transaction.size))
print("Average No. Items per purchase = {}".format(round(data.groupby('Transaction')['Transaction'].count().mean(),2)))
#Resample to 24hrs/1Day sum of transaction counts
fig = plt.figure(figsize = (20,5))
ax = plt.subplot(1,1,1)
ax.set_ylabel('Daily Items Sold')
data.Transaction.resample('D').count().plot(c = 'g', ax = ax)
ax.axvline(data.Transaction.resample('D').count().idxmax().date(), c = 'b', label = "Max Sales")
ax.axvline(data.Transaction.resample('D').count().idxmin().date(), c = 'r', label = "Holidays")
plt.legend();
fig = plt.figure(figsize = (20,5))
ax1 = plt.subplot(1,2,1)
ax1.set_ylabel('Mean Items Sold Daily')
data.Transaction.groupby(data.index.weekday_name).mean().plot(kind = 'bar', ax = ax1, color = 'r')
ax2 = plt.subplot(1,2,2)
ax2.set_ylabel('Mean Items Sold Daily')
data.Transaction.groupby(data.index.month_name()).mean().plot(kind = 'bar', ax = ax2, color = 'purple');
fig = plt.figure(figsize = (20,5))
ax = plt.subplot(1,1,1)
data.Transaction.groupby([data.index.day, data.index.hour]).count().unstack().plot(kind = 'bar',ax = ax, alpha = 0.5, stacked = True, cmap = 'tab20')
ax.set_title('Transactions by Day of Month \n Stacked on Hour of Day')
ax.set_xlabel('Day of Month')
ax.set_ylabel('Total Number of Transactions')
plt.legend(ncol = 5);