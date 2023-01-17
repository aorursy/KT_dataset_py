import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#input data
bakery = pd.read_csv('../input/BreadBasket_DMS.csv')
#dataset info
bakery.info()
# missing value
bakery.isnull().sum()
# check none value
print("There are " + str(sum(bakery['Date'] == 'NONE')) + (" NONE value(s) on Date column"))
print("There are " + str(sum(bakery['Time'] == 'NONE')) + (" NONE value(s) on Date column"))
print("There are " + str(sum(bakery['Item'] == 'NONE')) + (" NONE value(s) on Date column"))
#removing none value from dataset
bakery_new = bakery[bakery['Item'] != 'NONE']
#make a new column for splitting date
bakery_new['Day'] = bakery_new['Date'].apply(lambda x: x.split("-")[2])
bakery_new['Month'] = bakery_new['Date'].apply(lambda x: x.split("-")[1])
bakery_new['Year'] = bakery_new['Date'].apply(lambda x: x.split("-")[0])
bakery_new['Hour'] = bakery_new['Time'].apply(lambda x: x.split(":")[0])
bakery_new.head(20)
# number of transaction
startdate = bakery_new.iloc[0,0]
enddate = bakery_new.iloc[-1,0]
numoftransaction = bakery_new['Transaction'].nunique()

print("From " + str(startdate) + " to " + str(enddate) + " there are " + str(numoftransaction) + " transactions at the bakery.")
#grouping transaction by time
monthly = bakery_new.groupby('Month')['Transaction'].nunique()
daily = bakery_new.groupby('Day')['Transaction'].nunique()
hourly = bakery_new.groupby('Hour')['Transaction'].nunique()

#make it as dataframe
monthly_sales = pd.DataFrame({'Month':monthly.index, 'Transaction':monthly.values})
daily_sales = pd.DataFrame({'Day':daily.index, 'Transaction':daily.values})
hourly_sales = pd.DataFrame({'Hour':hourly.index, 'Transaction':hourly.values})
#plot monthly transaction
plt.figure(figsize=(20,8))
plt.bar(monthly_sales['Month'], monthly_sales['Transaction'], color = 'green')
plt.grid(True)
plt.ylabel('Total\n', size = 15)
plt.xlabel('\nMonth', size = 15)
plt.xticks(monthly_sales['Month'],['January','February','March','April','October','November','December'])
plt.title('Number of Transaction Per Month\n', fontweight="bold", fontsize = 21)
plt.show()
#plot daily transaction
plt.figure(figsize=(20,8))
plt.plot(daily_sales['Day'], daily_sales['Transaction'])
plt.scatter(daily_sales['Day'], daily_sales['Transaction'], color = 'red')
plt.grid(True)
plt.ylabel('Total\n', size = 15)
plt.xlabel('\nDay - n', size = 15)
plt.xticks(rotation = 45)
plt.title('Number of Transaction Per Day\n', fontweight="bold", fontsize = 21)
plt.show()
#plot hourly transaction
plt.figure(figsize=(20,8))
plt.plot(hourly_sales['Hour'], hourly_sales['Transaction'])
plt.scatter(hourly_sales['Hour'], hourly_sales['Transaction'], color = 'red')
plt.bar(hourly_sales['Hour'], hourly_sales['Transaction'], color = 'pink', alpha = 0.5)
plt.grid(True)
plt.ylabel('Total\n', size = 15)
plt.xlabel('\nHour - n', size = 15)
plt.xticks(rotation = 45)
plt.title('Number of Transaction Per Hour\n', fontweight="bold", fontsize = 21)
plt.show()
#Items in bakery
bakery_item = bakery_new['Item'].nunique()

print("There are " + str(bakery_item) + " items sold at the bakery.")
#best seller items at bakery
item = bakery_new['Item'].value_counts()
total_item = pd.DataFrame({'Item':item.index, 'Total':item.values}).head(10)
plt.figure(figsize=(20,8))
plt.bar(total_item['Item'], total_item['Total'], color = 'red')
plt.grid(True)
plt.ylabel('Total\n', size = 15)
plt.xticks(rotation = 45)
plt.title('Best Seller at Bakery\n', fontweight="bold", fontsize = 21)
plt.show()