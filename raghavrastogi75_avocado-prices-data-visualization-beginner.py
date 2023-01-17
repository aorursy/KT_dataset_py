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
avocado = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')

print(avocado.head())

#print(avocado['region'].unique())

#print(avocado['year'].unique())
avocado['Date'] = avocado['Date'].astype('Datetime64')

avocado_albany_c = avocado.loc[(avocado['region'] == 'Albany') & (avocado['type'] == 'conventional')].sort_values(by = 'Date')

avocado_albany_o = avocado.loc[(avocado['region'] == 'Albany') & (avocado['type'] == 'organic')].sort_values(by = 'Date')

#fig,ax = plt.subplots()

plt.figure(figsize = (20,5))

plt.plot(avocado_albany_c.loc[avocado['type'] == 'conventional']["Date"],avocado_albany_c['AveragePrice'],'bo-',label = 'Conventnioal')

plt.plot(avocado_albany_o.loc[avocado['type'] == 'organic']["Date"],avocado_albany_o['AveragePrice'],'go-',label = 'Organic')

plt.xlabel('Date',fontsize = 20)

plt.ylabel('Average prices in Albany (USD)',fontsize = 20)

plt.xticks(avocado_albany_c['Date'][::20])

plt.legend()

plt.show()



avocado_Sacramento_c = avocado.loc[((avocado['region'] == 'Sacramento')) & (avocado['type'] == 'conventional')].sort_values(by = 'Date')

avocado_Sacramento_o = avocado.loc[((avocado['region'] == 'Sacramento')) & (avocado['type'] == 'organic')].sort_values(by = 'Date')

plt.figure(figsize = (20,5))

plt.plot(avocado_Sacramento_c["Date"],avocado_Sacramento_c['AveragePrice'],'bo-',label = 'Conventional')

plt.plot(avocado_Sacramento_o["Date"],avocado_Sacramento_o['AveragePrice'],'go-', label = 'Organic')

plt.xlabel('Date',fontsize = 20)

plt.ylabel('Average prices in Sacramento (USD)',fontsize = 20)

plt.xticks(avocado_Sacramento_o['Date'][::20])

plt.legend()

plt.show()



#Finding all unique regions in the dataset

#print(avocado['region'].unique())



regions = ['Albany','Atlanta','BaltimoreWashington','Boise','Boston','BuffaloRochester','California','Charlotte','Chicago','CincinnatiDayton','Columbus','DallasFtWorth','Denver','Detroit','GrandRapids','GreatLakes','HarrisburgScranton','HartfordSpringfield','Houston','Indianapolis','Jacksonville','LasVegas','LosAngeles','Louisville','MiamiFtLauderdale','Midsouth','Nashville','NewOrleansMobile','NewYork','Northeast','NorthernNewEngland','Orlando','Philadelphia','PhoenixTucson','Pittsburgh','Plains','Portland','RaleighGreensboro','RichmondNorfolk','Roanoke','Sacramento','SanDiego','SanFrancisco','Seattle','SouthCarolina','SouthCentral','Southeast','Spokane','StLouis','Syracuse','Tampa','TotalUS','West','WestTexNewMexico']

avocado_region = avocado.loc[(avocado['region'] == 'Sacramento')].sort_values(by = 'Date')

plt.figure(figsize = (20,5))

for i in range(0,len(regions),10):

    

    avocado_region = avocado.loc[(avocado['region'] == regions[i])].sort_values(by = 'Date')

    #plt.figure(figsize = (20,5))

    plt.plot(avocado_region["Date"],avocado_region['AveragePrice'],label = regions[i])

    plt.xlabel('Date',fontsize = 20)

    plt.ylabel('Average prices in (USD)',fontsize = 20)

    plt.xticks(avocado_region['Date'][::50])

plt.legend()



plt.show()

    

average_prices = avocado.groupby([avocado['region'],avocado['Date'].dt.strftime('%Y %B')])['AveragePrice'].mean().sort_index()

#print(average_prices.loc['Albany'])

plt.figure(figsize=(20,5))

regions = ['Albany','Atlanta','BaltimoreWashington','Boise','Boston','BuffaloRochester','California']

for i in regions:

    plt.plot(average_prices.loc[i],label = i)



#print(avocado['year'].unique())

plt.xticks(range(0,40,5))

plt.xlabel('Date')

plt.ylabel('Prices in USD')

plt.legend()

plt.show()

avocado.head()



plu4046 = avocado['4046'].sum()

plt.bar(['4046','4225','4770'],[avocado['4046'].sum(),avocado['4225'].sum(),avocado['4770'].sum()])

plt.yticks(range(1,10000000000,500000000))

plt.show()


#Trying to map it with the average price during this time

average_price_region = avocado.groupby('Date')['AveragePrice'].mean()

#print(average_price_region)



fig = plt.figure(figsize = (20,5))

ax = fig.add_subplot(111)

#ax.figure(figsize = (20,5))

ax.plot(avocado.groupby('Date')['4046'].sum(),'b.-',label = '4046 sold')

ax2 = ax.twinx()

ax2.plot(average_price_region,'r-', label = 'Avg price')

ax.legend()

#ax.grid()

ax.set_xlabel('Date')

ax.set_ylabel('Goods sold')

ax2.set_ylabel('average price in USD')

ax2.set_ylim(1,2)

ax2.legend(loc=2)

plt.title('Comparison of "4046" goods sold with the average price')



#ax.set_ylim(-20,100)

plt.show()



average_price_region = avocado.groupby('Date')['AveragePrice'].mean()

#print(average_price_region)



fig = plt.figure(figsize = (20,5))

ax = fig.add_subplot(111)

#ax.figure(figsize = (20,5))

ax.plot(avocado.groupby('Date')['4225'].sum(),'g.-',label = '4225 sold')

#avocado['Date'].sort_values(),avocado['4225']

ax2 = ax.twinx()

ax2.plot(average_price_region,'r-', label = 'Avg price')

ax.legend()

#ax.grid()

ax.set_xlabel('Date')

ax.set_ylabel('Goods sold')

ax2.set_ylabel('average price in USD')

ax2.set_ylim(1,2)

ax2.legend(loc=2)

plt.title('Comparison of "4225" goods sold with the average price')



#ax.set_ylim(-20,100)

plt.show()



average_price_region = avocado.groupby('Date')['AveragePrice'].mean()

#print(average_price_region)



fig = plt.figure(figsize = (20,5))

ax = fig.add_subplot(111)

#ax.figure(figsize = (20,5))

ax.plot(avocado.groupby('Date')['4770'].sum(),'y.-',label = '4770 sold')

ax2 = ax.twinx()

ax2.plot(average_price_region,'r-', label = 'Avg price')

ax.legend()

#ax.grid()

ax.set_xlabel('Date')

ax.set_ylabel('Goods sold')

ax2.set_ylabel('average price in USD')

ax2.set_ylim(1,2)

ax2.legend(loc=2)

plt.title('Comparison of "4770" goods sold with the average price')



#ax.set_ylim(-20,100)

plt.show()



year_sales = avocado.groupby('year')["Total Volume"].sum()

print(year_sales)
region_sales = avocado.groupby('region')['Total Volume'].sum().drop(['TotalUS'])

regions_total = avocado['region'].unique()

#removing TotalUS entry

regions_total = np.delete(regions_total,-3)



plt.figure(figsize = (20,5))

plt.bar(regions_total,region_sales)

plt.xticks(rotation= 90)

plt.xlabel('Regions')

plt.ylabel('Total volume sold')

plt.title('Total volume sold region wise')

plt.show()
fig,ax = plt.subplots(2,1,figsize=(20,10))



avocados_org = avocado.loc[avocado['type'] == 'organic']

avocados_con = avocado.loc[avocado['type'] == 'conventional']



ax[0].plot(avocados_org.groupby(avocado['Date'])['Total Volume'].sum(),'g.-',label = 'Organic')

ax[1].plot(avocados_con.groupby(avocado['Date'])['Total Volume'].sum(),'b.-',label = 'Conventional')

ax[1].set_xlabel('Date')

ax[1].set_ylabel('Total volume sold (conventional)')

ax[0].set_xlabel('Date')

ax[0].set_ylabel('Total volume sold (organic)')

ax[0].set_title('Organic avocados sold')

ax[1].set_title('Conventional avocados sold')





#print(avocado.head())

import seaborn as sns



avocado_org = avocado.loc[avocado['type'] == 'organic']

avocado_con = avocado.loc[avocado['type'] == 'conventional']



avocado_date_org = avocado_org.groupby('Date')

avocado_date_con = avocado_org.groupby('Date')



sns.jointplot(x = "AveragePrice", y = 'Total Volume', data = avocado_org,height = 20,ratio = 10)

sns.jointplot(x = "AveragePrice", y = 'Total Volume', data = avocado_con,height = 20,ratio = 10)
#Calculating the revenue for each line item

avocado['revenue'] = avocado['AveragePrice']*avocado['Total Volume']



avocado_org = avocado.loc[avocado['type'] == 'organic']

avocado_con = avocado.loc[avocado['type'] == 'conventional']



# avocado_date_org = avocado_org.groupby('Date')

# avocado_date_con = avocado_org.groupby('Date')



print(int(avocado_org['revenue'].sum()),"--Revenue generated by organic Avocados")

print(int(avocado_con['revenue'].sum()),"--Revenue generated by conventional Avocados")




