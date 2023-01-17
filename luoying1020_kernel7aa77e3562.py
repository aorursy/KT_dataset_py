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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
E_seller = pd.read_csv('../input/online-shoppers-intention/online_shoppers_intention.csv')

E_seller.head()
print(E_seller.info())
E_seller[['OperatingSystems', 'Browser', 'Region', 'TrafficType']]=E_seller[['OperatingSystems', 'Browser', 'Region', 'TrafficType']].astype(str)

E_seller=E_seller.fillna(0)

print(E_seller.info())
E_seller.describe()
# Transaction Not Completed VS. Transaction Completed



reve_count=E_seller['Revenue'].value_counts()



transaction=reve_count.plot.bar(width=0.8, color=['gray', 'orange']) 

transaction.set_title('Whether or not the transaction happened?', color='red', fontsize=16)

transaction.set_xlabel('Transaction Happened', color='black', fontsize=14)

transaction.set_ylabel('Visits Count', color='black', fontsize=14)

transaction.set_xticklabels(('False', 'True'), rotation='horizontal')

transaction.text(-0.17, 9200, str((~E_seller['Revenue']).sum()), fontsize=15, color='white', fontweight='bold')

transaction.text(0.88, 800, str(E_seller['Revenue'].sum()), fontsize=15, color='white', fontweight='bold')



plt.show()
# Sum Visit Weekday vs. Sum Visit Weekend



weekend_count=E_seller['Weekend'].value_counts()



weekend=weekend_count.plot.bar(width=0.8, color=['gray', 'orange']) 

weekend.set_title('Whether or not visits on weekend?', color='red', fontsize=16)

weekend.set_xlabel('On Weekend', color='black', fontsize=14)

weekend.set_ylabel('Visits Count', color='black', fontsize=14)

weekend.set_xticklabels(('False', 'True'), rotation='horizontal')

weekend.text(-0.12, 8500, str((~E_seller['Weekend']).sum()), fontsize=15, color='white', fontweight='bold')

weekend.text(0.88, 1800, str(E_seller['Weekend'].sum()), fontsize=15, color='white', fontweight='bold')



plt.show()





# Averave Daily Visit Weekday vs. Average Daily Visit Weekend



x=[i for i in range(1,3)]

y=[(~E_seller['Weekend']).sum()/5, E_seller['Weekend'].sum()/2]



bars = plt.bar(x, height=y, width=.7, color=['gray', 'orange'])



xlocs, xlabs = plt.xticks()

xlocs=[i for i in x]

xlabs=[i for i in x]



plt.title('Whether more daily average visits on weekday or weekend?', color='red', fontsize=16)

plt.xlabel('Visit On', color='black', fontsize=14)

plt.ylabel('Average Daily Visit', color='black', fontsize=14)

plt.xticks(xlocs, ('Weekday', 'Weekend'))



for bar in bars:

    yval = bar.get_height()

    plt.text(bar.get_x()+0.18, yval-200, yval, fontsize=15, color='white', fontweight='bold')

    

plt.show()
# Average Daily transaction on Weekdays VS. on Weekends



weekend_revenue_avg = E_seller.query('Weekend == True & Revenue == True')['Revenue'].count()/2

weekday_revenue_avg = E_seller.query('Weekend == False & Revenue == True')['Revenue'].count()/5



b=[weekday_revenue_avg, weekend_revenue_avg]

a=[j for j in range(len(b))]



bars = plt.bar(a, height=b, width=.7, color=['gray', 'orange'])



xlocs, xlabs = plt.xticks()



xlocs=[j for j in a]

xlabs=[j for j in b]



plt.title('Whether more daily transaction on weekdays or weedends?', color='red', fontsize=16)

plt.xlabel('Transaction On', color='black', fontsize=14)

plt.ylabel('Average Transaction', color='black', fontsize=14)

plt.xticks(xlocs, ('Weekday', 'Weekend'))



for bar in bars:

    yval = bar.get_height()

    plt.text(bar.get_x()+0.22, yval-30, yval, fontsize=15, color='white', fontweight='bold')

    

plt.show()





# Average Daily Transaction Rate on Weekdays VS. on Weekends



weekend_revenue_pct = round(E_seller.query('Weekend == True & Revenue == True')['Revenue'].count()/E_seller.query('Weekend == True')['Revenue'].count()*100, 2)

weekday_revenue_pct = round(E_seller.query('Weekend == False & Revenue == True')['Revenue'].count()/E_seller.query('Weekend == False')['Revenue'].count()*100, 2)



d=[weekday_revenue_pct, weekend_revenue_pct]

c=[k for k in range(len(d))]



bars = plt.bar(c, height=d, width=.7, color=['gray', 'orange'])



xlocs, xlabs = plt.xticks()



xlocs=[k for k in c]

xlabs=[k for k in d]



plt.title('Whether higher daily transaction rate on weekdays or weedends?', color='red', fontsize=16)

plt.xlabel('Transaction On', color='black', fontsize=14)

plt.ylabel('Average Transaction', color='black', fontsize=14)

plt.xticks(xlocs, ('Weekday', 'Weekend'))



for bar in bars:

    yval = bar.get_height()

    plt.text(bar.get_x()+0.22, yval-2, str(yval)+'%', fontsize=15, color='white', fontweight='bold')

# Regions Online Visitors In



region_count=E_seller['Region'].value_counts()

region=region_count.plot.barh(width=0.6) 

region.set_title('Which regions visitors online shopping in?', color='red', fontsize=16)

region.set_xlabel('Visitor Count', color='black', fontsize=14)

region.set_ylabel('Region', color='black', fontsize=14)



for i, v in enumerate(region_count):

    region.text(v+3, i-0.2, str(v), color='black', fontsize=10, fontweight='bold')



plt.show()





# Important Regions with Most Online Visitors



region1 = E_seller.query('Region == "1"')['Region'].count()

region2 = E_seller.query('Region == "2"')['Region'].count()

region3 = E_seller.query('Region == "3"')['Region'].count()

region4 = E_seller.query('Region == "4"')['Region'].count()

region_rest = E_seller['Region'].count() - region1 - region2 - region3 -region4



region_value = [region1, region3, region2, region4, region_rest]

region_name = ['Region 1', 'Region 3', 'Region 2', 'Region 4', 'Region 5~9']

region_colors = ['orange', 'orange', 'gray', 'gray', 'gray']

plt.pie(region_value, labels=region_name, colors=region_colors, startangle=60, explode=(0.1, 0.06, 0.03, 0.03, 0.03), autopct='%.1f%%', textprops={'fontsize': 12})

plt.title('Which regions are more import?', color='red', fontsize=16)



plt.show()
# Important Regions with Most Online Transaction



region1 = E_seller.query('Region == "1" & Revenue == True')['Region'].count()

region2 = E_seller.query('Region == "2" & Revenue == True')['Region'].count()

region3 = E_seller.query('Region == "3" & Revenue == True')['Region'].count()

region4 = E_seller.query('Region == "4" & Revenue == True')['Region'].count()

region_rest = E_seller.query('Revenue == True')['Region'].count() - region1 - region2 - region3 -region4



region_value = [region1, region3, region2, region4, region_rest]

region_name = ['Region 1', 'Region 3', 'Region 2', 'Region 4', 'Region 5~9']

region_colors = ['orange', 'orange', 'gray', 'gray', 'gray']

plt.pie(region_value, labels=region_name, colors=region_colors, startangle=60, explode=(0.1, 0.06, 0.03, 0.03, 0.03), autopct='%.1f%%', textprops={'fontsize': 12})

plt.title('Which regions are more valuable?', color='red', fontsize=16)



plt.show()
# Operating systems visitors use



system_count=E_seller['OperatingSystems'].value_counts()

system=system_count.plot.barh(width=0.6) 

system.set_title('Which operating systems visitors prefer?', color='red', fontsize=18)

system.set_xlabel('Visitor Count', color='black', fontsize=14)

system.set_ylabel('Operating System', color='black', fontsize=14)



for i, v in enumerate(system_count):

    system.text(v+100, i-0.1, str(v), color='black', fontsize=10, fontweight='bold')



plt.show()





# Browsers visitors use



browser_count=E_seller['Browser'].value_counts()

browser=browser_count.plot.barh(width=0.6) 

browser.set_title('Which browser visitors prefer?', color='red', fontsize=18)

browser.set_xlabel('Visitor Count', color='black', fontsize=14)

browser.set_ylabel('Browser', color='black', fontsize=14)



for i, v in enumerate(browser_count):

    browser.text(v+100, i-0.2, str(v), color='black', fontsize=10, fontweight='bold')



plt.show()
# Valuables related to completed transaction



corr_df = E_seller.query('Revenue == True & Weekend == True')

E_seller_corr = corr_df.select_dtypes(include=['float64'])

corr = E_seller_corr.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(E_seller_corr.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(E_seller_corr.columns)

ax.set_yticklabels(E_seller_corr.columns)

plt.show()



corr.style.background_gradient(cmap='coolwarm')