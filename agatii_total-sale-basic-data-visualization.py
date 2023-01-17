# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


import numpy as np 

import pandas as pd



# Plotting

import matplotlib as mpl

import matplotlib.pyplot as plt



import datetime



# Statistical graphics

import seaborn as sns 



## To Show graphs in same window

%matplotlib inline
#import dataset

sell_day = pd.read_csv('../input/Day_sell_24_12_18.csv', delimiter=';', decimal=',')
sell_df=pd.read_csv('../input/SELL_1.csv', delimiter=';', decimal=',',encoding='latin-1')
#check sell_df

sell_df.head()

sell_df.info()

sell_df.describe()

sell_df.isnull().sum



#sprawdzenie typów zmiennych

sell_df.dtypes
#change type of Date

sell_df['Date'] = pd.to_datetime(sell_df['Date'],format="%d.%m.%Y")

sell_df.dtypes

sell_df.head(2)

sell_df['Date'].isnull().any

sell_df['Month']=sell_df['Date'].dt.strftime('%b')

sell_df['Month'].unique()
#check of data sell_day

sell_day.head()



sell_day.info()



sell_day.describe()

#checking missing data

sell_day.isnull().sum



#variables type

sell_day.dtypes
#Check data sell_df

sell_df.head()

sell_df.info()

sell_df.describe()

sell_df.isnull().sum



#check data type

sell_df.dtypes



#drop column pudz_sb

new_sell_df=sell_df.drop(['pudzsb'],axis=1)

new_sell_df.info()
#Plot gross sale value per month

sell_df.groupby('Date')['pwa_sb'].sum().plot(color='blueviolet')

plt.ylabel('Value of sales [PLN]')

plt.xlabel('Month')

plt.title('Gross sale value per month in 2018')

plt.show()



#Plot margin per month

sell_df.groupby('Date')['pkwmarza'].sum().plot(color='violet')

plt.ylabel('Sum of margin [PLN]')

plt.xlabel('Month')

plt.title('Margin per month in 2018')

plt.show()
#prepering columns for date.

##change type to date

sell_day['Date'] = pd.to_datetime(sell_day['Date']) 

sell_day.info()

#new columns for date

sell_day['f_year'] = sell_day['Date'].dt.year

sell_day['f_month'] = sell_day['Date'].dt.month

sell_day['f_day'] = sell_day['Date'].dt.day

sell_day['f_weekday'] = sell_day['Date'].dt.weekday



#check of new columns

sell_day.head(6)

sell_day.dtypes
#change type of new columns from float to object

sell_day['f_year'] = sell_day['f_year'].astype('object')

sell_day['f_month'] = sell_day['f_month'].astype('object')

sell_day['f_day'] = sell_day['f_day'].astype('object')

sell_day['f_weekday'] = sell_day['f_weekday'].astype('object')
#Visualisation

##Day grouping (column f_weekday)

### 0-Monday, 1-Tuesday, 2-Wensday, 3-Thursday, 4-Friday, 5-Saturday, 6-Sunday

 

#line chart of sum and avarage sale per day wykres liniowy sumy i sredniej sprzedazy w danym dniu

#sum

sell_day.groupby('f_weekday')['sb'].sum().plot.bar(color='plum')

plt.ylabel('Sum of sale [PLN]')

plt.xlabel('Day of week')

plt.title('Summary day sale', color='indigo')

bars=('Monday','Tuesday','Wensday','Thursday','Friday','Saturday','Sunday')

y_pos = np.arange(len(bars))

plt.xticks(y_pos, bars)

plt.show()



#avarage

sell_day.groupby('f_weekday')['sb'].mean().plot.bar(colormap='Paired')

plt.ylabel('Avarage sale [PLN]')

plt.xlabel('Day of week')

plt.title('Average day sale',weight='bold', color='slategrey')

bars=('Monday','Tuesday','Wensday','Thursday','Friday','Saturday','Sunday')

y_pos = np.arange(len(bars))

plt.xticks(y_pos, bars)

plt.show()

#drop of not needed columns

sell_day_marza=sell_day.drop([

              'zn',

              'tax'], axis=1)



##Only sundays dataset

ndz  = sell_day_marza.query("f_weekday == 6").sort_values(by=['Date'])

ndz.head(12) 



###### Sunday trade free: 11.03.18, 18.03.18,

####1.04.18, 08.04.18, 15.04.18, 22.04.18, 13.05.18, 20.05.18, 10.06.18, 17.06.18, 

####08.07.18, 15.07.18, 22.07.18, 12.08.18, 19.08.18, 09.09.18, 16.09.18, 23.09.18, 

####14.10.18, 21.10.18, 11.11.18, 18.11.18, 09.12.18

#### 2018-10-21 shop closed



##Trade free Sundays new dataset

ndz_nh= ndz[ndz['Date'].isin(['2018-03-11','2018-03-18','2018-04-01','2018-04-08',

'2018-04-15','2018-04-22','2018-05-13','2018-05-20','2018-06-10','2018-06-17',

'2018-07-08','2018-07-15','2018-07-22','2018-08-12','2018-08-19','2018-09-09','2018-09-16',

'2018-09-23','2018-10-14','2018-10-21','2018-11-11','2018-11-18','2018-12-09'])]

ndz_nh.head(40)



#new column "Trade free"

ndz_nh["Trade_free"]='Yes'

ndz_nh.head()



#concat of dataset ndz i ndz_nh:

concat = pd.concat([ndz, ndz_nh])



#drop duplicates

ndz_all = concat.drop_duplicates(keep=False, inplace=False)

ndz_all.head()

ndz_all.isnull().sum



#fill NaN in column "Trade free"

ndz_all['Trade_free'] = ndz_all.Trade_free.fillna('No')

ndz_all.info()

ndz_all.head()

ndz_all.isnull().sum



#grouping

ndz_all_group = ndz_all.groupby('Trade_free')        
#sale plot bar

ndz_all_sb=ndz_all_group['sb'].mean()

ndz_all_sb.head()

ndz_all_sb.plot.bar(color='chocolate')

plt.ylabel('Avarage sell [PLN/day]')

plt.xlabel('Was the Sunday trade free?')

plt.title('Comparison of sell on trade and trade free Sundays')

plt.xticks( rotation='horizontal')

plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.show()



#margin plot bar

ndz_all_marza=ndz_all_group['marza'].mean()

ndz_all_marza.plot.bar(color='goldenrod')

plt.ylabel('Avarage margin [PLN]')

plt.xlabel('Was the Sunday trade free?')

plt.title('Comparison of margin on trade and trade free Sundays')

plt.xticks( rotation='horizontal')

plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.show()
#procentage of difference if trade or not

increase=ndz_all_marza['Yes']/ndz_all_marza['No']

print(increase/ndz_all_marza['No']*100)
#check unique observations of column Pgrupa

sell_df.Pgroup.unique()



#change groups: 'WINE_ALCOHOL 18%', 'BEER','VODKA','VODKA_ALCOHOL' to one ALCOHOL

sell_df.replace(['WINE_ALCOHOL 18%', 'BEER','VODKA','VODKA_ALCOHOL'], ['ALCOHOL','ALCOHOL','ALCOHOL','ALCOHOL'], inplace=True)

sell_df.Pgroup.unique()
#Group data by Pgrupa

grouped_sell_df = sell_df.groupby('Pgroup')



###Visualization

#Bar chart with 20 the highest sell amount among product groups

sell_df_pgrupa_ilosc=grouped_sell_df['Pquantity'].sum().sort_values(ascending=False)



sell_df_pgrupa_ilosc.head(20).plot.bar(color='deeppink')

plt.ylabel('Quanty [pices]')

plt.xlabel('Group name')

plt.title('20 most sold quantity per product group')

plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.show()
#bar chart 20 the most sell value product group

sell_df_pgrupa_pwasb=grouped_sell_df['pwa_sb'].sum().sort_values(ascending=False)

sell_df_pgrupa_pwasb.head(20).plot.bar()

plt.ylabel('Sale value [PLN]')

plt.xlabel('Group name')

plt.title('Sale value per group name')

plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.show()
#import liblary for colours

from itertools import cycle, islice



#lista kolorow do wyswietlenia

my_colors = list(islice(cycle(['indigo','purple', 'hotpink', 'royalblue', 'orchid', 'lightpink']), None, len(grouped_sell_df)))

my_colors_1 = list(islice(cycle(['c','darkcyan', 'powderblue', 'lightskyblue', 'lightblue', 'skyblue']), None, len(grouped_sell_df)))



# Graph for 20 product groups with the highest margin

marza_sum = grouped_sell_df['pkwmarza'].sum().sort_values(ascending=False)

marza_sum.head(20).plot.bar(color=my_colors_1)

plt.ylabel('Sum of margin [PLN]')

plt.xlabel('Group name')

plt.xticks(fontsize = 7)

plt.title('20 product groups with the highest sum of margin',fontsize = 14, weight = 'bold')

plt.show()

#select group of product- alcohol 

sell_df_alkohol= sell_df.query('Pgroup == "ALCOHOL"')

sell_df_alkohol.head()



#plot 20 products with highest sum of margin in ALCOHOL group

sell_df_alkohol1=sell_df_alkohol.groupby('Pname')['pkwmarza'].sum().sort_values(ascending=False)

sell_df_alkohol1.head()

sell_df_alkohol1.tail(10)

sell_df_alkohol1.head(20).plot.bar(color= ('g'))

plt.ylabel('Sum of Margin [PLN]')

plt.xlabel('Product name in polish')

plt.title('20 products with highest sum of margin in ALCOHOL group',color='sienna')

plt.show()
#graph  20 product group with the highest mean of margin in % 

marza_mean=grouped_sell_df['pmarza'].mean().sort_values(ascending=False)

marza_mean.head(20).plot.bar(color=my_colors)

plt.ylabel('Mean of margin [%]')

plt.xlabel('Product group')

plt.title('20 product group with the highest mean of margin')

plt.show()

#graph  20 product group with the highest mean of margin in %

marza_median=grouped_sell_df['pmarza'].median().sort_values(ascending=False)

marza_median.head(20).plot.bar(color='c')

plt.ylabel('Median of margin [%] ')

plt.xlabel('Product group')

plt.title('20 product group with the highest median of margin')

plt.show()
#boxplot margin of products in group alcohol

df_boxplot1= sell_df.loc[sell_df['Pgroup']=='ALCOHOL']

sns.boxplot(x='Pgroup',y='pmarza',data=df_boxplot1)

plt.xlabel('Group name',weight='bold', color='brown')

plt.ylabel('Margin [%]')

plt.title('Margin for products in group Alcohol', fontsize = 14,color='indigo')

plt.show()
#boxplot margin of products in group bread

df_boxplot1= sell_df.loc[sell_df['Pgroup']=='BREAD']

sns.boxplot(x='Pgroup',y='pmarza',data=df_boxplot1)

plt.xlabel('Group name',weight='bold', color='brown')

plt.ylabel('Margin [%]')

plt.title('Margin for products in group Bread', fontsize = 14,color='k')

plt.show()
sell_corr1=sell_df[['Pquantity','pmarza']]

plt.scatter(sell_corr1['Pquantity'],sell_corr1['pmarza'])

plt.title("Corelation between quantity and margin ")

plt.xlabel("Quantity[pieces]")

plt.ylabel("Margin [%]")

plt.show()
#corelation matrix

sell_co=sell_df.drop(['PKod','pudzmarza','pkwmarza',], axis=1)

corr = sell_co.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)