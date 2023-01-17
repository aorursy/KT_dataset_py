# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data=pd.read_csv("../input/Melbourne_housing.csv")

data.head()
%matplotlib inline

data.Price.hist(bins=100)
data['date']=pd.to_datetime(data.Date)

data['year']=data.date.dt.year

data['month']=data.date.dt.year

data['day']=data.date.dt.year

data['weekday']=data.date.dt.dayofweek

prices_df=data.groupby(['Suburb','year','Type','Rooms'],as_index=False).Price.median()

prices_df.columns=['Suburb','year','Type','Rooms','median_price']

prices_df.head()
data=data.merge(prices_df,on=['Suburb','year','Type','Rooms'])

data['Sold_Above_Median']=0

data.loc[data.Price>data.median_price,'Sold_Above_Median']=1

data.head()
bins=[0,5,10,15,20]

import numpy as np

data['distance_bins']=np.digitize(data.Distance,bins)

data.distance_bins.value_counts()
df_zone=data.groupby(['year','Type','Rooms','distance_bins'],as_index=False).Price.median()

df_zone.columns=['year','Type','Rooms','distance_bins','Zone_price']

data=data.merge(df_zone,on=['year','Type','Rooms','distance_bins'])

data['Sold_Above_Zone_Median']=0

data.loc[data.Price>data.Zone_price,'Sold_Above_Zone_Median']=1
import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

import seaborn as sns

melbourne_house_prices=data.loc[data.Type=='h','Price'].median()

melbourne_unit_prices=data.loc[data.Type=='u','Price'].median()



inner_zone=data[data.distance_bins==1]

middle_zone=data[data.distance_bins==2]

outer_zone=data[data.distance_bins==3]

vouter_zone=data[data.distance_bins==4]



fig1=sns.kdeplot(inner_zone.Price,color='red',shade=True)

fig2=sns.kdeplot(middle_zone.Price,color='orange',shade=True)

fig3=sns.kdeplot(outer_zone.Price,color='green',shade=True)

fig4=sns.kdeplot(vouter_zone.Price,color='blue',shade=True)



plt.legend(['inner','middle','outer','v outer'])

#So lets see sale for house in inner_zone

inner_zone_h=inner_zone[inner_zone.Type=='h']

median_price_houses_inner_zone=inner_zone_h.Price.median()



figr=inner_zone_h.boxplot(by='Suburb',column='Price',rot=90)



plt.axhline(y=median_price_houses_inner_zone,color='red',linestyle='dashed')

plt.title('Inner zone Houses ')

inner_zone_u=inner_zone[inner_zone.Type=='u']



median_price_units_inner_zone=inner_zone_u.Price.median()



figr=inner_zone_u.boxplot(by='Suburb',column='Price',rot=90)

plt.axhline(y=median_price_units_inner_zone,color='red',linestyle='dashed',label='Zone unit prices')

plt.axhline(y=melbourne_unit_prices,color='blue',linestyle='dashed',label='Melbourne unit prices')



plt.title('Inner zone Unit ')





middle_zone_h=middle_zone[middle_zone.Type=='h']

figr=middle_zone_h.boxplot(by='Suburb',column='Price',rot=90)

middle_zone_median_h=middle_zone_h.Price.median()

plt.axhline(y=middle_zone_median_h,color='red',linestyle='dashed')

plt.axhline(y=melbourne_house_prices,color='blue',linestyle='dashed')



plt.title('Middle ring House prices')

middle_zone_u=middle_zone[middle_zone.Type=='u']

figr=middle_zone_u.boxplot(by='Suburb',column='Price',rot=90)

middle_zone_median_u=middle_zone_u.Price.median()

plt.axhline(y=middle_zone_median_u,color='red',linestyle='dashed')

plt.axhline(y=melbourne_unit_prices,color='blue',linestyle='dashed')



plt.title('Middle ring Unit prices')
outer_zone_h=outer_zone[outer_zone.Type=='h']

figr=outer_zone_h.boxplot(by='Suburb',column='Price',rot=90)

outer_zone_median_h=outer_zone_h.Price.median()

plt.axhline(y=outer_zone_median_h,linestyle='dashed',color='red')

plt.axhline(y=melbourne_house_prices,color='blue',linestyle='dashed')



plt.title('Outer ring houses')

outer_zone_u=outer_zone[outer_zone.Type=='u']

figr=outer_zone_u.boxplot(by='Suburb',column='Price',rot=90)

outer_zone_median_u=outer_zone_u.Price.median()

plt.axhline(y=outer_zone_median_u,linestyle='dashed',color='red')

plt.axhline(y=melbourne_unit_prices,color='blue',linestyle='dashed')



plt.title('Outer ring Units')
data_2br_unit=data[(data.Type=='u')&(data.Rooms==2)]



df=data_2br_unit.groupby(['Suburb','year'],as_index=False).Price.median()

pv=df.pivot_table(index='Suburb',columns='year',values='Price')

pv['trend']='unknown'

pv.loc[pv[2017]>pv[2016],'trend']='up'

pv.loc[pv[2017]<pv[2016],'trend']='down'

pv.loc[pv[2017]==pv[2016],'trend']='same'

data_2br_unit=data_2br_unit.merge(pv,left_on='Suburb',right_index=True)



df_1=data_2br_unit[data_2br_unit.distance_bins==1]



median_df_1=df_1.Price.median()



df=pd.crosstab(df_1.Suburb,df_1.Sold_Above_Zone_Median).apply(lambda x:x*100/sum(x),axis=1).merge(df_1[['Suburb','trend']],left_index=True,right_on='Suburb')

df=df.loc[:,[0,1,'trend','Suburb']].drop_duplicates()

df[df.trend=='up']


df_2=data_2br_unit[data_2br_unit.distance_bins==2]



median_df_2=df_2.Price.median()



df=pd.crosstab(df_2.Suburb,df_2.Sold_Above_Zone_Median).apply(lambda x:x*100/sum(x),axis=1).merge(df_2[['Suburb','trend']],left_index=True,right_on='Suburb')

df=df.loc[:,[0,1,'trend','Suburb']].drop_duplicates()

df[df.trend=='up']