#Import all the necessary python libraries for this data analysis

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/world-development-indicators/Indicators.csv')

data.shape
data.head()
countries=data['CountryName'].unique().tolist()

len(countries)
countrycodes=data['CountryCode'].unique().tolist()

len(countrycodes)
indicators=data['IndicatorName'].unique().tolist()

len(indicators)
years=data['Year'].unique().tolist()

len(years)
print(min(years),'to' ,max(years))

# select Agriculture value added (current LCU) and GDP per capita (current LCU) in kenya.

hist_indicator1 = 'NV.AGR.TOTL.CN'

hist_indicator2 = 'NY.GDP.PCAP.CN'

hist_indicator3 = 'NE.EXP.GNFS.CN'

hist_country = 'KEN'



mask1 = data['IndicatorCode'].str.contains(hist_indicator1) 

mask2 = data['IndicatorCode'].str.contains(hist_indicator2)

mask3 = data['CountryCode'].str.contains(hist_country)

mask4=data['IndicatorCode'].str.contains(hist_indicator3)



# stage1 is just those indicators matching the KEN for country code and agriculture value added (current LCU).

#stage2 is those indicators matching KEN for country code and GDP per capita (current LCU)

stage1 = data[mask1 & mask3]

stage2=data[mask2 & mask3]

stage3=data[mask3 & mask4]
stage3.head()
stage2.head()

stage1.head()
#trends in Agriculture, value added (current LCU) 

stage1.head(5)
#line plot

years=stage1['Year'].values

agriculture= stage1['Value'].values



plt.plot(stage1['Year'].values , stage1['Value'].values)



plt.xlabel('Year')

plt.ylabel(stage1['IndicatorName'].iloc[0])



plt.title('AGRIC VALUE ADDED (CURRENT LCU) IN KENYA')



plt.grid(True)

plt.show
#trends in KEN	GDP per capita (current LCU) 

stage2.head(15)
#line plot

years=stage2['Year'].values

agriculture= stage2['Value'].values



plt.plot(stage2['Year'].values , stage2['Value'].values)



plt.xlabel('Year')

plt.ylabel(stage2['IndicatorName'].iloc[0])



plt.title('TREND IN GDP PER CAPITA(CURRENT LCU) IN KENYA')



plt.axis([1959,2015,0,130000])

plt.grid(True)

plt.show
#checking for missing values in the relevant part of the data set

print(stage1['Value'].isnull())

print(stage2['Value'].isnull())
stage2.loc[stage2.Year>=1999,:].head(10)

stage1.loc[stage1.Year>=1999,:].head(10)
# first descriptive statistics of both data frames 

stage1['Value'].describe()

stage2['Value'].describe()
stage2['Value']



#Create a new data frame called relation that merges stage1 and stage2

relation = stage1.merge(stage2, on='Year',how ='inner')

relation=relation.drop (columns=['CountryCode_y','CountryName_y'])

relation.head()
relation['Value_x'].corr(relation['Value_y'])


%matplotlib inline

import matplotlib.pyplot as plt



fig,axis=plt.subplots()



axis.yaxis.grid(True)

axis.set_title('AGRIC VALUE ADDED vs. GDP PER CAPITA', fontsize=10)

axis.set_xlabel(relation['IndicatorName_x'].iloc[0], fontsize=10)

axis.set_ylabel(relation['IndicatorName_y'].iloc[0], fontsize=10)



x=relation['Value_x']

y=relation['Value_y']









axis.scatter(x, y)

plt.show()

               
#Create a new data frame called relation1 that merges stage1 and stage3

relation1 = stage1.merge(stage3, on='Year',how ='inner')

relation1=relation1.drop (columns=['CountryCode_y','CountryName_y'])

relation1.head()
relation1['Value_x'].corr(relation1['Value_y'])
fig,axis=plt.subplots()



axis.yaxis.grid(True)

axis.set_title('AGRIC VALUE ADDED vs. EXPORTS OF GOODS AND SERVICES', fontsize=10)

axis.set_xlabel(relation1['IndicatorName_x'].iloc[0], fontsize=10)

axis.set_ylabel(relation1['IndicatorName_y'].iloc[0], fontsize=10)



x=relation1['Value_x']

y=relation1['Value_y']









axis.scatter(x, y)

plt.show()