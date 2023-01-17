# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import io
import requests
# import May15th stay-at-home overturn list
May15countiesCodes = pd.read_csv("https://raw.githubusercontent.com/theosaurusrex/StayAtHomeWIMay152020/master/May15saferhomeCodes%20(1).csv").set_index("Combined_Key", drop = True).loc[:,:]
May15countiesCodes = May15countiesCodes.sort_values('Combined_Key')
print(May15countiesCodes)
WI_county_pops = pd.read_json('../input/wi-county-pop-datajson/WI_county_pop_data.json').set_index('Combined_Key').sort_index()
WI_county_pops['pop2018'] = WI_county_pops['pop2018'].astype(float)
WI_county_pops['GrowthRate'] = WI_county_pops['GrowthRate'].astype(float)
WI_county_pops.head()
# WI_county_pops.info()

# categorize_label = lambda x: x.astype('float')

# # Convert df[LABELS] to a categorical type
# df[LABELS] = df[LABELS].apply(categorize_label,axis=0)

# # Print the converted dtypes
# print(df[LABELS].dtypes)
#This cell is not converting ints and floats, NaN problem for populations here
# May15countiesCodes = pd.concat([May15countiesCodes, WI_county_pops], axis=1, ignore_index=False)
May15countiesCodes = pd.merge(May15countiesCodes, WI_county_pops, how='left', on=['Combined_Key'])
May15countiesCodes.head()
# 'UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_','Combined_Key'
url="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
usconfirmed = pd.read_csv(url)
print(usconfirmed)

wiconfirmed =  usconfirmed.set_index("Province_State", drop = False).loc["Wisconsin",:]
print(wiconfirmed)
# wisconfirmed.loc["wisconsin":"wisconsin","2020":"2021"]
# usconfirmeddf = pd.DataFrame(data, columns =['fips', 'county', 'state', 'lat', 'long', 'date', 'cases', 'state_code', 'deaths'])
# print(usconfirmed.shape())
# print(usconfirmeddf.head(10))
wicountyconfirmedplot = wiconfirmed.set_index("Combined_Key", drop = False).loc[:,:]
print(wicountyconfirmedplot)
# Add May 15th column to wicountyconfirmedplot DF if first letters match
# merged_df = DF2.merge(DF1, how = 'inner', on = ['date', 'hours'])
# add column to wicountyconfirmedplot of 0 (zero)
wicountyconfirmedplot['StayHome']= 0
# If May15countiesCodes column StayHome = 1 then find by Combined code in wicountyconfirmedplot['combined_code'] and change wicountyconfirmedplot['StayHome'] to 1
# if May15countiesCodes['StayHome']=1
#  Use this: df1['pricesMatch?'] = np.where(df1['Price1'] == df2['Price2'], 'True', 'False')
# wicountyconfirmedplot['StayHome'] = np.where(wicountyconfirmedplot['StayHome'] == May15countiesCodes['StayHome'], True,False)
combinedSOHCountyCOVID = wicountyconfirmedplot[~wicountyconfirmedplot.isin(May15countiesCodes)]
# .dropna()

# May15countiesCodes.loc[May15countiesCodes['StayHome'] = 1, 'new column name'] = 'value if condition is met'

# wicountyconfirmedplot = wicountyconfirmedplot.merge(May15countiesCodes, how= 'inner',on = ['Combined_Key'])
# print(wicountyconfirmedplot)
print(combinedSOHCountyCOVID)
# Move stayathome column to second column location(1)
mid = combinedSOHCountyCOVID['StayHome']
combinedSOHCountyCOVID.drop(labels=['StayHome'], axis=1, inplace = True)
combinedSOHCountyCOVID.insert(1, 'StayHome', mid)
combinedSOHCountyCOVID
# May15countiesCodes
print(May15countiesCodes)

# StayHome values transfer
combinedSOHCountyCOVID['StayHome'] = May15countiesCodes['StayHome']
combinedSOHCountyCOVID
SimplecombinedSOHCountyCOVID = combinedSOHCountyCOVID.drop(['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_','Combined_Key'], axis=1)
SimplecombinedSOHCountyCOVID.head(20)
StayhomeAveDF = SimplecombinedSOHCountyCOVID.groupby(["StayHome"], as_index=False).mean()
# StayhomeAveDF.drop(['StayHome'])

# df1.loc[df2.index[0]] = df2.iloc[0]
StayhomeAveDF.loc[SimplecombinedSOHCountyCOVID.index[12]] = SimplecombinedSOHCountyCOVID.iloc[12]

StayhomeAveDF.rename(index={0:'No Safer-at-Home Order Mean',1:'County Safer-at-Home Order Mean',2:'Dane County'}, inplace=True)
StayhomeAveDF
# Plot StayhomeAveDF
lines = StayhomeAveDF.T.plot(figsize=(10,10))
plt.title('Infection Rates for Wisconsin Counties with and without Safer-at-Home Orders')
plt.xlabel("Date")
plt.ylabel("COVID Infections");
# https://pandas.pydata.org/pandas-docs/version/0.23.1/generated/pandas.DataFrame.plot.line.html
plt.show()
url="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
USDeathsdf = pd.read_csv(url, error_bad_lines=False)
df = pd.DataFrame(data, columns =['Name', 'Age', 'Score']) 
# df = pd.DataFrame(data, columns =['Name', 'Age', 'Score'])

print(confirmeddf.head(10))
print(confirmeddf.shape())

from google.colab import drive
drive.mount('/content/drive')
# import county stay-at-home data
# https://www.channel3000.com/list-of-counties-that-have-enacted-a-local-safer-at-home-order-through-may-26/
stayhomeMay15df = pd.read_csv("/content/drive/My Drive/covid-data-projects/May15saferhome.csv")
print(stayhomeMay15df.head())
# get wisconsin counties from Github datasets
# df.loc[df['column_name'] == some_value]
WIconfirmeddf = df.loc[usconfirmeddf['state_code']== 'wi']
# print(WIconfirmed.head())

# Combine data

# graph wisconsin counties COVID rates over time
# overlay point to line graph at point date for each county when stay-at-home order lifted
# Print graph
# plt.show()
