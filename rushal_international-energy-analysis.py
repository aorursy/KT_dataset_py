# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
energy_data = pd.read_csv('../input/all_energy_statistics.csv')
energy_data.head()
energy_data.info()
energy_data.category.unique()
energy_data.isnull().sum()
def production(year_list,energy_calculation,data):
    #Year_list : List of Year 
    #energey_calculation : Which type energy we want to calculate
    #data : data set of dataFrame object sort by values with year
    value = []
    for yr in year_list:
        data_year = data[data.year == yr]
        quantity = data_year[data_year.commodity_transaction == energy_calculation]['quantity']
        if len(quantity) == 0:
            value.append(0)
        else:
            value.append(quantity.values.item())
        
    return value
def select_country(country):
    # pass the value list format
    data = energy_data[energy_data.country_or_area.isin(country)].sort_values('year').reset_index()
    data.drop('index',axis=1,inplace=True)
    return data,data.year.unique().tolist()
canada_country_dataframe,canada_year_lt = select_country(['Canada'])
canada_wind_producation = production(canada_year_lt,'Electricity - total wind production',canada_country_dataframe)

india_country_dataframe,india_year_lt = select_country(["India"])
india_wind_producation = production(india_year_lt,'Electricity - total wind production',india_country_dataframe)

australia_country_dataframe,australia_year_lt = select_country(['Australia'])
australia_wind_producation = production(australia_year_lt,'Electricity - total wind production',australia_country_dataframe)

china_country_dataframe,china_year_lt = select_country(["China"])
chian_wind_producation = production(china_year_lt,'Electricity - total wind production',china_country_dataframe)

japan_country_dataframe,japan_year_lt = select_country(["Japan"])
japan_wind_producation = production(japan_year_lt,'Electricity - total wind production',japan_country_dataframe)

uk_country_dataframe,uk_year_lt = select_country(['United Kingdom'])
uk_wind_producation = production(uk_year_lt,'Electricity - total wind production',uk_country_dataframe)

us_country_dataframe,us_year_lt = select_country(['United States'])
us_wind_producation = production(us_year_lt,'Electricity - total wind production',us_country_dataframe)

singapore_country_dataframe,singapore_year_lt = select_country(['Singapore'])
singapore_wind_producation = production(singapore_year_lt,'Electricity - total wind production',singapore_country_dataframe)

finland_country_dataframe,finland_year_lt = select_country(['Finland'])
finland_wind_producation = production(finland_year_lt,'Electricity - total wind production',finland_country_dataframe)

france_country_dataframe,france_year_lt = select_country(['France'])
france_wind_producation = production(france_year_lt,'Electricity - total wind production',france_country_dataframe)
plt.figure(figsize=(14,8))
plt.plot(india_year_lt,india_wind_producation,label='India')
plt.plot(canada_year_lt,canada_wind_producation,label='Canada')
plt.plot(australia_year_lt,australia_wind_producation,label="Australia")
plt.plot(china_year_lt,chian_wind_producation,label="China")
plt.plot(japan_year_lt,japan_wind_producation,label='Japan')
plt.plot(uk_year_lt,uk_wind_producation,label='United Kingdom')
plt.plot(us_year_lt,us_wind_producation,label="United States")
plt.plot(singapore_year_lt,singapore_wind_producation,label="Singapore")
plt.plot(finland_year_lt,finland_wind_producation,label='Finland')
plt.plot(france_year_lt,france_wind_producation,label='France')
plt.title("Total Electricity(Wind Producation) per year")

plt.legend(loc='best')
#Wind Producation comparision on India,China,US
plt.figure(figsize=(15,8))
plt.bar(np.array(india_year_lt),india_wind_producation,0.25,label="India",color='yellowgreen')
plt.bar(np.array(china_year_lt)+0.25-0.5,chian_wind_producation,0.25,label="China",color='c')
plt.bar(np.array(us_year_lt)+0.25,us_wind_producation,0.25,label="United States",color='peru')
plt.legend()
plt.title("Wind Producation of Year wise",fontsize=20)
#Wind Producation comparasion of India,Canada,Uk
plt.figure(figsize=(15,8))
plt.bar(np.array(india_year_lt),india_wind_producation,0.25,label="India Wind Producation",color='g')
plt.bar(np.array(canada_year_lt)+0.25-0.5,canada_wind_producation,0.25,label="Canada Wind Producation",color='r')
plt.bar(np.array(uk_year_lt)+0.25,uk_wind_producation,0.25,label="UK Wind Producation",color='b')
plt.title("Wind Producation of Year wise",fontsize=20)
plt.legend()
plt.figure(figsize=(15,8))
sns.regplot(np.array(us_year_lt),np.array(us_wind_producation),order=3)
sns.regplot(np.array(china_year_lt),np.array(chian_wind_producation),order=3)
sns.regplot(np.array(india_year_lt),np.array(india_wind_producation),order=3)
plt.legend(['US',"China",'India'],loc='best')
plt.title("Regression plot for wind Producation between US,China and India",fontsize=18)
canada_country_dataframe,canada_year_lt = select_country(['Canada'])
canada_nuclear_production = production(canada_year_lt,'Electricity - total nuclear production',canada_country_dataframe)

india_country_dataframe,india_year_lt = select_country(["India"])
india_nuclear_producation = production(india_year_lt,'Electricity - total nuclear production',india_country_dataframe)

australia_country_dataframe,australia_year_lt = select_country(['Australia'])
australia_nuclear_producation = production(australia_year_lt,'Electricity - total nuclear production',australia_country_dataframe)

china_country_dataframe,china_year_lt = select_country(["China"])
chian_nucler_producation = production(china_year_lt,'Electricity - total nuclear production',china_country_dataframe)

japan_country_dataframe,japan_year_lt = select_country(["Japan"])
japan_nuclear_producation = production(japan_year_lt,'Electricity - total nuclear production',japan_country_dataframe)

uk_country_dataframe,uk_year_lt = select_country(['United Kingdom'])
uk_nuclear_producation = production(uk_year_lt,'Electricity - total nuclear production',uk_country_dataframe)

us_country_dataframe,us_year_lt = select_country(['United States'])
us_nuclear_producation = production(us_year_lt,'Electricity - total nuclear production',us_country_dataframe)

singapore_country_dataframe,singapore_year_lt = select_country(['Singapore'])
singapore_nuclear_producation = production(singapore_year_lt,'Electricity - total nuclear production',singapore_country_dataframe)

finland_country_dataframe,finland_year_lt = select_country(['Finland'])
finland_nuclear_producation = production(finland_year_lt,'Electricity - total nuclear production',finland_country_dataframe)

france_country_dataframe,france_year_lt = select_country(['France'])
france_nuclear_producation = production(france_year_lt,'Electricity - total nuclear production"',france_country_dataframe)
plt.figure(figsize=(14,8))
plt.plot(india_year_lt,india_nuclear_producation,label='India')
plt.plot(canada_year_lt,canada_nuclear_production,label='Canada')
plt.plot(australia_year_lt,australia_nuclear_producation,label="Australia")
plt.plot(china_year_lt,chian_nucler_producation,label="China")
plt.plot(japan_year_lt,japan_nuclear_producation,label='Japan')
plt.plot(uk_year_lt,uk_nuclear_producation,label='United Kingdom')
plt.plot(us_year_lt,us_nuclear_producation,label="United States")
plt.plot(singapore_year_lt,singapore_nuclear_producation,label="Singapore")
plt.plot(finland_year_lt,finland_nuclear_producation,label='Finland')
plt.plot(france_year_lt,france_nuclear_producation,label='France')
plt.title("Total Electricity(Nuclear Producation) per year")

plt.legend(loc='best')
#Nuclear Electricity Producation comparasion of India,Canada,Uk
plt.figure(figsize=(15,8))
plt.bar(np.array(india_year_lt),india_nuclear_producation,0.25,label="India Nuclear Producation",color='g')
plt.bar(np.array(canada_year_lt)+0.25-0.5,canada_nuclear_production,0.25,label="Canada Nuclear Producation",color='r')
plt.bar(np.array(uk_year_lt)+0.25,uk_nuclear_producation,0.25,label="UK Nuclear Producation",color='b')
plt.title("Nuclear(Electricity) Producation of Year wise",fontsize=20)
plt.legend()
plt.figure(figsize=(15,8))
sns.regplot(np.array(us_year_lt),np.array(us_nuclear_producation),order=3)
sns.regplot(np.array(china_year_lt),np.array(chian_nucler_producation),order=3)
sns.regplot(np.array(india_year_lt),np.array(india_nuclear_producation),order=3)
plt.legend(['US',"China",'India'],loc='best')
plt.title("Regression plot for Nuclear(Electricity) Producation between US,China and India",fontsize=18)
