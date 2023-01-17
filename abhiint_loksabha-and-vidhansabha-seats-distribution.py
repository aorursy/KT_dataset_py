# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
numpy_list = np.array(os.listdir("../input"))

numpy_list
my_list = np.arange(10)

my_list
my_list2 = np.arange(1,11)

my_list2
my_list3 = np.arange(1,11,2)

my_list3
lin_arr = np.linspace(1, 3, 15)

lin_arr
#my_series = pd.Series(numpy_list,lin_arr)

index_list = np.arange(1,3)

my_series = pd.Series(numpy_list,index_list)     # even if you don't pass index_list, pandas will create index starting from 0

my_series

my_series[1]
my_series[3] = "change_series_value"

my_series
my_dataframe = pd.DataFrame(numpy_list,index_list)     # even if you don't pass index_list, pandas will create index starting from 0

my_dataframe  # each column here represents a panda Series, Hence, it is safe to say a DataFrame is a collection of Series sharing the same index.

my_dataframe[0][1]      #[column][row]
#csv = pd.read_csv(my_dataframe[0][1])

type(my_dataframe[0])
type(my_dataframe[0][1])
new_dataframe = pd.DataFrame(['lucknow','bhopal','mumbai','hyd','bangalore'],['UP','MP','Maharashtra','Telangana','Karnataka'],['capital'])  

new_dataframe
new_series = {'Capital' : pd.Series(['lucknow','bhopal','mumbai','hyd','bangalore']

                                    ,['UP','MP','Maharashtra','Telangana','Karnataka'])}

type(new_series)
new_dataframe2 = pd.DataFrame(new_series)  

new_dataframe2
new_dataframe2['country'] = pd.Series(['India','India','India','India'],['UP','MP','Maharashtra','Telangana'])

pd.DataFrame(new_dataframe2)           #adding new column to the dataframe
new_dataframe2.hist
new_dataframe2['population'] = pd.Series([199812341,72626809,112374333,35003674,61095297],['UP','MP','Maharashtra','Telangana','Karnataka'])

pd.DataFrame(new_dataframe2)
new_dataframe2.reset_index(inplace = True)              #reset_index deosn't permanently reset the index. For permanently, use reset_index(inplace=true)
new_dataframe2


new_dataframe2.set_index('index')        #this way you can set any column to work as index

new_dataframe2.columns
new_dataframe2.columns[0]
new_dataframe2.reset_index(inplace=True)

new_dataframe2.columns = ['level_0','State','Capital','Country','Population']

new_dataframe2
new_dataframe2["State"].value_counts()      
new_dataframe2.describe()
#help.describe

new_dataframe2.hist()

#plt.show()
new_dataframe2['NoOfLoksabhaConstituency'] = pd.Series([80,29,48,17,28],[0,1,2,3,4])

pd.DataFrame(new_dataframe2)      
new_dataframe2.hist()
#new_dataframe2.hist("Population")    wrong syntax

new_dataframe2["Population"].hist()
new_dataframe2.corr()
new_dataframe2['NoOfVidhansabhaConstituency'] = pd.Series([403,230,288,119,224])

new_dataframe2
new_dataframe2.hist()
new_dataframe2.corr()
new_dataframe2['Country'].dropna(inplace = True)
new_dataframe2
new_dataframe2.loc[4]
new_dataframe2.iloc[4]        #can get using index too
#new_dataframe2.loc[5] = pd.Series()      #to add rows

for i in range(8):

    if i<5:

        continue

    new_dataframe2.loc[i] = pd.Series()    

new_dataframe2
new_dataframe2.loc[1].loc['State']
new_dataframe2.loc[1].loc['State']
new_dataframe2.groupby('Country').mean()     #groupby group rows based on the 'Country' column and call the aggregate function .mean()on it
new_dataframe2.groupby('Country').count()      #Using the count() method, we can get the number of times an item occurs in a DataFrame.
new_dataframe2.groupby('Country').describe() 
new_dataframe2.plot()
new_dataframe2.plot(kind='scatter', x=6,y='Population')
new_dataframe2.corr()
new_dataframe2.plot.area(alpha=0.1)
new_dataframe2['Population'].plot(kind='kde')
new_dataframe2['NoOfLoksabhaConstituency'].plot(kind='kde')
new_dataframe2['NoOfVidhansabhaConstituency'].plot(kind='kde')
new_dataframe2
new_dataframe2['areaOfStates'] = pd.Series([243290,308350,307713,112077,191791])

new_dataframe2
new_dataframe2.corr()
new_dataframe2.loc[0].loc['Population']
areaPlusPopulationList = np.arange(1,7)

for i in range(6):

    areaPlusPopulationSeries = pd.Series(areaPlusPopulationList)

    areaPlusPopulationSeries[i] = new_dataframe2.loc[i].loc['Population'] + new_dataframe2.loc[i].loc['areaOfStates']

new_dataframe2['areaPlusPopulation'] = pd.Series(areaPlusPopulationSeries)



new_dataframe2
new_dataframe2.corr()
new_dataframe2
seriesOfState = pd.Series(['Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chhattisgarh','Goa','Gujarat','Haryana',

                          'Himachal Pradesh','Jammu and Kashmir','Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra',

                          'Manipur','Meghalaya','Mizoram','Nagaland','Odisha','Punjab','Rajasthan','Sikkim','Tamil Nadu',

                           'Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal','NCT of Delhi','Puducherry'])

seriesOfNoOfLoksabhaConstituency = pd.Series([25,2,14,40,11,2,26,10,4,6,14,28,20,29,48,2,2,1,1,21,13,25,1,39,17,2,80,5,42,

                                        7,1])

seriesOfNoOfVidhansabhaConstituency = pd.Series([175,60,126,243,90,40,182,90,68,87,81,224,140,230,288,60,60,40,60,

                                                147,117,200,32,234,119,60,403,70,294,70,30])

seriesOfAreaOfState = pd.Series([160205,83743,78438,94165,135191,3702,196024,44212,55673,101387,79714,191791,38863,308350,

                                307713,22327,22429,21081,16579,155707,50362,342238,7096,130058,112077,10486,243290,53483,

                                88752,1483,492])

seriesOfPopulationOfState = pd.Series([49577103,1383727,31205576,104099452,25545198,1458545,60439692,25351462,6864602,

                                      12541302,32988134,61095297,33406061,72626809,112374333,2570390,2966889,1097206,1978502,

                                      41974218,27743338,68548437,610577,72147030,35003674,3673917,199812341,10086292,91276115,

                                      16787941,1247953])

for i in range(31):

    new_dataframe2.loc[i] = pd.Series()    
new_dataframe2['State'] = seriesOfState

new_dataframe2['Population'] = seriesOfPopulationOfState

new_dataframe2['NoOfLoksabhaConstituency'] = seriesOfNoOfLoksabhaConstituency

new_dataframe2['NoOfVidhansabhaConstituency'] = seriesOfNoOfVidhansabhaConstituency

new_dataframe2['areaOfStates'] = seriesOfAreaOfState
new_dataframe2
areaPlusPopulationList = np.arange(0,31)

for i in range(31):

    areaPlusPopulationSeries = pd.Series(areaPlusPopulationList)

    areaPlusPopulationSeries[i] = new_dataframe2.loc[i].loc['Population'] + new_dataframe2.loc[i].loc['areaOfStates']

new_dataframe2['areaPlusPopulation'] = pd.Series(areaPlusPopulationSeries)
new_dataframe2
new_dataframe2.corr()