import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from collections import Counter

import category_encoders as ce

import warnings

warnings.filterwarnings("ignore")

import datetime as dt

import matplotlib.pyplot as plt

%matplotlib inline
def what_is_dataset(data):

    print("[shape]\n"+str(data.shape[0])+','+str(data.shape[1]))

    print("[type]\n"+str(data.dtypes.value_counts()))

    Count=data.describe().T['count'].astype(int)

    Mean=data.describe().T['mean'].round(2)

    Max=data.describe().T['max'].round(2)

    Min=data.describe().T['min'].round(2)

    Count_missing = data.isnull().sum()

    Percent_missing = (data.isnull().sum()/data.isnull().count()*100).round(2)

    Count_unique = data.nunique()

    Percent_unique = (Count_unique / len(data)*100).round(2)

    Type = data.dtypes  

    data=[[i, Counter(data[i][data[i].notna()]).most_common(5)] for i in list(data)]

    top = pd.DataFrame(data, columns=['columns', 'Top_5']).set_index(['columns'])

    tmp = pd.concat([Count,Type,Count_missing, Percent_missing, Count_unique, Percent_unique,Mean,Max,Min], axis=1, 

                   keys=['Count','Type','n_missing', 'missing(%)', 'n_unique', 'unique(%)','mean','max','min'])

    tmp = pd.concat([tmp, top[['Top_5']]], axis=1, sort=False)

    

    return((tmp))
def day_Country_datasets(country):

    tmp_data = data[data['Country/Region'] == country]

    tmp_data = tmp_data.groupby('ObservationDate',as_index = False).sum()

    tmp_data['Active_rate'] = tmp_data['Active']/ tmp_data['Confirmed'] * 100

    tmp_data["Deaths_rate"] = tmp_data["Deaths"]/ tmp_data["Confirmed"] * 100

    tmp_data["Recovered_rate"] = tmp_data["Recovered"]/ tmp_data["Confirmed"] * 100

    return tmp_data
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data['ObservationDate'] = pd.to_datetime(data["ObservationDate"])#time-data

data_copy = data

data = data[data['ObservationDate'] >= dt.datetime(2020,2,15)]

data['ObservationDate'] = data['ObservationDate'].dt.strftime('%m-%d')

data['Active'] = data['Confirmed'] - data['Deaths'] - data['Recovered']

data.drop(['SNo','Last Update','Province/State'],axis = 1, inplace = True) #this column is not meaningful

data.head()
what_is_dataset(data)
data_Country = data.groupby('Country/Region',as_index = False).sum().sort_values(["Confirmed"],ascending = False).head(10)

data_Country['Active_rate'] = data_Country['Active']/ data_Country['Confirmed'] * 100

data_Country["Deaths_rate"] = data_Country["Deaths"]/ data_Country["Confirmed"] * 100

data_Country["Recovered_rate"] = data_Country["Recovered"]/ data_Country["Confirmed"] * 100

#data_Country
#Top 10 Confirmed

data_China = day_Country_datasets('Mainland China')

data_Italy = day_Country_datasets('Italy')

data_Korea = day_Country_datasets('South Korea')

data_Iran = day_Country_datasets('Iran')

data_Spain = day_Country_datasets('Spain')

data_France = day_Country_datasets('France')

data_Germany = day_Country_datasets('Germany')

data_US = day_Country_datasets('US')

data_Japan = day_Country_datasets('Japan')
data_Country_rate = data_Country.drop(['Confirmed',"Deaths",'Recovered','Active'],axis = 1)

data_Country_rate = data_Country_rate.set_index('Country/Region')

data_Country_rate
x = 'ObservationDate'

y = 'Confirmed'

plt.figure(figsize=(20, 10)) 

plt.plot(data_China[x], data_China[y] ,label = "Chi")
x = 'ObservationDate'

y = 'Confirmed'

plt.figure(figsize=(25, 13)) 

plt.plot(x,y, data = data_Italy, label = "Ita")

plt.plot(x,y, data = data_Korea, label = "Kor")

plt.plot(x,y, data = data_Iran, label = "Ira")

plt.plot(x,y, data = data_Spain, label = "Spa")

plt.plot(x,y, data = data_France, label = "Fra")

plt.plot(x,y, data = data_Germany, label = "Ger")

plt.plot(x,y, data = data_US, label = "US")

plt.plot(x,y, data = data_Japan, label = "Jap")





plt.legend(fontsize = 18 ,loc = 'upper left')
y = 'Active'

plt.figure(figsize=(25, 13)) 

plt.plot(x,y, data = data_Italy, label = "Ita")

plt.plot(x,y, data = data_Korea, label = "Kor")

plt.plot(x,y, data = data_Iran, label = "Ira")

plt.plot(x,y, data = data_Spain, label = "Spa")

plt.plot(x,y, data = data_France, label = "Fra")

plt.plot(x,y, data = data_Germany, label = "Ger")

plt.plot(x,y, data = data_US, label = "US")

plt.plot(x,y, data = data_Japan, label = "Jap")



plt.legend(fontsize = 18 ,loc = 'upper left')
plt.figure(figsize=(25, 13)) 

y = "Deaths_rate"

plt.plot(x,y, data = data_Italy, label = "Ita")

plt.plot(x, y, data = data_Korea, label = "Kor")

plt.plot(x,y,data = data_China,label = "Chi",linewidth = 5.0)

#plt.plot(x,y, data = data_Iran, label = "Ira")

plt.plot(x, y, data = data_Spain, label = "Spa")

plt.plot(x, y, data = data_France, label = "Fra")

plt.plot(x, y, data = data_Germany, label = "Ger")

plt.plot(x, y, data = data_US, label = "US")

plt.plot(x, y, data = data_Japan, label = "Jap")



plt.legend(fontsize = 18 ,loc = 'upper left')
plt.figure(figsize=(25, 13)) 

y = "Active_rate"

plt.plot(x,y, data = data_Italy, label = "Ita")

plt.plot(x, y, data = data_Korea, label = "Kor")

plt.plot(x,y,data = data_China,label = "Chi",linewidth = 5.0)

plt.plot(x,y, data = data_Iran, label = "Ira")

plt.plot(x, y, data = data_Spain, label = "Spa")

plt.plot(x, y, data = data_France, label = "Fra")

plt.plot(x, y, data = data_Germany, label = "Ger")

plt.plot(x, y, data = data_US, label = "US")

plt.plot(x, y, data = data_Japan, label = "Jap")



plt.legend(fontsize = 18 ,loc = 'upper left')
plt.figure(figsize=(25, 13)) 

y = "Recovered_rate"

plt.plot(x,y, data = data_Italy, label = "Ita")

plt.plot(x, y, data = data_Korea, label = "Kor")

plt.plot(x,y,data = data_China,label = "Chi",linewidth = 5.0)

plt.plot(x,y, data = data_Iran, label = "Ira")

plt.plot(x, y, data = data_Spain, label = "Spa")

plt.plot(x, y, data = data_France, label = "Fra")

plt.plot(x, y, data = data_Germany, label = "Ger")

plt.plot(x, y, data = data_US, label = "US")

plt.plot(x, y, data = data_Japan, label = "Jap")



plt.legend(fontsize = 18 ,loc = 'upper left')
x = 'ObservationDate'

y = 'Confirmed'

plt.figure(figsize=(25, 13)) 

plt.plot(data_Italy[x],np.log(data_Italy[y]), label = "Ita")

plt.plot(data_Korea[x],np.log(data_Korea[y]) ,label = "Kor")

plt.plot(data_Iran[x],np.log(data_Iran[y]), label = "Ira")

plt.plot(data_Spain[x], np.log(data_Spain[y]),label = "Spa")

plt.plot(data_France[x], np.log(data_France[y]),label = "Fra")

plt.plot(data_Germany[x],np.log(data_Germany[y]), label = "Ger")

plt.plot(data_US[x],np.log(data_US[y]), label = "US")

plt.plot(data_Japan[x],np.log(data_Japan[y]), label = "Jap")



#plt.xticks([0,5,10],["01","02","03"])

plt.ylabel("log(Confirmed)")



plt.legend(fontsize = 18 ,loc = 'upper left')
MSE_list = []

print("How close to China")

for i in range(10):

    MSE  = (data_Country_rate.iloc[i,0]-data_Country_rate.iloc[0,0])**2 / data_Country_rate.iloc[0,0]

    MSE += (data_Country_rate.iloc[i,1]-data_Country_rate.iloc[0,1])**2 / data_Country_rate.iloc[0,1]

    MSE += (data_Country_rate.iloc[i,2]-data_Country_rate.iloc[0,2])**2 / data_Country_rate.iloc[0,2]

    MSE_list.append(MSE/100)

data_Country_rate['China_metric']=MSE_list 

data_Country_rate["China_metric"]
data_ebola = pd.read_csv("../input/ebola-outbreak-20142016-complete-dataset/ebola_2014_2016_clean.csv")

data_ebola["Date"] = pd.to_datetime(data_ebola["Date"])

#data_ebola.Country.unique()

data_ebola.fillna(0, inplace = True)
what_is_dataset(data_ebola)
data_ebola_Guinea = data_ebola[data_ebola['Country'] == 'Guinea']

#data_ebola_Italy = data_ebola[data_ebola['Country'] == 'Italy']

what_is_dataset(data_ebola_Guinea)
x = 'Date'

plt.figure(figsize=(25, 13)) 

for y in ['No. of suspected cases', 'No. of probable cases',

       'No. of confirmed cases',

       'No. of confirmed, probable and suspected cases',

       'No. of suspected deaths', 'No. of probable deaths',

       'No. of confirmed deaths',

       'No. of confirmed, probable and suspected deaths']:

    plt.plot(x,y,data = data_ebola_Guinea,label = y) 



plt.legend(fontsize = 18 ,loc = 'upper left')