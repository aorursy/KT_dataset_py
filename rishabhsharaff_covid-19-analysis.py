# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_path1="../input/covid19testing/tested_worldwide.csv"
file_path2="../input/corona-virus-report/covid_19_clean_complete.csv"
test=pd.read_csv(file_path1)
test2=pd.read_csv(file_path2)
test.rename(columns={'Country_Region':'Country','active':'Active','positive':'total_cases'},inplace=True)

test['Date']=pd.to_datetime(test['Date'],dayfirst=True)
test['Week']=test['Date'].dt.weekofyear
test.head(50)
test2.head()
test_countrywise=test.groupby('Country').max()
pd.set_option('display.max_rows',None)
#test_countrywise.
test_countrywise.sort_values(by='total_cases',ascending=False).reset_index().style.background_gradient(cmap='Blues',subset=['total_cases','Active'])\
                                                                               .background_gradient(cmap='Reds',subset=['death'])\
                                                                               .background_gradient(cmap='Greens',subset=['recovered'])

                    
#Droping the columns with missing values
test_countrywise1=test_countrywise.drop(columns=['Province_State','Date','Active','hospitalized','hospitalizedCurr','recovered','death','daily_tested','daily_positive','Week'])
test_countrywise1 =test_countrywise1.reset_index()
test_countrywise1['Confirmed Cases per 100 tests']=(test_countrywise1['total_cases']/test_countrywise1['total_tested'])*100
test_countrywise1.head()

#Working on the 2nd dataset and creating two more columns 
test2['Date']=pd.to_datetime(test2['Date'])
test2.rename(columns={'Country/Region':'Country'},inplace=True)
test2['Active']=test2['Confirmed']-test2['Deaths']-test2['Recovered']
test2=test2[['Date','Country','Confirmed','Active','Deaths','Recovered']]
test3=test2.groupby('Country').max().reset_index()
test3['Mortality Rate']=(test3['Deaths']/test3['Confirmed'])*100
test3.head()
#Arranging the countries based on the cases reported-
test3.sort_values(by='Confirmed',ascending=False).reset_index(drop=True)

test_countrywise2=test_countrywise[['total_tested','total_cases','Active','death','daily_tested','daily_positive']]
test_countrywise2.sort_values(by='total_cases',ascending=False).reset_index().style.background_gradient(cmap='Blues',subset=['total_cases','Active'])\
                                                                                   .background_gradient(cmap='Reds',subset=['death'])\
                                                                                   .background_gradient(cmap='Greens',subset=['daily_tested'])\
                                                                                   .background_gradient(cmap='Greens',subset=['daily_tested'])

final_data=pd.merge(test_countrywise1,test3)
final_data.drop(columns=['Date'],inplace=True)
final_data.style.background_gradient(cmap='Blues',subset=['total_cases','Active'])\
                .background_gradient(cmap='Reds',subset=['Deaths','Mortality Rate'])\
                .background_gradient(cmap='Greens',subset=['total_tested','Recovered'])\
                .background_gradient(cmap='PuBu',subset=['Confirmed Cases per 100 tests'])

test.set_index(['Date'])
test.head()
#Making an 1D array of all the countries-
Country=pd.unique(test['Country'])
Country.sort()
np.where(Country=='India')
#This the array containing the names of all countries in the dataframe-
Country

for i in range(Country.shape[0]):
  Country[i]=test[test['Country']==Country[i]]
    
    
#Considering the data of all the states in the US-
Country[112]=Country[112][Country[112]['Province_State']=="All States"]

India=Country[44]
India.head()
India_weekwise=India.groupby('Week').max()
India_weekwise
for j in range(Country.shape[0]):
    Country[j]=Country[j].groupby('Week').max()
for j in range(Country.shape[0]):
    Country[j]=Country[j].drop(columns=['Province_State','Active','hospitalized','hospitalizedCurr','recovered','death','daily_tested','daily_positive'])
    Country[j]['Confirmed Cases per 100 tests']=(Country[j]['total_cases']/Country[j]['total_tested'])*100
    
Country[44]
Country[112]
sns.lineplot(x=Country[112].index,y=Country[112].total_cases)
sns.lineplot(x=Country[112].index,y=Country[112].total_tested)
plt.ylabel("Number of individuals")
plt.legend(['Total cases','Total tested'])
plt.show()
test2['Week']=test2['Date'].dt.weekofyear
Country_array=pd.unique(test2.Country)
Country_array.sort()
Country_array.shape[0]
Country_array
np.where(Country_array=="India")
test2.head()
for i in range(Country_array.shape[0]):
    Country_array[i]=test2[test2['Country']==Country_array[i]]
   

for i in range(Country_array.shape[0]):
    Country_array[i]=Country_array[i].groupby('Week').max()
Country_array[79]
''