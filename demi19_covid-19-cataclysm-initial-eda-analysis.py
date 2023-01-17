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

import seaborn as sns

%matplotlib inline



dataset = pd.read_csv('/kaggle/input/covid19/train(1).csv')

dataset = pd.DataFrame(dataset)

#dataset.columns = 'Id','Province_State','Country_Region', 'Date', 'ConfirmedCases', 'Fatalities'

from pandas.plotting import scatter_matrix

scatter_matrix(dataset, diagonal = 'kde')

dataset.dtypes
dataset.loc[dataset['Province_State'].isnull(),'Province_State'] = dataset['Country_Region']



#Changing Datatypes as per the specification

dataset[['Province_State', 

         'Country_Region']] = dataset[['Province_State', 

         'Country_Region']].astype('category')

dataset[['ConfirmedCases', 'Fatalities']] = dataset[['ConfirmedCases', 'Fatalities']].astype(int)                                       

dataset['Date'] = pd.to_datetime(dataset.Date)

#dataset['Date'] = dataset.Date.strip(when, '%Y-%m-%d').date()

#dataset['Date'] = dataset['Date'].strftime("%x")

dataset.dtypes
# Replacing nan Values in States

# dataset.Province_State = dataset.Province_State ['Province_State']isnull == True].mask(dataset['Country_Region'])  

# State= dataset.Province_State.fillna(dataset.Country_Region, inplace=True)

#  def f(x):

#     if np.isnan(x['Province_State']):

#         return x['Country_Region']

#     else:

#         return x['Province_State']





#Adjusting Facttable

dataset = dataset.rename(columns={"ID": "Id", "Province_State": "Region", "Country_Region": "Country" , "DATE": "Date" , "ConfirmedCases": "Confirmed_Cases" , "Fatalities": "Noof_Deaths"})

new_order = [0,2,1,3,4,5]

dataset = dataset[dataset.columns[new_order]]
#EDA

No_of_AffectedNations = dataset.Country.nunique()

Affected_Countries = []

Affected_Countries = dataset.Country.unique()

Affected_Countries = pd.DataFrame(data = Affected_Countries)

#Affected_Countries = Affected_Countries.rename(columns={"0": "Distinct_Nation"})



Affected_Countries_Cases = dataset['Confirmed_Cases'].groupby(dataset['Country']).unique()

Affected_Countries_Cases = Affected_Countries_Cases.to_frame()

#ffected_Countries_Cases = dataset['Noof_Deaths'].groupby(dataset['Country']).unique()



Fatalities_Cross_Nations  = []

Fatalities_Cross_Nations = dataset['Noof_Deaths'].groupby(dataset['Country']).unique()

Affected_Countries_Cases['Fatalities_Cross_Nations'] = Fatalities_Cross_Nations

  

Var_Bin1 = Affected_Countries_Cases["Confirmed_Cases"] 

Length_of_Feature = len(Var_Bin1)

for i in range(0,Length_of_Feature):

    current_rows_arr_len = len(Var_Bin1[i])

    for j in range(0,current_rows_arr_len):

        if j == current_rows_arr_len-1:

            Var_Bin1[i] = Var_Bin1[i][j]



Var_Bin2 = Affected_Countries_Cases['Fatalities_Cross_Nations']

Length_of_Feature_Copy = len(Var_Bin2)

for c in range(0, Length_of_Feature_Copy):

    current_rows_arr_len1 = len(Var_Bin2[c])

    for a in range(0, current_rows_arr_len1):

        if a == current_rows_arr_len1-1:

             Var_Bin2[c] =  Var_Bin2[c][a]
Affected_Countries_Cases = Affected_Countries_Cases.reset_index()

Affected_Countries_Cases = Affected_Countries_Cases.sort_values(['Confirmed_Cases'], ascending=[False])

Affected_Countries_Cases = Affected_Countries_Cases.reset_index() 

Affected_Countries_Cases = Affected_Countries_Cases.drop(columns = 'index')                                      



x = Affected_Countries_Cases['Confirmed_Cases']

y = Affected_Countries_Cases['Fatalities_Cross_Nations']

plt.xlabel('Total Confirmed Cases', fontsize=12)

plt.ylabel('Total Calamity', fontsize=12)

plt.title("Covid 19 Widespread", fontsize=25)

plt.scatter(x, y, color='Brown', marker="8")
Top_VictimNations = Affected_Countries_Cases.loc[0:9]

# Top_VictimNations = Top_VictimNations.to_dict()

#Top_VictimNations = Top_VictimNations.set_index('Country').T.to_dict()

My_Dict1 = {'Italy':92472,

            'Spain':73235,

            'Germany':57695,

            'France':37575,

            'Iran':35408,

            'United Kingdom':17089,

            'Switzerland':14076,

            'Netherlands':9762,

            'South Korea':9478,

            'Belguim': 9134}

plt.bar(My_Dict1.keys(), My_Dict1.values(), color='Brown')

plt.xlabel("Countries", fontsize=12)

plt.ylabel("Total Number of Confirmed Cases",fontsize=12)

plt.title("Top 10 Infected Nations", fontsize = 20)

plt.xticks(rotation = 40)

plt.figure(figsize=(12,24))
My_Dict2 = {'Italy':10023,

            'Spain':5982,

            'Germany':433,

            'France':2314,

            'Iran':2517,

            'United Kingdom':1019,

            'Switzerland':264,

            'Netherlands':639,

            'South Korea':144,

            'Belguim': 353}

plt.bar(My_Dict1.keys(), My_Dict1.values(), color='Brown')

plt.xlabel("Countries", fontsize=12)

plt.ylabel("Total Number of Death Cases",fontsize=12)

plt.title("Casulity w.r.t. Countries", fontsize = 20)

plt.xticks(rotation = 40)

plt.figure(figsize=(12,24))
#Country Spefic EDA

#1 Italy

Italy = dataset[dataset.Country == 'Italy']

Italy = Italy.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]

Italy = Italy.reset_index()

Italy = Italy.drop(columns = 'index')                                      

sns.set(style='white',)

sns.lineplot(x = "Date", y = "Confirmed_Cases", data=Italy, color= 'Red')

sns.lineplot(x = "Date", y = "Noof_Deaths", data=Italy, color = 'Blue')

plt.title("Covid 19 Knockdown in Italy", fontsize =18)

plt.ylabel('Confirmed_Cases/Casuality')

plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])

plt.show() 
#2 Spain

Spain = dataset[dataset.Country == 'Spain']

Spain = Spain.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]

Spain = Spain.reset_index()

Spain = Spain.drop(columns = 'index')                                      

sns.set(style='white',)

sns.lineplot(x = "Date", y = "Confirmed_Cases", data=Spain, color= 'Red')

sns.lineplot(x = "Date", y = "Noof_Deaths", data=Spain, color = 'Blue')

plt.title("Covid 19 Knockdown in Spain", fontsize =18)

plt.ylabel('Confirmed_Cases/Casuality')

plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])

plt.show() 


#3 Germany

Germany = dataset[dataset.Country == 'Germany']

Germany = Germany.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]

Germany = Germany.reset_index()

Germany = Germany.drop(columns = 'index')                                      

sns.set(style='white',)

sns.lineplot(x = "Date", y = "Confirmed_Cases", data=Germany, color= 'Red')

sns.lineplot(x = "Date", y = "Noof_Deaths", data=Germany, color = 'Blue')

plt.title("Covid 19 Knockdown in Germany", fontsize =18)

plt.ylabel('Confirmed_Cases/Casuality')

plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])

plt.show() 

#4 Iran

Iran = dataset[dataset.Country == 'Iran']

Iran = Iran.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]

Iran = Iran.reset_index()

Iran = Iran.drop(columns = 'index')                                      

sns.set(style='white',)

sns.lineplot(x = "Date", y = "Confirmed_Cases", data=Iran, color= 'Red')

sns.lineplot(x = "Date", y = "Noof_Deaths", data=Iran, color = 'Blue')

plt.title("Covid 19 Knockdown in Iran", fontsize =18)

plt.ylabel('Confirmed_Cases/Casuality')

plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])

plt.show() 



#5 UK

UK = dataset[dataset.Country == 'United Kingdom']

UK = UK.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]

UK = UK.reset_index()

UK = UK.drop(columns = 'index')                                      

sns.set(style='white',)

sns.lineplot(x = "Date", y = "Confirmed_Cases", data=UK, color= 'Red')

sns.lineplot(x = "Date", y = "Noof_Deaths", data=UK, color = 'Blue')

plt.title("Covid 19 Knockdown in United Kingdom", fontsize =14)

plt.ylabel('Confirmed_Cases/Casuality')

plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])

plt.show() 
#6 South korea

South_Korea = dataset[dataset.Country == 'Korea, South']

South_Korea = South_Korea.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]

South_Korea = South_Korea.reset_index()

South_Korea = South_Korea.drop(columns = 'index')                                      

sns.set(style='white',)

sns.lineplot(x = "Date", y = "Confirmed_Cases", data=South_Korea, color= 'Red')

sns.lineplot(x = "Date", y = "Noof_Deaths", data=South_Korea, color = 'Blue')

plt.title("Covid 19 Knockdown in South Korea", fontsize =18)

plt.ylabel('Confirmed_Cases/Casuality')

plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])

plt.show() 



#7 China

China = dataset[dataset.Country == 'China']

China = China.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]

China = China.reset_index()

China = China.drop(columns = 'index')                                      

sns.set(style='white',)

sns.lineplot(x = "Date", y = "Confirmed_Cases", data=China, color= 'Red')

sns.lineplot(x = "Date", y = "Noof_Deaths", data=China, color = 'Blue')

plt.title("Covid 19 Knockdown in China", fontsize =18)

plt.ylabel('Confirmed_Cases/Casuality')

plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])

plt.show() 


