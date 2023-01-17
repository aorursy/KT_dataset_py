# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
%matplotlib inline
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
train.columns
train.tail(5)
train.isnull().sum()
# Given a dataframe, a country and a date, 
# Cumulating the quantity of confirmed cases and fatalities by date 
def tranform_data_series(base, dt, country):
    lt = []
    lt.append( np.sum(base['ConfirmedCases']) )     
    lt.append( sum(base['Fatalities']) )
    lt.append(dt)
    lt.append(country)
    return lt
# Getting all dates of database 
date = set(train['Date'])
#date
# Getting all countries from database
paises = train['Country/Region'].unique()

col = ['ConfirmedCases', 'Fatalities', 'Date', 'Country/Region']
base = pd.DataFrame(columns=col)

for pais in paises: # for each country
    pais_base = train.loc[train['Country/Region']==pais] # Base of country
    
    for dt in date: # for each day from couuntry - cumulating cases
        day_base = pais_base.loc[pais_base['Date']==dt]
        temp = tranform_data_series(day_base, dt, pais)
        tmp = pd.DataFrame([temp], columns=col)
        base = pd.concat( [base, tmp], ignore_index=True )
     
from datetime import datetime
from pytz import timezone

data_atual = datetime.now()
fuso_horario = timezone('America/Manaus')
data_hora_mao = data_atual.astimezone(fuso_horario)
data_mao_text = data_hora_mao.strftime('%Y-%m-%d' )

print(data_mao_text)
#total_fat = base[base['Date']==data_mao_text]
total_fat = base[base['Date']=='2020-03-24']
total_fat = total_fat.sort_values(['ConfirmedCases'], inplace=False, ascending=False)
base_2 = total_fat.iloc[0:30, :]
#base_2
sns.set(style="whitegrid")
plt.figure(figsize=(12,6))
plt.title('Country X Number of Confirmed Cases ', fontsize=15)
ax = sns.barplot(x="ConfirmedCases", y="Country/Region", data=base_2)

total_fat_ = total_fat.sort_values(['Fatalities'], inplace=False, ascending=False)
base_3 = total_fat_.iloc[0:20, :]
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
plt.title('Country X Fatalities ', fontsize=15)
ax = sns.barplot(x="Fatalities", y="Country/Region", data=base_3)

# Plots confirmed/fatalities cases from the country
def cases_country(df, pais, a, b, y):
    sns.set(style="whitegrid")
    plt.figure(figsize=(a,b))
    
    sns.barplot(x='Date', y=y, data=df)
    sns.pointplot(x='Date', y=y, data=df, color= sns.xkcd_rgb["denim blue"])
    
    plt.xticks(rotation=70) #rotating x labels
    plt.tight_layout()
    plt.title("Country:{} \nNumber of {} by date".format(pais,y), fontsize=14) # You can comment this line out if you don't need title
    plt.show()
# Seek on the dataset registers about a country with more/equals then 1 confirmed cases
def get_country(base, country):
    df = base.loc[ base['Country/Region']==country ]
    df = df.loc[df['ConfirmedCases']>=1.0] 
    df = df.sort_values(['Date'])
    return df

brazil = get_country(base, 'Brazil')
cases_country(brazil, 'Brazil', 20, 8, 'ConfirmedCases')

us = get_country(base, 'US')
cases_country(us, 'US', 20, 8, 'ConfirmedCases')

italy = get_country(base, 'Italy')
cases_country(italy, 'Italy', 20, 8, 'ConfirmedCases')

china = get_country(base, 'China')
cases_country(china, 'China', 20, 8, 'ConfirmedCases')
cases_country(brazil, 'Brazil', 20, 8, 'Fatalities')
cases_country(us, 'US', 20, 8, 'Fatalities')
cases_country(italy, 'Italy', 20, 8, 'Fatalities')
cases_country(china, 'China', 20, 8, 'Fatalities')
# Function to plot curve
def f(x,a,b):
    return a * np.exp(b*x)
x = np.arange(1, brazil.shape[0]+1, 1)
z = brazil['ConfirmedCases']

p, pcov = curve_fit(f,x,z)

result= f(x,*p)

print('p = '+str(p))
plt.xlabel('Dia')
plt.ylabel('Confirmed Cases Number')
plt.plot(x,z,'ro') #confirmed cases - real
plt.plot(x,result,'b-') # estimated
plt.show()
#f = interpolate.interp1d(x, result, fill_value = "extrapolate")
#print('Brazil: Predicted cases number: {:.2f} (23/03)'.format(f(29)))
len(x)
p, pcov = curve_fit(f,x,z)
result= f(x,*p)

print('p = '+str(p))
plt.xlabel('Dia')
plt.ylabel('Confirmed Cases Number')
plt.plot(x,z,'ro') #confirmed cases - real
plt.plot(x,result,'b-') # estimated
plt.show()
x = np.arange(1, us.shape[0]+1, 1)
z = us['ConfirmedCases']
p, pcov = curve_fit(f,x,z)
result= f(x,*p)

print('p = '+str(p))
plt.xlabel('Dia')
plt.ylabel('Confirmed Cases Number')
plt.plot(x,z,'ro') #confirmed cases - real
plt.plot(x,result,'b-') # estimated
plt.show()
#tam=len(x)
#f = interpolate.interp1d(x, result, fill_value = "extrapolate")
#print('EUA: Predicted cases number: {:.2f} (22/03)'.format(f(tam+1)))