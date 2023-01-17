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
!ls /kaggle/input
df_hospitalbeds = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')
#df_covid_india = pd.read_csv('kaggle/input/covid19-in-india/covid_19_india.csv')
df_pop_2011 = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')
df_icmr_labs=pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingLabs.csv')
df_indv_details = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')
df_age_group = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
df_satewise_test = pd.read_csv('/kaggle/input/covid-india-updated/statewise_tested_numbers_data.csv')
df_statewise_daily = pd.read_csv('/kaggle/input/covid-india-updated/state_wise_daily.csv')
df_statewise = pd.read_csv('/kaggle/input/covid-india-updated/state_wise.csv')
df_rawdata = pd.read_csv('/kaggle/input/covid-india-updated/raw_data5.csv')
df_tested_numbers= pd.read_csv('/kaggle/input/covid-india-updated/tested_numbers_icmr_data.csv')
# df_death = pd.read_csv('/kaggle/input/covid-india-updated/death_and_recovered2.csv')
# df_districtwise= pd.read_csv('/kaggle/input/covid-india-updated/district_wise.csv')
df_satewise_test.head()
df_satewise_test.isnull().sum()
df_hospitalbeds.head()
df_hospitalbeds.isnull().sum()
df_hospitalbeds.shape
df_statewise_daily.head()
df_statewise_daily.isnull().sum()
df_statewise_daily.shape
print(df_statewise_daily)
df_statewise.head()
df_statewise.shape
df_statewise.isnull().sum()
print(df_statewise)
df_statewise.dtypes
df_age_group.head()
print(df_age_group)
print(df_rawdata)
df_rawdata.dtypes
df_rawdata.isnull().sum()
df_rawdata.shape
df = pd.DataFrame(df_rawdata)
selected_columns= df[["Entry_ID","Date Announced", "Age Bracket","Gender",  "Detected District","Detected State", "State code",
"Num Cases", "Current Status", "Patient Number"]]

df_patients = selected_columns.copy()

df_patients.head()
df_patients.isnull().sum()
df_patients.shape
# do not use plot, use plt instead, plot is an inherent function
import matplotlib.pyplot as plt
df = df_statewise



df.plot.bar("State", "Confirmed", rot=90, title= "State Wise Cases Comparison", figsize=(40, 10), fontsize=(20), color = 'b'  )
df.plot.bar("State", "Recovered", rot= 90, title = "State Wise Cases Comparison", figsize = (40, 10), fontsize= (20), color = 'r'  )
df.plot.bar("State", "Deaths", rot= 90, title = "State Wise Cases Comparison", figsize = (40, 10), fontsize= (20), color = 'g'  )


plt.show()
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 3), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot(1,1,1)
w=0.3
x=np.arange(len(df.loc[~df['State'].isin(['Total'])]['State'].unique()))
plt.xticks(ticks=x, labels=list(df.loc[~df['State'].isin(['Total'])]['State'].values), rotation='vertical')
# ax.bar(x-w, df.loc[~df['State'].isin(['Total'])]['Confirmed'], width=w, color='b', align='center')
ax.bar(x, df.loc[~df['State'].isin(['Total'])]['Recovered'], width=w, color='g', align='center')
plt.ylabel('# of Cases')
ax1 = ax.twinx()
ax1.bar(x+w, df.loc[~df['State'].isin(['Total'])]['Deaths'], width=w, color='r', align='center')
plt.ylabel('# of Cases')
plt.show()
print(df_statewise_daily)
df_statewise_daily.dtypes
df_statewise_daily.head()
df_statewise_daily['Status'].value_counts()
df = df_statewise_daily.groupby('Status')
df_confirmed= df.get_group('Confirmed')
df_recovered= df.get_group('Recovered')
df_deceased= df.get_group('Deceased')
df_confirmed.head()
df_confirmed.shape
df_confirmed.dtypes
df1 = df_confirmed

df1.head()
column_list = list(df1)
print(column_list)
column_list.remove("Date")
print(column_list)
df1.loc[:,"Total"]= df_confirmed[column_list].sum(axis=1)
df1.head()
print(df1)
df1.plot.bar("Date", "Total", rot=90, title= "Date Wise Cases", figsize=(40, 10), fontsize=(20), color = 'b'  )

df2 = df_recovered
col_list = list(df2)
print(col_list)
col_list.remove("Date")
print(col_list)
df2["Total"] = df2[col_list].sum(axis=1)
df2.head()
df2.plot.bar("Date", "Total", rot=90, title="Date Wise Recovered", fontsize= (20), figsize=(40, 10), color = 'g')
df3=df_deceased
col = list(df_deceased)

col.remove("Date")
print(col)
df3["Total"]=df3[col].sum(axis=1)
df3.head()
df3.plot.bar("Date", "Total", figsize=(40,10), fontsize= (20), title = "Date Wise Deceased", color= 'r')

print(df_hospitalbeds)
df1.head()
data = df1['Total']
data = data.reset_index(drop=False)
data.columns = ['Timestep', 'Total']
print(data)
print(data)
df_t= pd.read_csv('/kaggle/input/covidindialatest/COVID19India.csv')
df_t= df_t.reset_index(drop=True)
df_new = df_t.T
df_new
#df_t.head()
new_header= df_new.iloc[0]
df_new= df_new[1:]
df_new.columns= new_header
df_new.head()
columns = list(df_new)

print(columns)

df_new.loc[:,"Active"]= df_new["Confirmed"]-(df_new["Recovered"]+df_new["Deceased"])
df_new["New Cases"]= ""
print(df_new)
#Initialize iterator i
#i = 0
#for i in range(len(df_new["Confirmed"])):
#    if i == 0:
#        df_new["Confirmed"].iloc[i]=0
#    df_new["New Cases"].iloc[i] = df_new["Confirmed"].iloc[i+1] - df_new["Confirmed"].iloc[i]
        
#print(df_new)
        
df_new['New Cases'] = df_new['Confirmed'].shift(-1) - df_new['Confirmed']
print(df_new)
df_new['Daily Recovered'] = df_new['Recovered'].shift(-1)- df_new['Recovered']
df_new['Daily Deaths']= df_new['Deceased'].shift(-1) - df_new['Deceased']
print(df_new)
df_inf = df_new['Confirmed']
df_inf = df_inf.reset_index(drop=False)
df_inf.columns = ['Timestep', 'Confirmed']
df_inf.head()
df_i = df_inf['Confirmed']
df_i = df_i.reset_index(drop=False)
df_i.columns = ['Timestep', 'Confirmed']
df_i.head()
#Define logistic function
import numpy as np
def my_logistic(t, a, b, c): #a, b, c are constants 
    return c / (1 + a*np.exp(-b*t))
#Randmize a, b , c
p0 = np.random.exponential(size=3)
p0
#setting upper and lower bounds for a, b, c 
bounds = (0, [100000., 3., 1000000000.])
import scipy.optimize as optim
x = np.array(df_i['Timestep'])+1
y = np.array(df_i['Confirmed'])
(a, b, c), cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)
def my_logistic(t):
    return c/(1 + a*np.exp(-b*t))
plt.scatter(x,y)
plt.plot(x, my_logistic(x))
plt.title('Logistic Model vs Real data')
plt.legend(['Logistic Model', 'Real data'])
plt.xlabel('Time')
plt.ylabel('Infections')
print(a, b, c)
temp = df_new['New Cases']
temp = temp.reset_index(drop=False)
temp.columns = ['Timestep', 'New Cases']

df_spread= temp['New Cases']
df_spread = df_spread.reset_index(drop=False)
df_spread.columns = ['Timestep', 'New Cases']

print(df_spread)
df_spread.drop(df_spread.tail(1).index, inplace = True )
df_spread.head()
print(df_spread)
import scipy.optimize as optim
x = np.array(df_spread['Timestep'])+1
y = np.array(df_spread['New Cases'])
plt.scatter(x,y)
plt.title('Daily Increase in Infections')
plt.legend(['New Cases'])
plt.xlabel('Time')
plt.ylabel('New Infections')
#Define logistic function
import numpy as np
def my_logistic(t, a, b, c): #a, b, c are constants 
    return c / (1 + a*np.exp(-b*t))
#Randmize a, b , c
p0 = np.random.exponential(size=3)
p0

#setting upper and lower bounds for a, b, c 
bounds = (0, [100000., 3., 1000000000.])
import scipy.optimize as optim
x = np.array(df_spread['Timestep'])+1
y = np.array(df_spread['New Cases'])
(a, b, c), cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)
def my_logistic(t):
    return c/(1 + a*np.exp(-b*t))
plt.scatter(x,y)
plt.plot(x, my_logistic(x))
plt.title('Logistic Model vs Spread of Infection')
plt.legend(['Logistic Model', 'Spread of Infection'])
plt.xlabel('Time')
plt.ylabel('New Cases')
print(a, b, c)
temp = df_new['Recovered']
temp = temp.reset_index(drop=False)
temp.columns = ['Time Steps','Recovered']

df_rec = temp['Recovered']
df_rec = df_rec.reset_index(drop=False)
df_rec.columns = ['Time Steps', 'Recovered']
print (df_rec)
#Define logistic function
import numpy as np
def my_logistic(t, a, b, c): #a, b, c are constants 
    return c / (1 + a*np.exp(-b*t))
#Randmize a, b , c
p0 = np.random.exponential(size=3)
p0

#setting upper and lower bounds for a, b, c 
bounds = (0, [100000., 3., 1000000000.])
import scipy.optimize as optim
x = np.array(df_rec['Time Steps'])+1
y = np.array(df_rec['Recovered'])
(a, b, c), cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)
def my_logistic(t):
    return c/(1 + a*np.exp(-b*t))
plt.scatter(x,y)
plt.plot(x, my_logistic(x))
plt.title('Logistic Model vs Recovered')
plt.legend(['Logistic Model', 'Recovered'])
plt.xlabel('Time')
plt.ylabel('Recovery')
print (a, b , c)
df_new.head()
df_hospitalbeds.head()
