

import numpy as np 

import pandas as pd 

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

#%matplotlib inline

import seaborn as sns

import datetime as dt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print('Setup Complete')
my_filepath = '../input/moroccocoronavirus/corona_morocco.csv'

my_data = pd.read_csv(my_filepath)
my_data['Date']=[dt.datetime.strptime(x,'%d/%m/%Y') for x in my_data['Date'] ]
# Print the first five rows of the data

my_data.head()
my_data = my_data.fillna(0) 
my_data = my_data.set_index('Date')



my_data.head()
actif_column= my_data['Confirmed'] - my_data['Recovered'] - my_data['Deaths']

total_test_perDay = my_data['Confirmed'] + my_data['Excluded']

my_data['Actif'] = actif_column

my_data['Total Test'] = total_test_perDay



plt.figure(figsize=(10,5))

sns.lineplot(data = my_data['Confirmed'],label='Confirmed')

sns.lineplot(data = my_data['Recovered'],label='Recovered')

plt.xlabel('Date')

plt.ylabel('Total number')

plt.title('Graph chart of confirmed and recovered cases in Morocco')



plt.legend()


my_data['2020-04-13':'2020-04-16']
my_data_ori = my_data.copy() # later we will need all Confirmed,Death and Recovered collumns for all date uncluding 14,15 April



my_data = my_data[my_data.index != '2020-04-14']

my_data = my_data[my_data.index != '2020-04-15']
plt.figure(figsize=(10,5))

sns.lineplot(data = my_data['Total Test'],label='Total Test')

plt.xlabel('Date')

plt.ylabel('Number of tests')

plt.figure(figsize=(10,5))

sns.regplot(x = my_data['Total Test'],y = my_data['Confirmed'])
# Selecting regions

Regions1 = ['Beni Mellal-Khenifra',

       'Casablanca-Settat', 'Draa-Tafilalet', 'Dakhla-Oued Ed-Dahab',

       'Fes-Meknes', 'Laayoune-Sakia El Hamra','Guelmim-Oued Noun', 'Laayoune-Sakia El Hamra',

       'Marrakesh-Safi', 'Oriental', 'Rabat-Sale-Kenitra', 'Souss-Massa','Tanger-Tetouan-Al Hoceima']





plt.figure(figsize=(14,8))

sns.lineplot(data = my_data[Regions1],dashes=False)

plt.text('2020-05-10', 2250, 'Casabalanca-Settat', fontsize=12,color='#ff4500')

plt.text('2020-05-12', 1350, 'Marrakesh-Safi', fontsize=12,color='#4682B4')

plt.text('2020-05-10', 1100, 'Tanger-Tetouan-Al Hoceima', fontsize=10,color='#FF69B4')

plt.title('Total cases evolution per region',fontsize = 20)
region_bar = my_data.iloc[[-1]].transpose()

region_bar = region_bar.drop(['Confirmed','Recovered','Deaths','Excluded','Actif','Total Test'])

region_bar.columns=['Total cases']

region_bar.index.name = 'Regions'

plt.figure(figsize=(20,6))

sns.barplot(x=region_bar.index,y='Total cases',data=region_bar)

plt.title('Total case per region',fontsize=20)
plt.figure(figsize=(14,8))

sns.lineplot(data = my_data['Actif'],label='Acif')

plt.axvline('2020-03-20',ls = '--',c = 'r')

plt.axvline('2020-04-20',ls = '--',c = 'r')

plt.axvline('2020-05-20',ls = '--',c = 'r')

plt.axvline('2020-06-10',ls = '--',c = 'g')

plt.text('2020-03-21', 1500, 'Lockdown', fontsize=12,color='#FF0000')

plt.text('2020-04-21', 1500, 'Lockdown extended', fontsize=12,color='#FF0000')

plt.text('2020-05-21', 1500, 'Lockdown extended', fontsize=12,color='#FF0000')

plt.text('2020-06-11', 1500, 'Lockdown End', fontsize=12,color='#00FA00')

plt.xlabel('Date')

plt.ylabel('Total actif cases')

plt.title('Actif cases ')

plt.legend()
# getting the data that we want to plot

plt.figure(figsize=(15,5))

dataVar1 =(my_data_ori['Actif']*100/my_data_ori['Confirmed'])[15:]

dataVar3 =(my_data_ori['Deaths']*100/my_data_ori['Confirmed'])[15:]

dataVar2 =(my_data_ori['Recovered']*100/my_data_ori['Confirmed'])[15:]





# plot each data

p1 = plt.bar(dataVar1.index, 

             dataVar1,label='Actif')



p2 = plt.bar(dataVar2.index, 

             dataVar2,

             bottom=dataVar1,color='g',label='Recovered')



p3 = plt.bar(dataVar3.index,

             dataVar3,

             bottom=dataVar1+dataVar2,color='r',label='Deaths')



plt.xlabel('Date')

plt.ylabel('Pourcentage %')

plt.title('Pourcentage of distribution of cases')

plt.legend()


plt.figure(figsize=(25,10))



BM = my_data['Beni Mellal-Khenifra']

CS = my_data['Casablanca-Settat']

DT = my_data['Draa-Tafilalet']

DO = my_data['Dakhla-Oued Ed-Dahab']

FM = my_data['Fes-Meknes']

GO = my_data['Guelmim-Oued Noun']

LS = my_data['Laayoune-Sakia El Hamra']

MS = my_data['Marrakesh-Safi']

Or = my_data['Oriental']

RS = my_data['Rabat-Sale-Kenitra']

SM = my_data['Souss-Massa']

TT = my_data['Tanger-Tetouan-Al Hoceima']



ax1 = plt.subplot(4, 3, 1)

sns.lineplot(data = BM,label='Beni Mellal-Khenifra')

ax2 = plt.subplot(4, 3, 2)

sns.lineplot(data = CS,label='Casablanca-settat')

ax3 = plt.subplot(4, 3, 3)

sns.lineplot(data = DT,label='Draa-Tafilalet')

ax4 = plt.subplot(4, 3, 4)

sns.lineplot(data = DO,label='Dakhla-Oued Ed Dahab')

ax5 = plt.subplot(4, 3, 5)

sns.lineplot(data = FM,label='Fes-Meknes')

ax6 = plt.subplot(4, 3, 6)

sns.lineplot(data = GO,label='Guelmim Oued Noun')

ax7 = plt.subplot(4, 3, 7)

sns.lineplot(data = LS,label='Laayoune Sakia El Hamra')

ax8 = plt.subplot(4, 3, 8)

sns.lineplot(data = MS,label='Marakesh Safi')

ax9 = plt.subplot(4, 3, 9)

sns.lineplot(data = Or,label='Rabat Sale Kenitra')

ax10 = plt.subplot(4, 3, 10)

sns.lineplot(data = RS,label='Souss Massa')

ax11 = plt.subplot(4, 3, 11)

sns.lineplot(data = SM,label='Acif')

ax12 = plt.subplot(4, 3, 12)

sns.lineplot(data = TT,label='Tanger Tetouane Al Hoceima')

plt.figure(figsize=(16,8))

Slice = [my_data['Beni Mellal-Khenifra'][-1],my_data['Casablanca-Settat'][-1],my_data['Draa-Tafilalet'][-1],

        my_data['Dakhla-Oued Ed-Dahab'][-1],my_data['Fes-Meknes'][-1],my_data['Guelmim-Oued Noun'][-1],

        my_data['Rabat-Sale-Kenitra'][-1],my_data['Souss-Massa'][-1],my_data['Tanger-Tetouan-Al Hoceima'][-1],

        my_data['Laayoune-Sakia El Hamra'][-1],my_data['Marrakesh-Safi'][-1],my_data['Oriental'][-1]]

Labels = ['Beni Mellal-Khenifra','Casablanca-Settat','Draa-Tafilalet','Dakhla-Oued Ed-Dahab','Fes-Meknes',

         'Guelmim-Oued Noun','Rabat-Sale-Kenitra','Souss-Massa','Tanger-Tetouan-Al Hoceima','Laayoune-Sakia El Hamra',

         'Marrakesh-Safi','Oriental']

Colors = ['#f78fA7','#EE204D','#FF7538','#1F75FE','#4CB7A5','#1CAC78','#FCE833', '#926EAE', '#828E84' ,

          '#000000', '#008080','#808080']

Explode = [0,0,0,0,0,0,0,0,0,0,0,0]

plt.pie(Slice,colors = Colors,explode = Explode,shadow = True,autopct='%1.1f%%',

        labels = Labels,wedgeprops = {'edgecolor':'white'})





plt.title('Pie Chart of Total cases per region')

my_data
from fbprophet import Prophet
total_Deaths = my_data['Deaths']

total_Confirmed = my_data['Confirmed']

total_Recovered = my_data['Recovered']

mortaloty_rate = 100*total_Deaths/total_Confirmed
# Adding a new daily cases column to our data

new_cases = []

for i in range(len(total_Confirmed)):

    if i == 0:

        new_cases.append(0)

    elif total_Confirmed[i] < total_Confirmed[i-1]:

        new_cases.append(0)

    else:

        temp = int(total_Confirmed[i] - total_Confirmed[i-1])

        new_cases.append(temp)

    

new_cases = np.array(new_cases)

my_data['New cases'] = new_cases
df = pd.DataFrame(my_data['New cases'])

df.style.background_gradient(cmap='Reds')
# Hands of plotly to visualise the current situation



import plotly.express as px

import plotly.graph_objects as go





fig = px.bar(my_data, x=my_data.index, y="New cases", color='New cases', orientation='v', height=600,

             title='Confirmed Cases in Morocco', color_discrete_sequence = px.colors.cyclical.IceFire)

fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=my_data.index, y = my_data['Confirmed'], mode= 'lines+markers',name='Total Cases'))

fig.add_trace(go.Scatter(x=my_data.index, y = my_data['Recovered'], mode='lines+markers',name='Recovered',line=dict(color='Green', width=2)))

fig.add_trace(go.Scatter(x=my_data.index, y=my_data['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))

fig.update_layout(title_text='Trend of Coronavirus Cases in Morocco (Cumulative cases)',plot_bgcolor='rgb(230, 230, 230)')



fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=my_data.index, y = my_data['Actif'], mode= 'lines+markers',name='Total Cases'))

fig.update_layout(title_text='Trend of Actif Coronavirus Cases in Morocco ',plot_bgcolor='rgb(230, 230, 230)')

fig.show()
confirmed = my_data.groupby('Date').sum()['Confirmed'].reset_index()



confirmed.columns = ['ds','y']
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=10)

future.tail(10)
#predicting the future with date, and upper and lower limit of y value

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.ds, y = abs(forecast['yhat'].round()), mode= 'lines+markers',name='Predicted daily cases'))

fig.add_trace(go.Scatter(x=forecast.ds, y = abs(forecast['yhat_lower'].round()), mode= 'markers',name='Predicted daily cases yhat_lower'))

fig.add_trace(go.Scatter(x=forecast.ds, y = abs(forecast['yhat_upper'].round()), mode= 'markers',name='Predicted daily cases yhat_upper'))



fig.add_trace(go.Scatter(x=my_data.index, y = my_data['Confirmed'], mode= 'lines+markers',name='Daily cases'))
new_cases = my_data.groupby('Date').sum()['New cases'].reset_index()

new_cases.columns = ['ds','y']
m_1 = Prophet(interval_width=0.95)

m_1.fit(new_cases)

future_1 = m_1.make_future_dataframe(periods=10)

future_1.tail()
forecast_1 = m_1.predict(future_1)

forecast_1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast_1.ds, y = abs(forecast_1['yhat'].round()), mode= 'lines+markers',name='Predicted daily cases'))



fig.add_trace(go.Scatter(x=my_data.index, y = my_data['New cases'], mode= 'lines+markers',name='Daily cases'))
deaths = my_data.groupby('Date').sum()['Deaths'].reset_index()

deaths.columns = ['ds','y']
m_2 = Prophet(interval_width=0.95)

m_2.fit(deaths)

future_2 = m_2.make_future_dataframe(periods=10)

future_2.tail()
forecast_2 = m_2.predict(future_2)

forecast_2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast_2.ds, y = abs(forecast_2['yhat'].round()), mode= 'lines+markers',name='Predicted total deaths'))



fig.add_trace(go.Scatter(x=my_data.index, y = my_data['Deaths'], mode= 'lines+markers',name='Daily cases'))