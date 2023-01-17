# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter

from sklearn.preprocessing import MinMaxScaler

import plotly.express as px

import seaborn as sns

# geographical ploting

import folium

import geopandas as gpd

from keras.layers import LSTM,Dense

from keras.models import Sequential

import math

from sklearn.metrics import mean_squared_error

data = px.data.gapminder()

import plotly.graph_objects as go

from plotly.subplots import make_subplots

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path1 = '/kaggle/input/covid19-corona-virus-india-dataset/'



patient = pd.read_csv(path1+'patients_data.csv')

district = pd.read_csv(path1+'district_level_latest.csv')

test_daily = pd.read_csv(path1+'tests_daily.csv')

state = pd.read_csv(path1+'state_level_latest.csv')

test_state = pd.read_csv(path1+'tests_latest_state_level.csv')

nation_daily = pd.read_csv(path1+'nation_level_daily.csv')

complete = pd.read_csv(path1+'complete.csv')

zones = pd.read_csv(path1+'zones.csv')
gender = patient.gender.value_counts().reset_index()

gender.columns=['sex','count']

m = gender[gender.sex=='M']['count'].values[0]

f = gender[gender.sex=='F']['count'].values[0]

nb = gender[gender.sex=='Non-Binary']['count'].values[0]

x = [m,f,nb]

labels = 'Male','Female','Non-Binary'

explode=[0.1,0.1,0.1]

colors=['gold','yellowgreen','aqua']

plt.figure(figsize=(5,5))

plt.title("Infected Human Ratio in india",fontsize=20)

plt.pie(x,labels=labels,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True,startangle=90)

plt.show()
patient['type_of_transmission'] = patient['type_of_transmission'].replace('Imported ', 'Imported')

patient['type_of_transmission'] = patient['type_of_transmission'].replace('Unknown', 'TBD')

transmission = patient.type_of_transmission.value_counts().reset_index()

transmission.columns=['tran_type','counts']

transmission = transmission.sort_values('counts')

plt.figure(figsize=(8,3))

plt.barh(transmission.tran_type,transmission['counts'],label='Transmission Type',color='darkblue')

plt.xlabel('Counts')

plt.ylabel('Transmission Type')

plt.grid(alpha=0.3)

plt.show()
patient_dist = patient[patient.nationality !='India'].nationality.value_counts().reset_index()

patient_dist.columns=['Country','counts']

patient_dist = patient_dist.sort_values('counts')

plt.figure(figsize=(10,10))

fig = px.bar(patient_dist,x='counts',y='Country',orientation='h',text='counts', width=900, height=500,

       color_discrete_sequence = ['#35495e'],title='Foreign Patients Distribution')

fig.update_xaxes(title='')

fig.update_yaxes(title='')

fig.show()
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

fig.suptitle('District wise cases', fontsize=16)

axes = axes.flatten()



cols = ['confirmed', 'active', 'deceased', 'recovered']



for ind, col in enumerate(cols):

    sns.barplot(data=district.sort_values(col, ascending=False).head(15), 

                x=col, y='district', hue='state name', dodge=False,

                palette='Paired', ax=axes[ind])

    axes[ind].set_title(col)

    axes[ind].set_xlabel('')

    axes[ind].set_ylabel('')



plt.show()
dist_shape = gpd.read_file('../input/india-district-wise-shape-files/output.shp')

# subset columns

dist_shape = dist_shape[['objectid', 'statecode', 'statename', 'state_ut', 

                         'distcode', 'distname', 'geometry']]

# rename states

dist_shape['statename'] = dist_shape['statename'].str.replace('&', 'and')

dist_shape['statename'] = dist_shape['statename'].str.replace('NCT of ', '')

dist_shape['statename'] = dist_shape['statename'].str.replace('Chhatisgarh', 'Chhattisgarh')

dist_shape['statename'] = dist_shape['statename'].str.replace('Orissa', 'Odisha')

dist_shape['statename'] = dist_shape['statename'].str.replace('Pondicherry', 'Puducherry')

# State level shape file

# ======================



# groupby state to get state level shape file

states_shape = dist_shape.dissolve(by='statename').reset_index()

states_shape.head()
# grouped district file

# =====================



dist_count = patient.groupby(['detected_district'])['patient_number'].count().reset_index()

dist_count.columns = ['district', 'count']

# dist_count.head()





# grouped state file

# ==================



state_count = patient.groupby(['detected_state'])['patient_number'].count().reset_index()

state_count.columns = ['state', 'count']

# state_count.head()





# districts and zones

# ===================



zone_dist = zones[['district', 'zone']]

# zone_dist.head()
# merge shape file with count file

dist_map = pd.merge(dist_shape, dist_count, left_on='distname', right_on='district', how='left')

# droping missing values

dist_map = dist_map[~dist_map['count'].isna()]

# fixing datatype

dist_map['count'] = dist_map['count'].astype('int')

#dist_map.head(3)



# merge shape file with count file

state_map = pd.merge(states_shape, state_count, left_on='statename', right_on='state', how='right')

# fill na with 0

state_map['count'] = state_map['count'].fillna(0)

# state_map.head(3)





# Zones map

# ==========



# merge shape file with zone file

zones_map = pd.merge(dist_shape, zone_dist, left_on='distname', right_on='district', how='left')

zones_map = zones_map[~zones_map['zone'].isna()]

zones_map.head(3)
state_map.head(3)
dist_map.head()
leg_kwds={'title':'No. of case',

          'loc': 'upper right',

          'ncol':1}

fig, ax = plt.subplots(figsize=(10, 10))

states_shape.plot(ax=ax, color='whitesmoke')

state_map.plot(column='count',legend=True,cmap='Reds',k=5, ax=ax,scheme='Quantiles',legend_kwds=leg_kwds)

plt.title('State level cases')

ax.set_axis_off()

plt.show()
fig, ax = plt.subplots(figsize=(10, 10))

dist_shape.plot(ax=ax, color='whitesmoke')

dist_map.plot(column='count',legend=True,cmap='Reds',k=5, ax=ax,scheme='Quantiles',legend_kwds=leg_kwds)

plt.title('District level cases')

ax.set_axis_off()

plt.show()
vals_zone = zones.zone.value_counts().reset_index()

vals_zone.columns=['zone','count']

vals_zone = vals_zone.sort_values('count')

plt.figure(figsize=(8,3))

plt.barh(vals_zone.zone,vals_zone['count'],label='Zone Type',color=vals_zone.zone)

plt.xlabel('Counts')

plt.ylabel('Zone Type')

plt.grid(alpha=0.3)

plt.show()
zones_count = zones_map.groupby(['statename','zone'])['district'].count().reset_index()

fig = px.treemap(zones_count,path=["statename",'zone'], values='district', 

                 height=700, title='Number of Zones in the Respective State')

fig.show()
fig = px.treemap(zones_count,path=["zone","statename"], values='district', 

                 height=700, title='Number of Zones in the Respective State', 

                 color_discrete_sequence = ['orange','green','red'])

fig.show()
fig, ax = plt.subplots(figsize=(10, 10))

dist_shape.plot(ax=ax, color='whitesmoke')

zones_map.plot(color=zones_map['zone'], ax=ax, legend=True, legend_kwds=leg_kwds)

plt.title('Zones')

ax.set_axis_off()

plt.show()
temp = district.groupby(['state name','district']).sum().reset_index()

fig = px.treemap(temp, path=["state name", "district"], values='confirmed', 

                 height=700, title='Number of Confirmed Cases', 

                 color_discrete_sequence = px.colors.qualitative.Vivid)

fig.data[0].textinfo = 'label+text+value'

fig.show()
px.treemap(temp,path=['state name','district'],values='deceased',height=700,

           title='Deceased Cases in india',color_discrete_sequence = px.colors.qualitative.Plotly)
fig = px.treemap(temp, path=["state name", "district"], values='recovered', 

                 height=700, title='Number of Recovered Cases', 

                 color_discrete_sequence = px.colors.qualitative.G10)

fig.data[0].textinfo = 'label+text+value'

fig.show()
path = '/kaggle/input/covid19-in-india/'



nation = pd.read_csv(path+'StatewiseTestingDetails.csv')

india = pd.read_csv(path+'covid_19_india.csv')

hospital = pd.read_csv(path+'HospitalBedsIndia.csv')

indiv = pd.read_csv(path+'IndividualDetails.csv')

icmr = pd.read_csv(path+'ICMRTestingLabs.csv')

popp = pd.read_csv(path+'population_india_census2011.csv')
nation['Date'] = pd.to_datetime(nation.Date)
nation.isnull().sum()
nation['Negative'].fillna(nation['TotalSamples']- nation['Positive'], inplace=True)

nation['Positive'].fillna(nation['TotalSamples']- nation['Negative'], inplace=True)

nation['Negative'].fillna(0, inplace=True)

nation['Positive'].fillna(0, inplace=True)
nation.isnull().sum()
tests_daily = nation.groupby('Date')['TotalSamples','Negative','Positive'].sum().reset_index()

tests_daily
fig = px.line(tests_daily,x='Date',y='TotalSamples',title='Daily Testing graph')

fig.show()
melts = tests_daily.melt(id_vars='Date',value_vars=['TotalSamples','Negative','Positive'],var_name='Samples',value_name='Count')

#print(melts)

fig = px.bar(melts,x='Date',y='Count',color='Samples',height=600,title='Tests per day')

fig.show()
nation[nation['Date']== max(nation['Date'])].sort_values('TotalSamples',ascending=False).style.background_gradient(cmap='YlOrRd')
fig = px.bar(nation.sort_values('TotalSamples',ascending=False),x='Date',y='TotalSamples',color='State',

            color_discrete_sequence=px.colors.sequential.Inferno,title='Daily Tests State/wise')

fig.show()
nation_grouped=nation.groupby(['State']).sum()

nation_grouped = nation_grouped.sort_values('TotalSamples',ascending=True)

plt.figure(figsize=(16,10))

plt.barh(nation_grouped.index,nation_grouped['TotalSamples'],label="Total Samples",color='gold')

plt.barh(nation_grouped.index, nation_grouped['Positive'],label="Positive Cases",color='red')

plt.barh(nation_grouped.index, nation_grouped['Negative'],label="Negative Cases",color='green')

plt.xlabel('Tests',size=30)

plt.ylabel("States",size=30)

plt.legend(frameon=True, fontsize=12)

plt.title('Total Number of Test Statewise',fontsize = 20)

plt.show()
hospital_grouped = hospital.groupby('State/UT').sum()

hospital_grouped['total_beds'] = hospital_grouped['NumPublicBeds_HMIS']+hospital_grouped['NumRuralBeds_NHP18']+hospital_grouped['NumUrbanBeds_NHP18']



hospital_grouped_inda = hospital_grouped[hospital_grouped.index=='All India'].T

hospital_grouped_inda = hospital_grouped_inda[hospital_grouped_inda.index !='Sno']

hospital_grouped_inda.style.background_gradient(cmap='Blues')
hospital_grouped_state = hospital_grouped[hospital_grouped.index!='All India']

plt.figure(figsize=(12,12))

plt.barh(hospital_grouped_state.sort_values('total_beds').index,hospital_grouped_state.sort_values('total_beds').total_beds,label='total Beds',color='orange')

plt.title('Total Beds per State/UT')

plt.xlabel('Total Beds')

plt.ylabel('States/UT')

plt.grid(alpha=0.3)

plt.show()
temp = icmr.type.value_counts().reset_index()

plt.figure(figsize=(10,5))

plt.bar(temp['index'],temp['type'],label='Total No. of ICMR Labs',color='lightgreen')

plt.title('Total Labs as per ICMR')

plt.grid(alpha=0.3)

plt.show()
popp.rename(columns={'State / Union Territory':'State/UT'},inplace=True)

temp = pd.merge(popp,hospital_grouped_state[['total_beds']],on='State/UT')

plt.figure(figsize=(16,10))

plt.bar(temp['State/UT'],temp['Population'],label="Population",color='gray')

plt.bar(temp['State/UT'], temp['total_beds'],label="Total Beds",color='coral')

plt.xlabel('States',size=30)

plt.xticks(rotation=90)

plt.ylabel("Count (log)",size=30)

plt.yscale("log")

plt.gca().yaxis.set_major_formatter(ScalarFormatter())

plt.legend(frameon=True, fontsize=12)

plt.title('Population Vs Total Beds',fontsize = 20)

plt.show()
cnf = '#fac04d'

dth = '#fc0356'

crd = '#03fc9d'

nc = '#4dc0fa'

nr = '#4dfad2'

nd = '#fa4dac'
df = complete.copy()

# replace 'union territory' with ''

df['Name of State / UT'] = df['Name of State / UT'].str.replace('Union Territory of ', '')

df.columns = ['Date','State/UT','Latitude','Longitude','Confirmed','Deaths','Cured']

for i in ['Confirmed','Deaths','Cured']:

    df[i] = df[i].astype(int)

df['active'] = df['Confirmed']-df['Deaths']-df['Cured']

df['mortality_rate'] = np.round(df['Deaths']/df['Confirmed']*100,2)

df['recovery_rate'] = np.round(df['Cured']/df['Confirmed']*100,2)
df.head(2)
#New Cases

nc = df.groupby(['State/UT','Date'])['Confirmed','Deaths','Cured'].sum().diff().reset_index()

mask = nc['State/UT'] != nc['State/UT'].shift(1)

nc.loc[mask,'Confirmed'] = np.nan

nc.loc[mask,'Deaths'] = np.nan

nc.loc[mask,'Cured'] = np.nan

nc = nc[['Date','State/UT','Confirmed','Deaths','Cured']]

nc.columns = ['Date','State/UT','New Confirmed','New Deaths','New Cured']
df = pd.merge(df,nc,on=['State/UT','Date'])

df = df.fillna(0)

cols = ['New Confirmed','New Deaths','New Cured']

df[cols] = df[cols].astype('int')

df['New Confirmed'] = df['New Confirmed'].apply(lambda x: 0 if x<0 else x)

df.head(10)
state_wise = df[df['Date'] == max(df['Date'])]

state_wise = state_wise.groupby('State/UT')['Confirmed','Deaths','Cured','active','mortality_rate','recovery_rate','New Confirmed','New Deaths','New Cured'].sum().reset_index()

#per 100 cases

state_wise['Deaths/100 cases'] = np.round((state_wise['Deaths']/state_wise['Confirmed'])*100,2)

state_wise['Cured/100 cases'] = np.round((state_wise['Cured']/state_wise['Confirmed'])*100,2)

state_wise['Death/100 Cured'] = np.round((state_wise['Deaths']/state_wise['Cured'])*100,2)

state_wise['New/100 Confirmed'] = np.round((state_wise['New Confirmed']/state_wise['Confirmed'])*100,2)

cols=['Deaths/100 cases','Cured/100 cases','Death/100 Cured','New/100 Confirmed']

state_wise[cols] = state_wise[cols].fillna(0)

state_wise.sort_values('Confirmed').tail(3)
live_data = state_wise.sort_values('Confirmed',ascending=False).reset_index(drop=True)

live_data.style.background_gradient(cmap="Blues", subset=['Confirmed', 'active', 'New Confirmed','New/100 Confirmed']).background_gradient(cmap="Greens",                                                                                                                      subset=['Cured','recovery_rate','New Cured','Cured/100 cases']).background_gradient(cmap='Reds',subset=['Deaths','mortality_rate','New Deaths','Deaths/100 cases']).background_gradient(cmap='twilight',subset=['Death/100 Cured'])
fig = px.bar(df.sort_values('Confirmed', ascending=False), x="Date", 

             y="Confirmed", color='State/UT', title='State wise confirmed',

             color_discrete_sequence = px.colors.qualitative.Vivid)

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.bar(df.sort_values('New Confirmed',ascending=False),x='Date',y='New Confirmed',color='State/UT',

            color_discrete_sequence=px.colors.qualitative.Dark2,title='New Cases state/wise')

fig.show()
fig = px.bar(df.sort_values('Deaths',ascending=False),x='Date',y='Deaths',color='State/UT',

            color_discrete_sequence=px.colors.sequential.Burg,title='Death Cases state/wise')

fig.show()
fig = px.bar(df.sort_values('New Deaths',ascending=False),x='Date',y='New Deaths',color='State/UT',

            color_discrete_sequence=px.colors.sequential.Reds,title='Death Cases state/wise')

fig.show()
fig = px.bar(df.sort_values('Cured',ascending=False),x='Date',y='Cured',color='State/UT',

            color_discrete_sequence=px.colors.sequential.Greens,title='Recovered Cases state/wise')

fig.show()
fig = px.bar(df.sort_values('New Cured',ascending=False),x='Date',y='New Cured',color='State/UT',

            color_discrete_sequence=px.colors.sequential.Teal,title='New Recovered Cases state/wise')

fig.show()
fig_c = px.bar(live_data.sort_values('Confirmed').tail(10),x='Confirmed',y='State/UT',orientation='h',text='Confirmed',color_discrete_sequence=px.colors.sequential.Jet)

fig_a = px.bar(live_data.sort_values('active').tail(10),x='active',y='State/UT',orientation='h',text='active',color_discrete_sequence=px.colors.sequential.Jet)

fig_d = px.bar(live_data.sort_values('Deaths').tail(10),x='Deaths',y='State/UT',orientation='h',text='Deaths',color_discrete_sequence=[dth])

fig_m = px.bar(live_data[live_data['Confirmed']>100].sort_values('mortality_rate').tail(10),x='mortality_rate',y='State/UT',orientation='h',text='mortality_rate',color_discrete_sequence=[dth])

fig_cu = px.bar(live_data.sort_values('Cured').tail(10),x='Cured',y='State/UT',orientation='h',text='Cured',color_discrete_sequence=[crd])

fig_r = px.bar(live_data[live_data['Confirmed']>100].sort_values('recovery_rate').tail(10),x='recovery_rate',y='State/UT',orientation='h',text='recovery_rate',color_discrete_sequence=[crd])



fig = make_subplots(rows=3, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('Confirmed cases', 'Active cases', 

                                    'Deaths reported', 'Mortality rate (< 100 cases)', 

                                    'Cured', 'Recovery rate (< 100 cases)'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_a['data'][0], row=1, col=2)

fig.add_trace(fig_d['data'][0], row=2, col=1)

fig.add_trace(fig_m['data'][0], row=2, col=2)

fig.add_trace(fig_cu['data'][0], row=3, col=1)

fig.add_trace(fig_r['data'][0], row=3, col=2)



fig.update_layout(height=1200, title_text="Leading State/UT")
worst_list = list(live_data.sort_values('Confirmed',ascending=False).head(10)['State/UT'])

worst_list
fig = go.Figure()

for state in worst_list:

   fig.add_trace(go.Scatter(x=df[df['State/UT']==state]["Confirmed"], y=df[df['State/UT']==state]["active"],

                    mode='lines',name=state))

fig.update_layout(height=600,title="COVID-19 Journey of some worst affected States in India",

                 xaxis_title="Confirmed Cases",yaxis_title="Active Cases")

fig.show()
fig = go.Figure()

for state in worst_list:

   fig.add_trace(go.Scatter(x=df[df['State/UT']==state]["Cured"], y=df[df['State/UT']==state]["Deaths"],

                    mode='lines',name=state))

fig.update_layout(height=600,title="COVID-19 Journey of some worst affected States in India",

                 xaxis_title="Recovered Cases",yaxis_title="Death Cases")

fig.show()
df_plot = live_data

fig = px.pie(df_plot, values='New Confirmed', names='State/UT',color_discrete_sequence=px.colors.sequential.RdBu,

            title='Distribution of New Cases')

fig.show()
#Daily

temps = df.groupby('Date')['Confirmed','active','Deaths','Cured','New Confirmed','New Deaths','New Cured'].sum().reset_index()
india_till_date = temps[temps['Date'] == max(temps['Date'])]

india_till_date.set_index('Date').style.background_gradient(cmap='Blues',subset=['Confirmed','New Confirmed']).background_gradient(cmap='Reds',subset=['Deaths','New Deaths']).background_gradient(cmap='Greens',subset=['Cured','New Cured'])
s = india_till_date.melt(id_vars='Date',value_vars=['Confirmed','Deaths','Cured'],var_name='Case',value_name='Count')

fig_1 = px.treemap(s, path=["Case"], values="Count", height=250, width=800,

                 color_discrete_sequence=[cnf,crd,dth], title='Latest stats')

fig_1.data[0].textinfo = 'label+text+value'

fig_1.show()
s = india_till_date.melt(id_vars='Date',value_vars=['active','Deaths','Cured'],var_name='Case',value_name='Count')

fig_1 = px.treemap(s, path=["Case"], values="Count", height=250, width=800,

                 color_discrete_sequence=[crd,cnf,dth], title='Latest stats')

fig_1.data[0].textinfo = 'label+text+value'

fig_1.show()
s = india_till_date.melt(id_vars='Date',value_vars=['New Confirmed','New Deaths','New Cured'],var_name='New Case',value_name='Count')

fig_1 = px.treemap(s, path=["New Case"], values="Count", height=250, width=800,

                 color_discrete_sequence=[cnf,crd,dth], title='Latest stats')

fig_1.data[0].textinfo = 'label+text+value'

fig_1.show()
melts = temps.melt(id_vars='Date',value_vars=['Cured','Deaths','active'],var_name='Case',value_name='Count')

fig = px.bar(melts,x='Date',y='Count',color='Case',height=600,title='Cases per day')

fig.show()
fig = px.line(temps,x='Date',y='New Confirmed',title='New Cases per day')

fig.show()
fig = px.line(temps,x='Date',y='New Deaths',title='New Deaths per day',color_discrete_sequence=[dth])

fig.show()
melts3 = temps.melt(id_vars='Date',value_vars=['Confirmed','active'],var_name='Cases',value_name='Count')

fig = px.line(melts3,x='Date',y='Count',color='Cases',title='Confirmed Vs Active Cases',line_dash='Cases')

fig.show()
melts4 = temps.melt(id_vars='Date',value_vars=['Cured','Deaths'],var_name='Cases',value_name='Count')

fig = px.line(melts4,x='Date',y='Count',color='Cases',title='Recovered Vs Death Cases',line_dash='Cases',color_discrete_sequence=[crd,dth])

fig.show()
melts1 = temps.melt(id_vars='Date',value_vars=['New Confirmed','New Cured'],var_name='Cases',value_name='Count')

fig = px.line(melts1,x='Date',y='Count',color='Cases',title='New Cases Vs New Recovered',line_dash='Cases',color_discrete_sequence=[cnf,crd])

fig.show()
melts2 = temps.melt(id_vars='Date',value_vars=['New Confirmed','New Deaths'],var_name='Cases',value_name='Count')

fig = px.line(melts2,x='Date',y='Count',color='Cases',title='New Cases Vs New Deaths',line_dash='Cases',color_discrete_sequence=[cnf,dth])

fig.show()
plt.style.use('seaborn')

g = sns.FacetGrid(df.sort_values(['State/UT', 'Date']), 

                  col="State/UT", hue="State/UT", 

                  sharey=False, col_wrap=7)

g = g.map(plt.plot, "Date", "New Confirmed")

g.set_xticklabels(rotation=90)

g.fig.subplots_adjust(top=.9)

g.fig.suptitle('Daily new case in each state over time', 

               fontsize=20)

plt.show()
a = temps[['Date','Confirmed','Deaths','Cured','New Confirmed','New Deaths']]
data_confirmed  = a.iloc[:,1]

data_confirmed
def prepare_data(data,length):

  hist = []

  target = []

  for i in range(len(data)-length):

    x = data[i:i+length]

    y = data[i+length]

    hist.append(x)

    target.append(y)

  return np.array(hist),np.array(target)
history,target = prepare_data(data_confirmed,30)

target = target.reshape(-1,1)

print(history.shape,target.shape)
print(history[1][29])

print(data_confirmed[29])

print(target[0])
scaler = MinMaxScaler()

history = scaler.fit_transform(history)

target = scaler.fit_transform(target)
history_scaled = history.reshape(history.shape[0],history.shape[1],1)
x_train = history_scaled[:80,:,:]

y_train = target[:80,:]

x_test = history_scaled[80:,:,:]

y_test = target[80:,:]
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
model = Sequential()

model.add(LSTM(units=80,batch_input_shape=(None,30,1),return_sequences=True))

model.add(LSTM(units=35, return_sequences=False))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

model.summary()
his = model.fit(x_train,y_train,epochs=35,validation_data=(x_test,y_test))
loss = his.history['loss']

epoch_count = range(1, len(loss) + 1)

plt.figure(figsize=(12,8))

plt.plot(epoch_count, loss, 'r--')

plt.legend(['Training Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show()
train_prediction = model.predict(x_train)

prediction = model.predict(x_test)
plt.scatter(range(len(prediction)),prediction,c='r')

plt.scatter(range(len(y_test)),y_test,c='g')

plt.show()
prediction_transformed = scaler.inverse_transform(prediction)

y_test_transformed = scaler.inverse_transform(y_test)
plt.figure(figsize=(16,8))

plt.plot(y_test_transformed, color='blue', label='Real')

plt.plot(prediction_transformed, color='red', label='Prediction')

plt.title('Confirmed Cases Prediction')

plt.ylabel('Cases')

plt.xlabel('Days')

plt.legend()

plt.show()