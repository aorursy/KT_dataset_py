import pandas as pd

import numpy as np

import datetime

import time



from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import davies_bouldin_score



import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Dropout

from tensorflow.keras.losses import MSE
Case = pd.read_csv("../input/coronavirusdataset/Case.csv")

PatientInfo = pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")

PatientRoute = pd.read_csv("../input/coronavirusdataset/PatientRoute.csv")

Region = pd.read_csv("../input/coronavirusdataset/Region.csv")

SearchTrend = pd.read_csv("../input/coronavirusdataset/SearchTrend.csv")

SeoulFloating = pd.read_csv("../input/coronavirusdataset/SeoulFloating.csv")

Time = pd.read_csv("../input/coronavirusdataset/Time.csv")

TimeAge = pd.read_csv("../input/coronavirusdataset/TimeAge.csv")

TimeGender = pd.read_csv("../input/coronavirusdataset/TimeGender.csv")

TimeProvince = pd.read_csv("../input/coronavirusdataset/TimeProvince.csv")

Weather = pd.read_csv("../input/coronavirusdataset/Weather.csv")
p_info = PatientInfo[['patient_id', 'sex', 'age','country',

                      'province', 'city', 'infection_case', 'infection_order',

                      'confirmed_date', 'state']]



p_info = p_info.fillna(method = 'ffill')

#Calculatiing exact year of patients

p_info['age'] = PatientInfo['birth_year'].apply(lambda x: 2020 - x)



#Assigning numerical value to all string columns

sexes = {}

countries = {}

provinces = {}

cities = {}

infection_cases = {}

states = {}





for i in p_info['sex'].unique():

  key = list(p_info['sex'].unique()).index(i)

  sexes[key] = i



for i in p_info['country'].unique():

  key = list(p_info['country'].unique()).index(i)

  countries[key] = i



for i in p_info['province'].unique():

  key = list(p_info['province'].unique()).index(i)

  provinces[key] = i



for i in p_info['city'].unique():

  key = list(p_info['city'].unique()).index(i)

  cities[key] = i



for i in p_info['infection_case'].unique():

  key = list(p_info['infection_case'].unique()).index(i)

  infection_cases[key] = i



for i in p_info['state'].unique():

  key = list(p_info['state'].unique()).index(i)

  states[key] = i



for i in range(0, len(p_info)):

  p_info.loc[i,'sex'] = (list(sexes.keys())[list(sexes.values()).index(p_info.iloc[i]['sex'])])

  p_info.loc[i,'country'] = (list(countries.keys())[list(countries.values()).index(p_info.iloc[i]['country'])])

  p_info.loc[i,'province'] = (list(provinces.keys())[list(provinces.values()).index(p_info.iloc[i]['province'])])

  p_info.loc[i,'city'] = (list(cities.keys())[list(cities.values()).index(p_info.iloc[i]['city'])])

  p_info.loc[i,'infection_case'] = (list(infection_cases.keys())[list(infection_cases.values()).index(p_info.iloc[i]['infection_case'])])

  p_info.loc[i,'state'] = (list(states.keys())[list(states.values()).index(p_info.iloc[i]['state'])])





p_info['confirmed_date'] = p_info['confirmed_date'].apply(

    lambda x: time.mktime(

        datetime.datetime.strptime(str(x), "%Y-%m-%d").timetuple()))



p_info['age'] = p_info['age'].fillna(method = 'ffill')



p_info = p_info.set_index('patient_id')
#Take a look to our new dataset

p_info.head()
#Perform a clustering on our new dataset

scaler = MinMaxScaler()

scaler.fit(p_info)

p_info_scaled = scaler.transform(p_info)



kmeans = KMeans(n_clusters=3)

kmeans.fit(p_info_scaled)

labels = kmeans.predict(p_info_scaled)

print("Davies-Bouldin score for clustering is :{}".format(davies_bouldin_score(p_info, labels)))
p_info['cluster'] = labels

# Make the plot

fig1 = px.parallel_coordinates(p_info, color="cluster")

py.offline.iplot(fig1)
from pandas.plotting import scatter_matrix

scatter_matrix(p_info, alpha=0.2, figsize=(15, 15), diagonal='kde')
#Plotting based on cluster

p_new = PatientInfo[['patient_id', 'sex', 'age','country',

                      'province', 'city', 'infection_case', 'infection_order',

                      'confirmed_date', 'state']]

p_new['cluster'] = labels

temp = p_new[['sex', 'cluster']]

temp = temp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=3,

                     specs=[[{'type':'domain'}, {'type':'domain'},

                             {'type':'domain'}]])

col = 1

for item in p_new['cluster'].unique():

  grp = temp[temp['cluster'] == item].groupby(['sex'])

  print(grp.size())

  vals = dict(grp.size())

  sum_val = 0

  to_plot = {}

  for key in vals:

    sum_val += vals[key]

  for key in vals:

    to_plot[key] = (vals[key]/sum_val) * 100

    

  fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                       values=list(to_plot.values()),

                        name="Cluster " + str(item)),

                  row=1, col=col)

  col += 1



fig1.update_layout(title_text="Sexes Based on cluster")

py.offline.iplot(fig1)
temp = p_new[['age', 'cluster']]

temp = temp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=3,

                     specs=[[{'type':'domain'}, {'type':'domain'},

                             {'type':'domain'}]])

col = 1

for item in p_new['cluster'].unique():

  grp = temp[temp['cluster'] == item].groupby(['age'])

  print(grp.size())

  vals = dict(grp.size())

  sum_val = 0

  to_plot = {}

  for key in vals:

    sum_val += vals[key]

  for key in vals:

    to_plot[key] = (vals[key]/sum_val) * 100

    

  fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                       values=list(to_plot.values()),

                        name="Cluster " + str(item)),

                  row=1, col=col)

  col += 1



fig1.update_layout(title_text="Age Based on cluster")

py.offline.iplot(fig1)
temp = p_new[['country', 'cluster']]

temp = temp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=3,

                     specs=[[{'type':'domain'}, {'type':'domain'},

                             {'type':'domain'}]])

col = 1

for item in p_new['cluster'].unique():

  grp = temp[temp['cluster'] == item].groupby(['country'])

  print(grp.size())

  vals = dict(grp.size())

  sum_val = 0

  to_plot = {}

  for key in vals:

    sum_val += vals[key]

  for key in vals:

    to_plot[key] = (vals[key]/sum_val) * 100

    

  fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                       values=list(to_plot.values()),

                        name="Cluster " + str(item)),

                  row=1, col=col)

  col += 1



fig1.update_layout(title_text="Countries Based on cluster")

py.offline.iplot(fig1)
temp = p_new[['province', 'cluster']]

temp = temp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=3,

                     specs=[[{'type':'domain'}, {'type':'domain'},

                             {'type':'domain'}]])

col = 1

for item in p_new['cluster'].unique():

  grp = temp[temp['cluster'] == item].groupby(['province'])

  print(grp.size())

  vals = dict(grp.size())

  sum_val = 0

  to_plot = {}

  for key in vals:

    sum_val += vals[key]

  for key in vals:

    to_plot[key] = (vals[key]/sum_val) * 100

    

  fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                       values=list(to_plot.values()),

                        name="Cluster " + str(item)),

                  row=1, col=col)

  col += 1



fig1.update_layout(title_text="Provinces Based on cluster")

py.offline.iplot(fig1)
temp = p_new[['state', 'cluster']]

temp = temp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=3,

                     specs=[[{'type':'domain'}, {'type':'domain'},

                             {'type':'domain'}]])

col = 1

for item in p_new['cluster'].unique():

  grp = temp[temp['cluster'] == item].groupby(['state'])

  print(grp.size())

  vals = dict(grp.size())

  sum_val = 0

  to_plot = {}

  for key in vals:

    sum_val += vals[key]

  for key in vals:

    to_plot[key] = (vals[key]/sum_val) * 100

    

  fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                       values=list(to_plot.values()),

                        name="Cluster " + str(item)),

                  row=1, col=col)

  col += 1



fig1.update_layout(title_text="States Based on cluster")

py.offline.iplot(fig1)
temp = p_new[['city', 'cluster']]

temp = temp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=3,

                     specs=[[{'type':'domain'}, {'type':'domain'},

                             {'type':'domain'}]])

col = 1

for item in p_new['cluster'].unique():

  grp = temp[temp['cluster'] == item].groupby(['city'])

  print(grp.size())

  vals = dict(grp.size())

  sum_val = 0

  to_plot = {}

  for key in vals:

    sum_val += vals[key]

  for key in vals:

    to_plot[key] = (vals[key]/sum_val) * 100

    

  fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                       values=list(to_plot.values()),

                        name="Cluster " + str(item)),

                  row=1, col=col)

  col += 1



fig1.update_layout(title_text="Cities Based on cluster")

py.offline.iplot(fig1)
temp = PatientInfo

temp = temp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=1,

                     specs=[[{'type':'domain'}]])

col = 1



grp = temp.groupby(['city'])

print(grp.size())

vals = dict(grp.size())

sum_val = 0

to_plot = {}

for key in vals:

  sum_val += vals[key]

for key in vals:

  to_plot[key] = (vals[key]/sum_val) * 100

    

fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                      values=list(to_plot.values()),

                      name="Cities"),

                row=1, col=col)



fig1.update_layout(title_text="Cities")

py.offline.iplot(fig1)
temp = PatientInfo

temp = temp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=1,

                     specs=[[{'type':'domain'}]])

col = 1



grp = temp.groupby(['province'])

print(grp.size())

vals = dict(grp.size())

sum_val = 0

to_plot = {}

for key in vals:

  sum_val += vals[key]

for key in vals:

  to_plot[key] = (vals[key]/sum_val) * 100

    

fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                      values=list(to_plot.values()),

                      name="Provinces"),

                row=1, col=col)



fig1.update_layout(title_text="Provinces")

py.offline.iplot(fig1)
temp = PatientInfo

temp = temp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=1,

                     specs=[[{'type':'domain'}]])

col = 1



grp = temp.groupby(['state'])

print(grp.size())

vals = dict(grp.size())

sum_val = 0

to_plot = {}

for key in vals:

  sum_val += vals[key]

for key in vals:

  to_plot[key] = (vals[key]/sum_val) * 100

    

fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                      values=list(to_plot.values()),

                      name="States"),

                row=1, col=col)



fig1.update_layout(title_text="States")

py.offline.iplot(fig1)
temp = PatientInfo

temp = temp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=1,

                     specs=[[{'type':'domain'}]])

col = 1



grp = temp.groupby(['sex'])

print(grp.size())

vals = dict(grp.size())

sum_val = 0

to_plot = {}

for key in vals:

  sum_val += vals[key]

for key in vals:

  to_plot[key] = (vals[key]/sum_val) * 100

    

fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                      values=list(to_plot.values()),

                      name="Sexes"),

                row=1, col=col)



fig1.update_layout(title_text="Sexes")

py.offline.iplot(fig1)
sizes = list(

    PatientRoute[['longitude', 'latitude']].groupby(

        ['longitude', 'latitude']).size())

fig = go.Figure(data=go.Scattergeo(

        lon = PatientRoute['longitude'],

        lat = PatientRoute['latitude'],

        mode = 'markers'

        ))

fig.update_layout(

        title = 'Infected cordinates',

        geo_scope='asia'

    )

py.offline.iplot(fig)
fp = SeoulFloating[['city', 'fp_num']]

fp = fp.fillna('Unknown')

fig1 = make_subplots(rows=1, cols=1,

                     specs=[[{'type':'domain'}]])

col = 1



grp = fp.groupby(['city'])

vals = pd.DataFrame(grp.sum().reset_index())

sum_val = 0

to_plot = {}

for item in vals['fp_num']:

  sum_val += item

key = vals['city']

for item in range(len(vals['fp_num'])):

  key = vals['city'].iloc[item]

  to_plot[key] = (int(vals['fp_num'].iloc[item])/sum_val) * 100

    

fig1.add_trace(go.Pie(labels=list(to_plot.keys()),

                      values=list(to_plot.values()),

                      name="Cities"),

                row=1, col=col)



fig1.update_layout(title_text="Cities")

py.offline.iplot(fig1)
time_df = Time[['test', 'negative', 'released',

       'deceased', 'confirmed']]



def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset) - look_back - 1):

        a = dataset[i:(i + look_back), :]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 2])

    return np.array(dataX), np.array(dataY)



dataset = time_df.values

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.7) 

test_size = len(dataset) - train_size

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]





look_back = 10

trainX, trainY = create_dataset(train, look_back)  

testX, testY = create_dataset(test, look_back)



trainX = np.reshape(trainX, (trainX.shape[0], look_back, 5))

testX = np.reshape(testX, (testX.shape[0],look_back, 5))





model = Sequential()

model.add(LSTM(5, return_sequences=True, input_shape=(look_back, 5)))

model.add(LSTM(5, return_sequences=True, input_shape=(look_back, 5)))

model.add(LSTM(5, return_sequences=True, input_shape=(look_back, 5)))

model.add(LSTM(5, return_sequences=True, input_shape=(look_back, 5)))

model.add(LSTM(5, input_shape=(look_back, 5)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='sgd')

history= model.fit(trainX, trainY, validation_split=0.33, nb_epoch=200,

                   batch_size=32)
#Make new prediction

look_back = 10

x, y = create_dataset(dataset, look_back)

x = np.reshape(x, (x.shape[0], look_back, 5))



predicted = model.predict(x)
fig = go.Figure()

fig.add_trace(go.Scatter(x=Time['date'], y=y,

                    mode='lines',

                    name='Original'))

fig.add_trace(go.Scatter(x=Time['date'], y=predicted.reshape(y.shape),

                    mode='lines',

                    name='Predicted'))

py.offline.iplot(fig)