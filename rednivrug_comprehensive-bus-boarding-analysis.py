%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime

import os

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from math import sqrt

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input/unisys/ptsboardingsummary"))

# Any results you write to the current directory are saved as output.
import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from bubbly.bubbly import bubbleplot

init_notebook_mode(connected=True)



from bokeh.plotting import figure, save

from bokeh.io import output_file, output_notebook, show

from bokeh.models import ColumnDataSource, GMapOptions,HoverTool

from bokeh.plotting import gmap



import tensorflow as tf

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Input, Dense, GRU,LSTM, Embedding

from tensorflow.python.keras.optimizers import RMSprop

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
## For Multiple Output in single cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
data = pd.read_csv('../input/unisys/ptsboardingsummary/20140711.CSV')
out_geo = pd.read_csv('../input/outgeo/output_geo.csv')

route = pd.read_csv('../input/trann11/transit/routes.csv')
data.shape

data.head(2)
route.head(2)

out_geo.head(2)
from math import sin, cos, sqrt, atan2, radians

def calc_dist(lat1,lon1):

    ## approximate radius of earth in km

    R = 6373.0

    dlon = radians(138.604801) - radians(lon1)

    dlat = radians(-34.921247) - radians(lat1)

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(-34.921247)) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
out_geo['dist_from_centre'] = out_geo[['latitude','longitude']].apply(lambda x: calc_dist(*x), axis=1)
##Fill the missing values with mode

out_geo['type'].fillna('street_address',inplace=True)

out_geo['type'] = out_geo['type'].apply(lambda x: str(x).split(',')[-1])
out_geo['type'].unique()
'''Holidays--

2013-09-01,Father's Day

2013-10-07,Labour day

2013-12-25,Christmas day

2013-12-26,Proclamation Day

2014-01-01,New Year

2014-01-27,Australia Day

2014-03-10,March Public Holiday

2014-04-18,Good Friday

2014-04-19,Easter Saturday

2014-04-21,Easter Monday

2014-04-25,Anzac Day

2014-06-09,Queen's Birthday'''
def holiday_label (row):

    if row == datetime.date(2013, 9, 1) :

          return '1'

    if row == datetime.date(2013, 10, 6) :

          return '1'

    if row == datetime.date(2013, 12, 22) :

          return '2'

    if row == datetime.date(2013, 12, 29):

          return '1'

    if row  == datetime.date(2014, 1, 26):

          return '1'

    if row == datetime.date(2014, 3, 9):

          return '1'

    if row == datetime.date(2014, 4, 13) :

          return '2'

    if row == datetime.date(2014, 4, 20):

          return '2'

    if row == datetime.date(2014, 6, 8):

          return '1'

    return '0'
data['WeekBeginning'] = pd.to_datetime(data['WeekBeginning']).dt.date
data['holiday_label'] = data['WeekBeginning'].apply (lambda row: holiday_label(row))
data= pd.merge(data,out_geo,how='left',left_on = 'StopName',right_on = 'input_string')
data = pd.merge(data, route, how='left', left_on = 'RouteID', right_on = 'route_id')
col = ['TripID', 'RouteID', 'StopID', 'StopName', 'WeekBeginning','NumberOfBoardings','formatted_address',

      'latitude', 'longitude','postcode','type','route_desc','dist_from_centre','holiday_label']
data = data[col]
##saving the final dataset

data.to_csv('Weekly_Boarding.csv',index=False)
## getting the addresses for geolocation api.

# Address data['StopName'].unique()

# sub = pd.DataFrame({'Address': Address})

# sub=sub.reindex(columns=["Address"])

# sub.to_csv('addr.csv')
# st_week_grp1 = pd.DataFrame(data.groupby(['StopName','WeekBeginning','type']).agg({'NumberOfBoardings': ['sum', 'count']})).reset_index()

grouped = data.groupby(['StopName','WeekBeginning','type']).agg({'NumberOfBoardings': ['sum', 'count','max']})

grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]
st_week_grp = pd.DataFrame(grouped).reset_index()

st_week_grp.shape

st_week_grp.head()
st_week_grp1 = pd.DataFrame(st_week_grp.groupby('StopName')['WeekBeginning'].count()).reset_index()
aa=list(st_week_grp1[st_week_grp1['WeekBeginning'] == 54]['StopName'])
bb = st_week_grp[st_week_grp['StopName'].isin(aa)]
## save the aggregate data

bb.to_csv('st_week_grp.csv', index=False)
data.nunique()
data.shape

data.columns

data.head(3)
data.isnull().sum()
data['WeekBeginning'].unique()
##can assign the each chart to one axes at a time

fig,axrr=plt.subplots(3,2,figsize=(18,18))



data['NumberOfBoardings'].value_counts().sort_index().head(20).plot.bar(ax=axrr[0][0])

data['WeekBeginning'].value_counts().plot.area(ax=axrr[0][1])

data['RouteID'].value_counts().head(20).plot.bar(ax=axrr[1][0])

data['RouteID'].value_counts().tail(20).plot.bar(ax=axrr[1][1])

data['type'].value_counts().head(5).plot.bar(ax=axrr[2][0])

data['type'].value_counts().tail(10).plot.bar(ax=axrr[2][1])
data['postcode'].value_counts().head(20).plot.bar()
# data['dist_from_centre'].nunique()

bb_grp = data.groupby(['dist_from_centre']).agg({'NumberOfBoardings': ['sum']}).reset_index()

bb_grp.columns = bb_grp.columns.get_level_values(0)

bb_grp.head()

bb_grp.columns
trace0 = go.Scatter(

    x = bb_grp['dist_from_centre'],

    y = bb_grp['NumberOfBoardings'],mode = 'lines+markers',name = 'X2 King William St')



data1 = [trace0]

layout = dict(title = 'Distance Vs Number of boarding',

              xaxis = dict(title = 'Distance from centre'),

              yaxis = dict(title = 'Number of Boardings'))

fig = dict(data=data1, layout=layout)

iplot(fig)
lat = out_geo['latitude'].tolist()

long = out_geo['longitude'].tolist()

nam = out_geo['input_string'].tolist()
map_options = GMapOptions(lat=-34.96, lng=138.592, map_type="roadmap", zoom=9)

key = open('../input/geolockey/api_key.txt').read()

p = gmap(key, map_options, title="Adelaide South Australia")

source = ColumnDataSource(data=dict(lat=lat,lon=long,nam=nam))



p.circle(x="lon", y="lat", size=5, fill_color="blue", fill_alpha=0.8, source=source)

TOOLTIPS = [("Place", "@nam")]

p.add_tools( HoverTool(tooltips=TOOLTIPS))

output_notebook()

show(p)
## for finding highest number of Boarding Bus stops

bb_grp = bb.groupby(['StopName']).agg({'NumberOfBoardings_sum': ['sum']}).reset_index()['NumberOfBoardings_sum'].sort_values('sum')

bb_grp[1000:1005]

bb.groupby(['StopName']).agg({'NumberOfBoardings_sum': ['sum']}).reset_index().iloc[[2325,1528,546,1043,1905]]

# bb_grp.iloc[[3054]]
source_1 = bb[bb['StopName'] == 'X2 King William St'].reset_index(drop = True)

source_2 = bb[bb['StopName'] == 'E1 Currie St'].reset_index(drop = True)

source_3 = bb[bb['StopName'] == 'I2 North Tce'].reset_index(drop = True)

source_4 = bb[bb['StopName'] == 'F2 Grenfell St'].reset_index(drop = True)

source_5 = bb[bb['StopName'] == 'D1 King William St'].reset_index(drop = True)
trace0 = go.Scatter(

    x = source_1['WeekBeginning'],

    y = source_1['NumberOfBoardings_sum'],mode = 'lines+markers',name = 'X2 King William St')

trace1 = go.Scatter(

    x = source_2['WeekBeginning'],

    y = source_2['NumberOfBoardings_sum'],mode = 'lines+markers',name = 'E1 Currie St')

trace2 = go.Scatter(

    x = source_3['WeekBeginning'],

    y = source_3['NumberOfBoardings_sum'],mode = 'lines+markers',name = 'I2 North Tce')

trace3 = go.Scatter(

    x = source_4['WeekBeginning'],

    y = source_4['NumberOfBoardings_sum'],mode = 'lines+markers',name = 'F2 Grenfell St')

trace4 = go.Scatter(

    x = source_5['WeekBeginning'],

    y = source_5['NumberOfBoardings_sum'],mode = 'lines+markers',name = 'D1 King William St')



data = [trace0,trace1,trace2,trace3,trace4]

layout = dict(title = 'Weekly Boarding Total',

              xaxis = dict(title = 'Week Number'),

              yaxis = dict(title = 'Number of Boardings'),

              shapes = [{# Holidays Record: 2013-09-01

'type': 'line','x0': '2013-09-01','y0': 0,'x1': '2013-09-02','y1': 18000,'line': {

        'color': 'rgb(55, 128, 191)','width': 1,'dash': 'dashdot'},},

              {# 2013-10-07

'type': 'line','x0': '2013-10-07','y0': 0,'x1': '2013-10-07','y1': 18000,'line': {

        'color': 'rgb(55, 128, 191)','width': 1,'dash': 'dashdot'},},

              {# 2013-12-25

'type': 'line','x0': '2013-12-25','y0': 0,'x1': '2013-12-26','y1': 18000,'line': {

        'color': 'rgb(55, 128, 191)','width': 3,'dash': 'dashdot'},},

              {# 2014-01-27

'type': 'line','x0': '2014-01-27','y0': 0,'x1': '2014-01-28','y1': 18000,'line': {

        'color': 'rgb(55, 128, 191)','width': 1,'dash': 'dashdot'},},

              {# 2014-03-10

'type': 'line','x0': '2014-03-10','y0': 0,'x1': '2014-03-11','y1': 18000,'line': {

        'color': 'rgb(55, 128, 191)','width': 1,'dash': 'dashdot'},},

              {# 2014-04-18

'type': 'line','x0': '2014-04-18','y0': 0,'x1': '2014-04-19','y1': 18000,'line': {

        'color': 'rgb(55, 128, 191)','width': 3,'dash': 'dashdot'},},

              {# 2014-06-09

'type': 'line','x0': '2014-06-09','y0': 0,'x1': '2014-06-10','y1': 18000,'line': {

        'color': 'rgb(55, 128, 191)','width': 1,'dash': 'dashdot'},},])

fig = dict(data=data, layout=layout)

iplot(fig)
source_6 = bb[bb['StopName'] == '57A Hancock Rd'].reset_index(drop = True)

source_7 = bb[bb['StopName'] == '37 Muriel Dr'].reset_index(drop = True)

source_8 = bb[bb['StopName'] == '18B Springbank Rd'].reset_index(drop = True)

source_9 = bb[bb['StopName'] == '27E Sir Ross Smith Av'].reset_index(drop = True)

source_10 = bb[bb['StopName'] == '46A Baldock Rd'].reset_index(drop = True)
trace0 = go.Scatter(

    x = source_6['WeekBeginning'],

    y = source_6['NumberOfBoardings_sum'],mode = 'lines+markers',name = '57A Hancock Rd')

trace1 = go.Scatter(

    x = source_7['WeekBeginning'],

    y = source_7['NumberOfBoardings_sum'],mode = 'lines+markers',name = '37 Muriel Dr')

trace2 = go.Scatter(

    x = source_8['WeekBeginning'],

    y = source_8['NumberOfBoardings_sum'],mode = 'lines+markers',name = '18B Springbank Rd')

trace3 = go.Scatter(

    x = source_9['WeekBeginning'],

    y = source_9['NumberOfBoardings_sum'],mode = 'lines+markers',name = '27E Sir Ross Smith Av')

trace4 = go.Scatter(

    x = source_10['WeekBeginning'],

    y = source_10['NumberOfBoardings_sum'],mode = 'lines+markers',name = '46A Baldock Rd')



data = [trace0,trace1,trace2,trace3,trace4]

layout = dict(title = 'Weekly Boarding Total',

              xaxis = dict(title = 'Week Number'),

              yaxis = dict(title = 'Number of Boardings'),

              shapes = [{# Holidays Record: 2013-09-01

'type': 'line','x0': '2013-09-01','y0': 0,'x1': '2013-09-02','y1': 80,'line': {

        'color': 'rgb(55, 128, 191)','width': 1,'dash': 'dashdot'},},

              {# 2013-10-07

'type': 'line','x0': '2013-10-07','y0': 0,'x1': '2013-10-07','y1': 80,'line': {

        'color': 'rgb(55, 128, 191)','width': 1,'dash': 'dashdot'},},

              {# 2013-12-25

'type': 'line','x0': '2013-12-25','y0': 0,'x1': '2013-12-26','y1': 80,'line': {

        'color': 'rgb(55, 128, 191)','width': 3,'dash': 'dashdot'},},

              {# 2014-01-27

'type': 'line','x0': '2014-01-27','y0': 0,'x1': '2014-01-28','y1': 80,'line': {

        'color': 'rgb(55, 128, 191)','width': 1,'dash': 'dashdot'},},

              {# 2014-03-10

'type': 'line','x0': '2014-03-10','y0': 0,'x1': '2014-03-11','y1': 80,'line': {

        'color': 'rgb(55, 128, 191)','width': 1,'dash': 'dashdot'},},

              {# 2014-04-18

'type': 'line','x0': '2014-04-18','y0': 0,'x1': '2014-04-19','y1': 80,'line': {

        'color': 'rgb(55, 128, 191)','width': 3,'dash': 'dashdot'},},

              {# 2014-06-09

'type': 'line','x0': '2014-06-09','y0': 0,'x1': '2014-06-10','y1': 80,'line': {

        'color': 'rgb(55, 128, 191)','width': 1,'dash': 'dashdot'},},])

fig = dict(data=data, layout=layout)

iplot(fig)
bb1=bb.copy()
## Label encode the Date type for easy Plotting

le = LabelEncoder()

bb1['WeekBeginning'] = le.fit_transform(bb1['WeekBeginning'])
figure = bubbleplot(dataset=bb1, x_column='NumberOfBoardings_sum', y_column='NumberOfBoardings_count', 

    bubble_column='StopName', time_column='WeekBeginning', size_column='NumberOfBoardings_max',

    color_column='type', 

    x_title="Total Boardings", y_title="Frequency Of Boardings",show_slider=True,

    title='Adelaide Weekly Bus Transport Summary 2D',x_logscale=True, scale_bubble=2,height=650)



iplot(figure, config={'scrollzoom': True})
figure = bubbleplot(dataset=bb1[bb1['StopName'].isin(bb1['StopName'].unique()[:30])], x_column='NumberOfBoardings_sum', y_column='NumberOfBoardings_count', 

    bubble_column='StopName', time_column='WeekBeginning', size_column='NumberOfBoardings_max',

    color_column='type', 

    x_title="Total Boardings", y_title="Frequency Of Boardings",show_slider=False,

    title='Adelaide Weekly Bus Transport Summary 2D',x_logscale=True, scale_bubble=2,height=650)



iplot(figure, config={'scrollzoom': True})
figure = bubbleplot(dataset=bb1, x_column='NumberOfBoardings_sum', y_column='NumberOfBoardings_count', 

    bubble_column='StopName', time_column='WeekBeginning', z_column='NumberOfBoardings_max',

    color_column='type',show_slider=False, 

    x_title="Total Boardings", y_title="Frequency Of Boardings", z_title="Maximum Boardings",

    title='Adelaide Weekly Bus Transport Summary 3D', x_logscale=True, z_logscale=True,y_logscale=True,

    scale_bubble=0.8, marker_opacity=0.8, height=700)



iplot(figure, config={'scrollzoom': True})
d=[]

for i in bb['StopName'].unique():

    d.append({'StopName': i,'Boarding_sum':np.sum(bb[bb['StopName'] == i]['NumberOfBoardings_sum'].pct_change())/54,

             'Boarding_count':np.sum(bb[bb['StopName'] == i]['NumberOfBoardings_count'].pct_change())/54,

             'Boarding_max':np.sum(bb[bb['StopName'] == i]['NumberOfBoardings_max'].pct_change())/54})

pct_chng = pd.DataFrame(d)
#pct_chng.head()

pct_chng['Boarding_sum'].nlargest(5)

pct_chng['Boarding_sum'].nsmallest(5)

pct_chng[pct_chng['Boarding_sum']<0].shape

pct_chng.iloc[[3110,2134,214,1538,1290]]
bb1 = pd.merge(bb, out_geo, how='left', left_on = 'StopName', right_on = 'input_string')
bb1['holiday_label'] = bb1['WeekBeginning'].apply (lambda row: holiday_label(row))
##Final 11 features have been used for the forecastng.

cols = ['StopName','WeekBeginning','type_x','NumberOfBoardings_sum','NumberOfBoardings_count','NumberOfBoardings_max','latitude','longitude','postcode','dist_from_centre','holiday_label']

bb1=bb1[cols]

bb1.shape

bb1.head()
##Replace all Nan by Mode

for i in bb1.columns:

    bb1[i].fillna(bb1[i].mode()[0], inplace=True)

bb1[["postcode", "holiday_label"]] = bb1[["postcode", "holiday_label"]].apply(pd.to_numeric)
le = LabelEncoder()

bb1['StopName'] = le.fit_transform(bb1['StopName'])

bb1['type_x'] = le.fit_transform(bb1['type_x'])
train = bb1[bb1['WeekBeginning'] < datetime.date(2014, 6, 1)]

test = bb1[bb1['WeekBeginning'] >= datetime.date(2014, 6, 1)]

train.shape

test.shape
le = LabelEncoder()

train['WeekBeginning'] = le.fit_transform(train['WeekBeginning'])

test['WeekBeginning'] = le.fit_transform(test['WeekBeginning'])
# tr_col = ['StopName', 'WeekBeginning', 'type_x', 'latitude',

#        'longitude', 'postcode', 'dist_from_centre', 'holiday_label']

# tr_target = ['StopName','NumberOfBoardings_sum','NumberOfBoardings_count','NumberOfBoardings_max']

# train1 = train[tr_col]

# test1 = test[tr_col]

# train_tg = train[tr_target]

# test_tg = test[tr_target]
# ## model each StopName Separately

# train.StopName.nunique()

# for i in train['StopName']:

#     col = 'NumberOfBoardings_sum'

#     train_x = train1[train1['StopName']==i]

#     test_x = test1[test1['StopName']==i]

#     tr_target = train_tg[train_tg['StopName'] ==i][col]

#     ts_target = test_tg[test_tg['StopName'] == i][col]

#     print(i,train_x.shape,test_x.shape,tr_target.shape,ts_target.shape)

#     xgb_model = xg.XGBRegressor()

#     xgb_model.fit(train_x.values,tr_target.values)

#     preds = xgb_model.predict(test_x.values)

#     print('original ',ts_target)

#     print('prediction: ',preds)

#     break
tr_col = ['StopName', 'WeekBeginning', 'type_x', 'latitude',

       'longitude', 'postcode', 'dist_from_centre', 'holiday_label']

train_sum_y = train[['StopName','NumberOfBoardings_sum']]

train_count_y = train[['StopName','NumberOfBoardings_count']]

train_max_y = train[['StopName','NumberOfBoardings_max']]

train_x = train[tr_col]

test_x = test[tr_col]



test_sum_y = test[['StopName','NumberOfBoardings_sum']]

test_count_y = test[['StopName','NumberOfBoardings_count']]

test_max_y = test[['StopName','NumberOfBoardings_max']]
from sklearn.ensemble import RandomForestRegressor

# model = lgb.LGBMRegressor()

model = RandomForestRegressor(n_estimators=700, min_samples_leaf=3, max_features=0.5,n_jobs=-1)

# model = lgb.LGBMRegressor(max_depth=10,learning_rate=0.0227,n_estimators=195,num_leaves=11,reg_alpha=1.5764,reg_lambda=0.0478,subsample=0.7776,colsample_bytree=0.7761)

model.fit(train_x.values,train_sum_y['NumberOfBoardings_sum'].values)

preds = model.predict(test_x.values)
rms = sqrt(mean_squared_error(test_sum_y['NumberOfBoardings_sum'].values, preds))

rms
test_sum_y.values[:15]

preds[:15]
fig, ax = plt.subplots(figsize=(6,10))

lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

ax.grid(False)

plt.title("LightGBM - Feature Importance", fontsize=15)

plt.show()
plt.figure(figsize=(15,5))

plt.plot(test_sum_y['NumberOfBoardings_sum'].values, label='true')

plt.plot(preds, label='pred')

plt.ylabel("Total Number of Boarding")

plt.xlabel("Index")

plt.title("Comparison Between Prediction & True Values")

plt.legend()

plt.show()
bb1['WeekBeginning'] = le.fit_transform(bb1['WeekBeginning'])
df = bb1.sort_values(['WeekBeginning','StopName'])
##Replace all Nan by Mode

for i in df.columns:

    df[i].fillna(df[i].mode()[0], inplace=True)

df[["postcode", "holiday_label"]] = df[["postcode", "holiday_label"]].apply(pd.to_numeric)
target_names = ['NumberOfBoardings_sum', 'NumberOfBoardings_count', 'NumberOfBoardings_max']

train_col = ['StopName','WeekBeginning','type_x','latitude','longitude','postcode','dist_from_centre','holiday_label']

##want to predict 1 day in future.

shift_days = 6

shift_steps = shift_days * 3249
df_targets = df[target_names].shift(-shift_steps)

x_data = df.iloc[:,1:].values[0:-shift_steps]

y_data = df_targets.values[:-shift_steps]

print(type(y_data))

print("Shape:", y_data.shape)
##data split into 90% training and 10% testing

num_data = len(x_data)

train_split = 0.9

num_train = int(train_split * num_data)

x_train = x_data[0:num_train]

x_test = x_data[num_train:]

print(len(x_train) + len(x_test))
##target values for test and train

y_train = y_data[0:num_train]

y_test = y_data[num_train:]

print(len(y_train) + len(y_test))

##input dimension and output dimension

num_x_signals = x_data.shape[1]

print(num_x_signals)

num_y_signals = y_data.shape[1]

print(num_y_signals)
##scale data to get values between 0 to 1.

print("Min:", np.min(x_train))

print("Max:", np.max(x_train))

x_scaler = MinMaxScaler()

x_train_scaled = x_scaler.fit_transform(x_train)

print("Min:", np.min(x_train_scaled))

print("Max:", np.max(x_train_scaled))

x_test_scaled = x_scaler.transform(x_test)

y_scaler = MinMaxScaler()

y_train_scaled = y_scaler.fit_transform(y_train)

y_test_scaled = y_scaler.transform(y_test)

print(x_train_scaled.shape)

print(y_train_scaled.shape)
def batch_generator(batch_size, sequence_length):

    while True:

        # Allocate a new array for the batch of input,output signals.

        x_shape = (batch_size, sequence_length, num_x_signals)

        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        y_shape = (batch_size, sequence_length, num_y_signals)

        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        for i in range(batch_size):

            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.

            x_batch[i] = x_train_scaled[idx:idx+sequence_length]

            y_batch[i] = y_train_scaled[idx:idx+sequence_length]

        yield (x_batch, y_batch)
batch_size = 256

sequence_length = 1344

print(sequence_length)

generator = batch_generator(batch_size=batch_size,sequence_length=sequence_length)

x_batch, y_batch = next(generator)

print(x_batch.shape)

print(y_batch.shape)

validation_data = (np.expand_dims(x_test_scaled, axis=0),

                   np.expand_dims(y_test_scaled, axis=0))
##model

model = Sequential()

model.add(LSTM(units=512,return_sequences=True,input_shape=(None, num_x_signals,)))

model.add(Dense(num_y_signals, activation='sigmoid'))
#loss function define.

warmup_steps = 0

def loss_mse_warmup(y_true, y_pred):

    # [batch_size, sequence_length, num_y_signals].

    y_true_slice = y_true[:, warmup_steps:, :]

    y_pred_slice = y_pred[:, warmup_steps:, :]

    # Calculate the MSE loss for each value in these tensors.

    loss = tf.losses.mean_squared_error(labels=y_true_slice,predictions=y_pred_slice)

    loss_mean = tf.reduce_mean(loss)

    return loss_mean
##optimizer and model summary

optimizer = RMSprop(lr=1e-3)

model.compile(loss=loss_mse_warmup, optimizer=optimizer)

print(model.summary())
##early stopping and learning rate decrease callbacks

callback_early_stopping = EarlyStopping(monitor='val_loss',patience=5, verbose=1)

callbacks = [callback_early_stopping]
%%time

#model.fit(generator=generator,epochs=2,steps_per_epoch=5,validation_data=validation_data,callbacks=callbacks)
# model.load_weights(path_checkpoint)

# result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),

#                         y=np.expand_dims(y_test_scaled, axis=0))

# print("loss (test-set):", result)
## Training on Gru Model take more memory than whats available on kaggle.So need to Comment out that part.