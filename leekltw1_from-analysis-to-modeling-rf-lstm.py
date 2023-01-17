import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm

import warnings

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf

%matplotlib inline

warnings.filterwarnings('ignore')

data = pd.read_csv('../input/2015_Air_quality_in_northern_Taiwan.csv')
# We want to predict PM2.5, so we delete the rows with NA in 'PM2.5'

data = data[data['PM2.5'].notna()]

data.info()
(data.isna().sum() / len(data)).sort_values(ascending=False)
plt.figure(figsize=(20,5))

plt.title('The NA ratio in each column')

plt.xticks(rotation='vertical')

plt.plot([0,22],[0.5,0.5],'g:')

plt.plot((data.isna().sum() / len(data)).sort_values(ascending=False).index,

         (data.isna().sum() / len(data)).sort_values(ascending=False).values,'-',label=r'$NA \ ratio = \frac{counts \ of \ NA}{Total \ row \ of \ NA}$')

plt.annotate('A notable shrink', xy=(6.5, 0.15), xytext=(3, 0.6),fontsize='x-large',

            arrowprops={'facecolor':'black','shrink':1.0}

            )

plt.xlim(0,22)

_ = plt.legend(fontsize='x-large')
data['UVB'].value_counts()
data.drop(['UVB','RAIN_COND','PH_RAIN'],axis=1,inplace=True)
data[data['CO'].isna()].head()
plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.title('NA counts in each row')

plt.plot(data.isna().sum(axis=1).value_counts().sort_index())

plt.xlabel('Number of NA in each row')

plt.ylabel('Counts')

plt.xlim(0)

plt.ylim(0)



plt.subplot(1,2,2)

plt.title('Accumulated NA counts in each row')

plt.plot(data.isna().sum(axis=1).value_counts().sort_index().cumsum())

plt.annotate('Exclude rows with more than 10 NAs \n will last  215716 rows', xy=(10, 210000), xytext=(10, 160000),fontsize='medium',

            arrowprops={'facecolor':'black','shrink':1.0})

plt.annotate('Exclude rows with more than 3 NAs \n will last 180129 rows', xy=(3, 170000), xytext=(3, 100000),fontsize='medium',

            arrowprops={'facecolor':'black','shrink':1.0})

plt.xlabel('Accumulated Number of NA in each row')

plt.ylabel('Accumulated Counts')

plt.xlim(0)

_= plt.ylim(0)
data = data.dropna(thresh=17) # 17 = len(data.columns) - 3
def numeric(row):

    try:

        if np.isnan(row):

            return

        else:

            row =str(row)

            return float(row.replace('x','').replace('#','').replace('*',''))

    except TypeError:

        row =str(row)

        return float(row.replace('x','').replace('#','').replace('*',''))
data['WS_HR'] = data['WS_HR'].apply(numeric)

print(data['WS_HR'].describe())

print('\nThe skewness:',data['WS_HR'].skew())

print('Right skewed') if data['WS_HR'].skew()>0 else print('Left skewed')
plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.plot([data['WS_HR'].median(),data['WS_HR'].median()],[0,7000],'g:',label='median={0:.2f}'.format(data['WS_HR'].median()))

plt.plot([data['WS_HR'].mean(),data['WS_HR'].mean()],[0,7000],'r:',label='mean   ={0:.2f}'.format(data['WS_HR'].mean()))

plt.plot([data['WS_HR'].mode(),data['WS_HR'].mode()],[0,7000],'y:',label='mode   ={}'.format(data['WS_HR'].mode()[0]))

plt.plot(data['WS_HR'].value_counts().sort_index(),label='distribution')

plt.legend(loc='upper right')

plt.xlim(0)

plt.ylim(0)

plt.subplot(1,2,2)

_=sns.boxplot(data['WS_HR'])
data['WS_HR'].fillna(value=data['WS_HR'].median(),inplace=True)
for col in ['NO2','NO','NOx','PM10','CO','O3','AMB_TEMP','SO2','WD_HR','RH','WIND_DIREC', 'WIND_SPEED','PM2.5']:

    data[col]=data[col].apply(numeric)

    data[col].fillna(value=data[col].median(),inplace=True)

data['RAINFALL'] = data['RAINFALL'].apply(lambda x:0 if x=='NR' else x).apply(numeric)
print((data['CH4'].notna() & data['NMHC'].notna() & data['THC'].isna()).value_counts())

print('\n')

print(data['THC'].apply(numeric).describe())

print('\nMode :',data['THC'].apply(numeric).mode())

print('\nKurtosis is ',data['THC'].apply(numeric).kurt(),'>3, it is leptokurtic')
plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.plot(data['THC'].apply(numeric).value_counts().sort_index(),label='distribution')

plt.plot([data['THC'].apply(numeric).mean()]*2,[0,17500],'g:',label='mean   ={0:.2f}'.format(data['THC'].apply(numeric).mean()))

plt.plot([data['THC'].apply(numeric).median()]*2,[0,17500],'r:',label='median={0:.2f}'.format(data['THC'].apply(numeric).median()))

plt.plot([data['THC'].apply(numeric).mode()[0]]*2,[0,17500],'y:',label='mode   ={0:.2f}'.format(data['THC'].apply(numeric).mode()[0]))

plt.legend()



plt.subplot(1,2,2)

sns.boxplot(data['THC'].apply(numeric).value_counts().sort_index())
data.drop(['CH4','NMHC','THC'],axis=1,inplace=True)
data['year'] = pd.to_datetime(data['time']).dt.year

data['month'] = pd.to_datetime(data['time']).dt.month

data['day'] = pd.to_datetime(data['time']).dt.day

data['hour'] = pd.to_datetime(data['time']).dt.hour

# data.drop('time',axis=1,inplace=True)
plt.figure(figsize=(20,5))

plt.subplot(1,3,1)

plt.title('AVG. PM2.5 in each month')

plt.plot(data.groupby('month').mean()['PM2.5'])

plt.xlim(1,12)

plt.xlabel('month')

plt.ylabel('PM2.5')



plt.subplot(1,3,2)

plt.title('AVG. PM2.5 in each hour')

plt.plot(data.groupby('hour').mean()['PM2.5'])

plt.xlim(0,23)

plt.xlabel('hour')

plt.ylabel('PM2.5')



plt.subplot(1,3,3)

plt.title('AVG. PM2.5 in each station')

plt.bar(data.groupby('station').mean().index,data.groupby('station').mean()['PM2.5'])

plt.xticks(rotation='vertical')

plt.xlabel('station')

_=plt.ylabel('PM2.5')
continous_columns=['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL','RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']

discrete_columns=['station','year','month','day','hour']
plt.figure(figsize=(20,60))

for number,col in enumerate(continous_columns):

    plt.subplot(15,2,number*2+1)

    sns.distplot(data[col],fit=norm)

    plt.subplot(15,2,number*2+2)

    res = stats.probplot(data[col],plot=plt)
plt.figure(figsize=(20,20))

for number,col in enumerate(['station','month','day','hour']):

    plt.subplot(4,2,number*2+1)

    plt.xticks(rotation='vertical')

    sns.boxplot(data[col],data['PM2.5'])

    plt.subplot(4,2,number*2+2)

    plt.title('With log')

    plt.xticks(rotation='vertical')

    sns.boxplot(data[col],np.log(data['PM2.5']))
# # This will take long time to plot

# plt.figure(figsize=(50,50))

# sns.set(style='darkgrid')

# fig=sns.pairplot(data[continous_columns+['station']],hue='station')
plt.figure(figsize=(20,20))

hm=sns.heatmap(data[continous_columns].corr().values,annot=True,

               xticklabels=continous_columns,

               yticklabels=continous_columns,

               cmap='Reds')
abs(data[continous_columns].corr()['PM2.5']).sort_values(ascending=False) # absolute number
data_Banqiao = data[data['station']=='Banqiao'].sort_values(by=['year','month','day','hour'])
for col in ['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10','RAINFALL', 'RH', 'SO2', 'WD_HR']:

    stdscl = StandardScaler()

    data_Banqiao[col] = stdscl.fit_transform(data_Banqiao[col].values.reshape(-1,1))
data_Banqiao['next_hour_pm2.5'] = data_Banqiao['PM2.5'].shift(-1)

data_Banqiao.drop(8759,inplace=True)

y_train = data_Banqiao[data_Banqiao['month'] != 12].loc[:,'next_hour_pm2.5'].values

x_train = data_Banqiao[data_Banqiao['month'] != 12].loc[:,'AMB_TEMP':'hour'].values

y_test = data_Banqiao[data_Banqiao['month'] == 12].loc[:,'next_hour_pm2.5'].values

x_test = data_Banqiao[data_Banqiao['month'] == 12].loc[:,'AMB_TEMP':'hour'].values
# Need to check if it's suitable, because it will interrupt the continuity of time, e.g. is it proper to use time T+1 and Time T-1 to predirct Time T? Maybe not, but grid search is doing so. Therefore, I think maybe it's not fair to compare this RF to LSTM.

g_search = GridSearchCV(RandomForestRegressor(n_jobs=-1),param_grid={'n_estimators':[5,10,20],'max_features':[5,10],'max_depth':[10,20,30]},scoring='neg_mean_squared_error',n_jobs=-1,cv=2)

g_search.fit(x_train,y_train)

y_rf_pred = g_search.best_estimator_.predict(x_test)

pd.DataFrame(g_search.cv_results_)
plt.figure(figsize=(20,5))

plt.plot(y_rf_pred,label='prediction')

plt.plot(y_test,label='real data')

plt.xlim(0,737)

plt.ylim(0,)

_ =plt.legend(loc='upper left')

print('MSE',mean_squared_error(y_test, y_rf_pred))
device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

  raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))

import timeit



# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth

config = tf.ConfigProto()

config.gpu_options.allow_growth = True



with tf.device('/cpu:0'):

  random_image_cpu = tf.random_normal((100, 100, 100, 3))

  net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)

  net_cpu = tf.reduce_sum(net_cpu)



with tf.device('/gpu:0'):

  random_image_gpu = tf.random_normal((100, 100, 100, 3))

  net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)

  net_gpu = tf.reduce_sum(net_gpu)



with tf.Session(config=config) as sess:

    tf.global_variables_initializer().run()

    def cpu():

      sess.run(net_cpu)

    def gpu():

      sess.run(net_gpu)

    

    print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '

          '(batch x height x width x channel). Sum of ten runs.')

    print('CPU (s):')

    cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")

    print(cpu_time)

    print('GPU (s):')

    gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")

    print(gpu_time)

    print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))
n_steps = 1

n_inputs = 19

n_outputs = 1

n_units = [100, 100,100]

learning_rate= 0.5 #Adam perform learning rate decay 

n_epochs = 100

n_iterations = n_epochs * len(x_train)

with tf.device('/gpu:0'):

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])

    y = tf.placeholder(tf.float32,[None,n_steps,n_outputs])



    cells = [tf.contrib.rnn.LSTMCell(num_units=n) for n in n_units]

    cells = tf.contrib.rnn.MultiRNNCell(cells)

    cells = tf.contrib.rnn.DropoutWrapper(cells,0.5)

    cells = tf.contrib.rnn.OutputProjectionWrapper(cells, output_size=n_outputs)

    outputs, states = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)

    loss = tf.reduce_mean(tf.square(outputs - y))

    training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
from IPython.display import clear_output, Image, display, HTML



def strip_consts(graph_def, max_const_size=32):

    """Strip large constant values from graph_def."""

    strip_def = tf.GraphDef()

    for n0 in graph_def.node:

        n = strip_def.node.add() 

        n.MergeFrom(n0)

        if n.op == 'Const':

            tensor = n.attr['value'].tensor

            size = len(tensor.tensor_content)

            if size > max_const_size:

                tensor.tensor_content = "<stripped %d bytes>"%size

    return strip_def



def show_graph(graph_def, max_const_size=32):

    """Visualize TensorFlow graph."""

    if hasattr(graph_def, 'as_graph_def'):

        graph_def = graph_def.as_graph_def()

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)

    code = """

        <script>

          function load() {{

            document.getElementById("{id}").pbtxt = {data};

          }}

        </script>

        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>

        <div style="height:600px">

          <tf-graph-basic id="{id}"></tf-graph-basic>

        </div>

    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))



    iframe = """

        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>

    """.format(code.replace('"', '&quot;'))

    display(HTML(iframe))

show_graph(tf.get_default_graph())
config = tf.ConfigProto()

config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    tf.global_variables_initializer().run()

    for iteration in range(n_iterations):

        index = iteration%len(x_train)

        x_batch = x_train[index,:].reshape(-1,n_steps,n_inputs)

        y_batch = y_train[index].reshape(-1,n_steps,n_outputs)

        sess.run(training_op, feed_dict={x:x_batch,y:y_batch})

        output_val=outputs.eval(feed_dict={x:x_batch,y:y_batch})

        if iteration % len(x_train)==0:

            loss_val = loss.eval(feed_dict={x:x_batch,y:y_batch})

            if loss_val <= 300: #early break

                break

    y_lstm_pred = outputs.eval(feed_dict={x:x_test.reshape(-1,n_steps,n_inputs)})

    y_lstm_pred = [i.reshape(1)[0] for i in y_lstm_pred]
plt.figure(figsize=(20,5))

plt.plot(y_lstm_pred,label='LSTM',)

plt.plot(y_rf_pred.reshape(-1),label='RandomForest')

plt.plot(y_test,label='Real data')

plt.xlabel('Hour start from 2015/12/1 0:00  to  2015/12/31 22:00 ')

plt.ylabel('PM2.5')

plt.xlim(0,737)

plt.ylim(0,)

_=plt.legend(loc='upper left')

print('Random Forest MSE',mean_squared_error(y_test, y_rf_pred))

print('LSTM MSE:',mean_squared_error(y_test, y_lstm_pred))
plt.figure(figsize=(100,50))

for i in range(1,13):

    y_train = data_Banqiao[data_Banqiao['month'] != i].loc[:,'next_hour_pm2.5'].values

    x_train = data_Banqiao[data_Banqiao['month'] != i].loc[:,'AMB_TEMP':'hour'].values

    y_test = data_Banqiao[data_Banqiao['month'] == i].loc[:,'next_hour_pm2.5'].values

    x_test = data_Banqiao[data_Banqiao['month'] == i].loc[:,'AMB_TEMP':'hour'].values

    est = RandomForestRegressor(max_depth=10, max_features=10, n_estimators=20,n_jobs=-1)    

    y_rf_pred = est.fit(x_train,y_train).predict(x_test)

    plt.subplot(4,3,i)

    plt.title(f'Month {i}')

    plt.plot(y_rf_pred,label='prediction')

    plt.plot(y_test,label='real data')

    plt.plot(lable=mean_squared_error(y_test, y_rf_pred))

    plt.xlim(0,)

    plt.ylim(0,)

    _ =plt.legend(loc='upper left')

    annotation_location = max(max(y_test), max(y_rf_pred))*0.7

    plt.annotate('Month {0} MSE: {1:.2f}'.format(i, mean_squared_error(y_test, y_rf_pred)), 

             xy=(10, annotation_location), xytext=(10, annotation_location))
# https://e-service.cwb.gov.tw/wdps/obs/state.htm 

Banqiao = {'name':'Banqiao','longitude':121.4420,'latitude':24.9976}

Dayuan = {'name':'Dayuan','longitude':121.2260,'latitude':25.0478}

Guanyin = {'name':'Guanyin','longitude':121.1533,'latitude':25.0271}

Keelung = {'name':'Keelung','longitude':121.7405,'latitude':25.1333}

Linkou = {'name':'Linkou','longitude':121.3808,'latitude':25.0723}

Longtan = {'name':'Longtan','longitude':121.2214,'latitude':24.8701}

Pingzhen = {'name':'Pingzhen','longitude':121.2146,'latitude':24.8975}

Shilin = {'name':'Shilin','longitude':121.5030,'latitude':25.0903}

Songshan = {'name':'Songshan','longitude':121.5504,'latitude':25.0487}

Taoyuan = {'name':'Taoyuan','longitude':121.3232,'latitude':24.9924}

Tucheng = {'name':'Tucheng','longitude':121.4452,'latitude':24.9732}

Xinzhuang = {'name':'Xinzhuang','longitude':121.4468,'latitude':25.0515}

Xizhi = {'name':'Xizhi','longitude':121.6588,'latitude':25.0669}

Yonghe = {'name':'Yonghe','longitude':121.5081,'latitude':25.0113}

Zhongli = {'name':'Zhongli','longitude':121.2564,'latitude':24.9777}

all_stations= [Banqiao,Dayuan,Guanyin,Keelung,Linkou,Longtan,Pingzhen,Shilin,Songshan,Taoyuan,Tucheng,Xinzhuang,Xizhi,Zhongli]
import folium
m = folium.Map(location=[25.0, 121.5],zoom_start=10)

for station in all_stations:

    name = station['name']

    lat = station['latitude']

    lon = station['longitude']

    folium.Marker([lat,lon],popup=name).add_to(m)

print('Welcome to Taiwan!')

m
plt.figure(figsize=(20,5))

plt.title('Simpler map')

plt.xlabel('longitude')

plt.ylabel('latitude')

for station in all_stations:

    plt.scatter(station['longitude'],station['latitude'])

    plt.annotate(s=station['name'], xy=(station['longitude'],station['latitude']))
def all_stations_plot(month,day):

    plt.figure(figsize=(50,25))

    plt.title('month {} day {}'.format(month,day))

    for station in data['station'].unique():

        plt.plot(data[(data['station']==station)&(data['month'] == month)&(data['day'] == day)]['PM2.5'].values,label=station)

        plt.legend(loc='upper right',prop={'size': 20})

    plt.xticks(np.arange(0,25,1))

    _=plt.xlim(0,23)

    plt.show()

for i in range(1,32):

    all_stations_plot(1,i)