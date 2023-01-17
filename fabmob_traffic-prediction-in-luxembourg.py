import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import dates

import matplotlib.pyplot as plt

import networkx as nx

from datetime import datetime

from tensorflow import keras

import math as math

from sklearn.metrics import mean_squared_error

plt.style.use('seaborn')



        

lux = (49.61167 , 6.13)

highway = ['A1', 'A3', 'A4', 'A6', 'A7', 'A13', 'B40']        

directory = '/kaggle/input/motorway-traffic-in-luxembourg/'

files=['datexDataA1.csv','datexDataA3.csv','datexDataA4.csv','datexDataA6.csv','datexDataA7.csv','datexDataA13.csv','datexDataB40.csv']
# Describe Data

def describeData(data):

    print("shape = {}".format(data.shape))

    description = data.describe().T

    description["isNull"] = data.isnull().sum()

    print(description)

# root mean squared error or rmse

def measure_rmse(actual, predicted):

    return math.sqrt(mean_squared_error(actual, predicted))



# filter data     

def getFilteredData(data,idCamera,indexMin='2000-01-01 00:00:00+0000',indexMax='2030-01-01 00:00:00+0000'):

    result = data[(data.index>indexMin) ]

    result = result[(result.index<indexMax)]

    result = result[(result['id'] == idCamera)].fillna(method = 'ffill')

    return result



def getFilteredDataByHighway(data,road,direction='outboundFromTown',indexMin='2000-01-01 00:00:00+0000',indexMax='2030-01-01 00:00:00+0000'):

    result = data[(data.index>indexMin) ]

    result = result[(result.index<indexMax)]

    result = result[(result['direction'] == direction)]

    result = result[(result['road'] == road)].fillna(method = 'ffill')

    return result



# Get Previous Camera Dict

def getPreviousCamDict(camera):

    prev_cam = {}

    for road in camera['road'].unique():

        fromLux = camera.loc[(camera['road']==road)&(camera['direction']=='outboundFromTown')].copy()

        toLux = camera.loc[(camera['road']==road)&(camera['direction']=='inboundTowardsTown')].copy()

        fromLux.sort_values(by=['direction_dist'],inplace=True,ascending = True)

        toLux.sort_values(by=['direction_dist'],inplace=True,ascending = False)

        

        toLux['previd']=toLux['id'].shift(1)

        toLux.dropna(inplace=True)

        fromLux['previd']=fromLux['id'].shift(1)

        fromLux.dropna(inplace=True)

        

        for lux in [fromLux,toLux]:

            if lux.shape[0]>0:

                for index, row in lux.iterrows():

                    prev_cam[row['id']] = row['previd']  

    return prev_cam    

# Graph

def getGraph(cameras,direction = "outboundFromTown"):

    G = nx.DiGraph(label="TRAFFIC")

    dict_previous_cam = getPreviousCamDict(cameras)

    for i, rowi in cameras.loc[cameras["direction"]==direction].iterrows():

        G.add_node(rowi['id'],key=rowi['id'],label="id",road=rowi['road'],latitude=rowi['latitude'],longitude=rowi['longitude'])

        previous = dict_previous_cam.get(rowi['id'])

        if previous != None:

            G.add_edge(previous,rowi['id'])            

    return G



def printMap(G,title=""):

    colors=[]

    pos={}

    pos2={}

    labels={}

    dict_colors = {'A1':'orange', 'A3':'blue', 'A4':'red', 'A6':'green', 'A7':'black', 'A13':'yellow', 'B40':'cyan'}    

    fig, ax = plt.subplots(figsize=(12,12), dpi=200)  

    # graph    

    for e in G:

        pos[e]=(G.nodes[e]['longitude'],G.nodes[e]['latitude'])

        pos2[e]=(G.nodes[e]['longitude']+0.010,G.nodes[e]['latitude']+0.001)

        new = e.split(".")

        labels[e] = new[2]

        colors.append(dict_colors.get(G.nodes[e]['road']))

    # nodes

    nx.draw_networkx_nodes(G, pos,node_size=50,node_color=colors)

    nx.draw_networkx_edges(G, pos, node_size=50, arrowstyle='->',arrowsize=20)

    # labels

    nx.draw_networkx_labels(G, pos2, labels= labels,font_size=10)

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in dict_colors.values()]

    plt.legend(markers, dict_colors.keys(), numpoints=1,loc='upper left') 

    plt.title(title)

    plt.show()

    

def plot_long_serie(data,title='',label='',xlabel='Time',ylabel='', dpi=100):

    days = dates.DayLocator()

    dfmt = dates.DateFormatter('%b %d')

    fig, ax = plt.subplots(figsize=(16,9), dpi=dpi)  

    ax.set_title(title)

    ax.plot(data.index.values, data.values, label=label,linewidth=1)

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    ax.legend()

    ax.xaxis.set_major_locator(days)

    ax.xaxis.set_major_formatter(dfmt)

    ax.xaxis.set_tick_params(which='major',labelsize=7)

    ax.grid(True)

    plt.show()



def plot_day_traffic(data0,data1,titles,dpi=100):    

    fig, axes = plt.subplots(2, 7, figsize=(16,9),dpi=dpi)    

    yLimite=data0.max()*1.1

    for i, a in zip(range(7), axes[0].ravel()):

        df = data0[data0.index.dayofweek==i]

        df.index = [df.index.time,df.index.date]

        df = df.unstack().interpolate()

        for column in df:

            a.plot(df.index.map(lambda x: (x.minute+x.hour*60)/60 ), df[column], marker='', linewidth=1, alpha=0.9) 

            a.set_ylim(0,yLimite)

            a.set_title(titles[i])

            a.set_xticks(np.arange(0,25,3))

    yLimite=data1.max()*1.1             

    for i, a in zip(range(7), axes[1].ravel()):

        df = data1[data1.index.dayofweek==i]

        df.index = [df.index.time,df.index.date]

        df = df.unstack().interpolate()

        for column in df:

            a.plot(df.index.map(lambda x: (x.minute+x.hour*60)/60 ), df[column], marker='', linewidth=1, alpha=0.9)                     

            a.set_ylim(0,yLimite)

            a.set_xticks(np.arange(0,25,3))

    plt.show()

    



# Draw multiple plots 

def plot_mult(datas,labels,xlabel,ylabel,dpi=100):

    days = dates.DayLocator()

    hours = dates.HourLocator(byhour=[0,6,12,18])

    dfmt = dates.DateFormatter('%b %d')

    fig, ax = plt.subplots(figsize=(16,9), dpi=dpi)  

    

    for i, data in enumerate(datas):

        plabel = labels[i]

        ax.plot(data.index.values, data.values, label=plabel,linewidth=1)

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    ax.legend()

    ax.xaxis.set_major_locator(days)

    ax.xaxis.set_major_formatter(dfmt)

    ax.xaxis.set_minor_locator(hours)

    ax.xaxis.set_minor_formatter(dates.DateFormatter('%H'))

    ax.xaxis.set_tick_params(which='major', pad=15,labelsize=7)

    ax.xaxis.set_tick_params(which='minor', labelsize=4)

    ax.grid(True)

    plt.show()

frames=[]

for file in files:

    #print(file + ' start')

    df = pd.read_csv(directory+file, parse_dates = False, header = None,sep=';')

    df.columns=['id','time','latitude','longitude','direction','road','trafficStatus','avgVehicleSpeed','vehicleFlowRate','trafficConcentration']

    df.loc[df['time'].str.len()<25,'time']=pd.NaT

    df['time'] = pd.to_datetime(df['time'],errors='coerce',utc=False)  

    df['time'] = df['time'].fillna(method = 'ffill')

    #df.index = df['time','id']

    df.set_index(['time','id'],drop=False,inplace=True)

    df=df[~df.index.duplicated()]

    df.set_index(['time'],drop=False,inplace=True)

    df['dayofweek'] = df.index.dayofweek

    df['day'] = df.index.day

    df['hour'] = df.index.hour

    new = df["id"].str.split(".", expand = True) 

    df['highway']=new[0]

    df['direction_code']=new[1]

    df['direction_dist']=pd.to_numeric(new[2])    

    # df=df[~camera.index.duplicated()]

    frames.append(df)

    print(file + ': end loading {} rows '.format(df.shape[0]))

data = pd.concat(frames)



data.dropna(inplace=True)

# Build camera dataframe

camera = data[['id','latitude','longitude','direction_dist','direction_code','road','direction']]

camera.index = camera['id']

camera=camera[~camera.index.duplicated()]

del(new,frames,file,df)

data.head(5)
print("min index is",data.index.min())

print("max index is",data.index.max())

nRow, nCol = data.shape

print(f'There are {nRow} rows and {nCol} columns in the dataset')
nRow, nCol = camera.shape

print(f'There are {nRow} rows and {nCol} columns in camera dataset')

camera.head(5)
# print the cameras map

GFrom = getGraph(camera,direction="outboundFromTown")

GTo = getGraph(camera,direction="inboundTowardsTown")



printMap(GTo,title="Cameras - direction = to Luxembourg")
printMap(GFrom,title="Cameras - direction = From Luxembourg")
road = 'A3'

  

fromLuxCam = 'A3.VM.11397'

toLuxCam = 'A3.MV.11397'

fromDate = '2019-11-25 00:00:00+0000'

toDate = '2019-12-23 00:00:00+0000'



fromLux = getFilteredData(data,fromLuxCam,fromDate,toDate)

toLux = getFilteredData(data,toLuxCam,fromDate,toDate)



fromLux_rolling_avg = fromLux.avgVehicleSpeed.rolling(window=10,center=True).mean()

fromLux_rolling_flow = fromLux.vehicleFlowRate.rolling(window=10,center=True).mean()

fromLux_rolling_traf = fromLux.trafficConcentration.rolling(window=10,center=True).mean()

fromLux_rolling_avg.dropna(inplace=True)

fromLux_rolling_flow.dropna(inplace=True)

fromLux_rolling_traf.dropna(inplace=True)



toLux_rolling_avg = toLux.avgVehicleSpeed.rolling(window=10,center=True).mean()

toLux_rolling_flow = toLux.vehicleFlowRate.rolling(window=10,center=True).mean()

toLux_rolling_traf = toLux.trafficConcentration.rolling(window=10,center=True).mean()

toLux_rolling_avg.dropna(inplace=True)

toLux_rolling_flow.dropna(inplace=True)

toLux_rolling_traf.dropna(inplace=True)
plot_long_serie(fromLux_rolling_avg,title='From LUX camera={}'.format(fromLuxCam),label='Average Vehicle Speed',xlabel='Date',ylabel='Speed')
plot_long_serie(fromLux_rolling_flow,title='From LUX camera={}'.format(fromLuxCam),label='Vehicle Flow Rate',xlabel='Date',ylabel='Rate')
plot_long_serie(fromLux_rolling_traf,title='From LUX camera={}'.format(fromLuxCam),label='Traffic Concentration',xlabel='Date',ylabel='Concentration')

plot_long_serie(toLux_rolling_avg,title='To LUX camera={}'.format(toLuxCam),label='Average Vehicle Speed',xlabel='Date',ylabel='Speed')
plot_long_serie(toLux_rolling_flow,title='To LUX camera={}'.format(toLuxCam),label='Vehicle Flow Rate',xlabel='Date',ylabel='Rate')
plot_long_serie(toLux_rolling_traf,title='To LUX camera={}'.format(toLuxCam),label='Traffic Concentration',xlabel='Date',ylabel='Concentration')
titles=['Monday','Tueday','Wednesday','Thursday','Friday','Saturday','Sunday']

plot_day_traffic(toLux_rolling_avg,fromLux_rolling_avg,titles)
plot_day_traffic(toLux_rolling_flow,fromLux_rolling_flow,titles)
plot_day_traffic(toLux_rolling_traf,fromLux_rolling_traf,titles)
for direction in ['outboundFromTown','inboundTowardsTown']:

    df = getFilteredDataByHighway(data,road,direction,fromDate,toDate)

    cams = camera.loc[(camera['road']==road)&(camera['direction']==direction)].sort_values(by=['direction_dist'])['id'].values

    datas=[]

    datas2=[]

    datas3=[]

    labels=[]

    for cam in cams:    

        temp=df.loc[(df['id']==cam)&(df.index.dayofweek.isin([0,1,2,3,4]))].copy()

        temp['minute']=temp.index.time

        temp.avgVehicleSpeed = temp.avgVehicleSpeed.rolling(window=3,center=True).mean()

        temp.vehicleFlowRate = temp.vehicleFlowRate.rolling(window=3,center=True).mean()

        temp.trafficConcentration = temp.trafficConcentration.rolling(window=3,center=True).mean()

        temp.dropna(inplace=True)

        data1 = temp.groupby('minute').mean()["avgVehicleSpeed"].copy()

        data1.index = data1.index.map(lambda x : x.strftime("%H:%M:%S"))

        datas.append(data1)

        data2 = temp.groupby('minute').mean()["vehicleFlowRate"].copy()

        data2.index = data2.index.map(lambda x : x.strftime("%H:%M:%S"))

        datas2.append(data2)

        data3 = temp.groupby('minute').mean()["trafficConcentration"].copy()

        data3.index = data3.index.map(lambda x : x.strftime("%H:%M:%S"))

        datas3.append(data3)

        labels.append(cam)

        

    plot_mult(datas,labels,'Date','avgVehicleSpeed')    

    plot_mult(datas2,labels,'Date','vehicleFlowRate')     

    plot_mult(datas3,labels,'Date','vehicleFlowRate') 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score,SCORERS

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from timeit import default_timer as timer

from sklearn.preprocessing import MinMaxScaler
## Generate X and y

######################################################

def generateDf(dataIn,cam,cam1):

    df0 = getFilteredData(dataIn,cam)

    df1 = getFilteredData(dataIn,cam1)

    df1 = df1[['avgVehicleSpeed', 'vehicleFlowRate']]

    col_rename = {}

    for col in df1.columns:

        col_rename[col]='prev_station_' + col

    

    df1.rename(columns=col_rename,inplace=True)

    df = df0.join(df1)

    df=df[['avgVehicleSpeed', 'vehicleFlowRate','trafficConcentration','dayofweek','hour','prev_station_avgVehicleSpeed', 'prev_station_vehicleFlowRate']].copy()

    df['isWeekend'] = df['dayofweek'].map(lambda x : 0 if x < 5 else 1)



    # Diff %

    for i in range(1,backward+1):

         df['avgDiff'+str(i)] = df['avgVehicleSpeed'].shift(i-1)/ df['avgVehicleSpeed'].shift(i) - 1

         df['avgDiff'+str(i)].replace([np.inf, -np.inf], np.nan,inplace=True)

         df['avgDiff'+str(i)].fillna(method='bfill')

         df['flowDiff'+str(i)] = df['vehicleFlowRate'].shift(i-1)/ df['vehicleFlowRate'].shift(i) - 1

         df['flowDiff'+str(i)].replace([np.inf, -np.inf], np.nan,inplace=True)

         df['flowDiff'+str(i)].fillna(method='bfill')

         df['flowTraffic'+str(i)] = df['trafficConcentration'].shift(i-1)/ df['trafficConcentration'].shift(i) - 1

         df['flowTraffic'+str(i)].replace([np.inf, -np.inf], np.nan,inplace=True)

         df['flowTraffic'+str(i)].fillna(method='bfill')

         

    # EWL

    df['EWMavg']=df['avgVehicleSpeed'].ewm(span=3, adjust=False).mean()

    df['EWMflow']=df['vehicleFlowRate'].ewm(span=3, adjust=False).mean()

    df['EWMtraffic']=df['trafficConcentration'].ewm(span=3, adjust=False).mean()

    return df



def generateXYspeed20(df):    

    df['ydiff'] = df['avgVehicleSpeed'].shift(forward)/df['avgVehicleSpeed'] - 1    

    df['y'] = 0

    df.loc[df['ydiff']<-0.2,['y']]=1

    df.dropna(inplace=True)

    y = df['y']

    X = df.drop(['y','ydiff'], axis=1)

    return X , y



def generateXYspeedUnder(df):    

    mean = df['avgVehicleSpeed'].mean()

    df['ydiff'] = df['avgVehicleSpeed'].shift(forward)

    df['y'] = 0

    df.loc[df['ydiff']<mean*0.6,['y']]=1

    df.dropna(inplace=True)

    y = df['y']

    X = df.drop(['y','ydiff'], axis=1)

    return X , y



def generateXYspeedAndFlowUnder(df):    

    means = df['avgVehicleSpeed'].mean()

    meanf = df['vehicleFlowRate'].mean()

    df['ydiffSpeed'] = df['avgVehicleSpeed'].shift(forward)

    df['ydiffFlow'] = df['vehicleFlowRate'].shift(forward)

    df['y'] = 0

    df.loc[(df['ydiffSpeed']<means*0.6) &(df['ydiffFlow']<meanf*0.6),['y']]=1

    df.dropna(inplace=True)

    y = df['y']

    X = df.drop(['y','ydiffSpeed','ydiffFlow'], axis=1)

    return X , y



def print_metrics(y_true,y_pred):

    conf_mx = confusion_matrix(y_true,y_pred)

    print(conf_mx)

    print (" Accuracy    : ", accuracy_score(y_true,y_pred))

    print (" Precision   : ", precision_score(y_true,y_pred))

    print (" Sensitivity : ", recall_score(y_true,y_pred))





def train_model(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    start = timer()

    forest = RandomForestClassifier(max_depth = 10, n_estimators = 500, random_state = 42)

    random_forest = forest.fit(X_train,y_train)

    end = timer()

    

    y_pred = random_forest.predict(X_train)

    print ("------------------------------------------")

    print ("TRAIN")

    print_metrics(y_train,y_pred)

    importances = list(zip(random_forest.feature_importances_, X.columns))

    importances.sort(reverse=True)

    print([x for (_,x) in importances[0:5]])

    y_pred = random_forest.predict(X_test)

    print ("------------------------------------------")

    print ("TEST")

    print_metrics(y_test,y_pred)

    

    return random_forest 
cam = 'A3.MV.10437'

cam1= 'A3.MV.11397'   



forward = -3

backward = 3    

df = generateDf(data,cam,cam1)





print ('camera :',cam)

print ("---------------------------------------------------------------")

print ("Predict 20% speed drop")

X,y = generateXYspeed20(df)

model = train_model(X,y)



print ("---------------------------------------------------------------")

print ("Predict speed less than 60% of the average speed")

X,y = generateXYspeedUnder(df)

model = train_model(X,y)



print ("---------------------------------------------------------------")

print ("Predict speed anf flow less than 60% of the average")

X,y = generateXYspeedAndFlowUnder(df)

model = train_model(X,y)

def train_model_and_get_metrics(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    start = timer()

    forest = RandomForestClassifier(max_depth = 10, n_estimators = 500, random_state = 42)

    random_forest = forest.fit(X_train,y_train)

    end = timer()

    y_pred = random_forest.predict(X_test)

    #print (" Accuracy    : ", accuracy_score(y_true,y_pred))

    #print (" Precision   : ", precision_score(y_true,y_pred))

    #print (" Sensitivity : ", recall_score(y_true,y_pred))    

    return [accuracy_score(y_test,y_pred), precision_score(y_test,y_pred),recall_score(y_test,y_pred)]
cams = camera.loc[(camera['road']=='A3')&(camera['direction']=='inboundTowardsTown')].sort_values(by=['direction_dist'],ascending=False)['id'].values

prev_cam_dict = getPreviousCamDict(camera)

speed20=[]

speedUnder=[]

speedAndFlowUnder=[]

for cam in (cams[1:]):

    cam1 = prev_cam_dict.get(cam)

    print ('camera :',cam)

    df = generateDf(data,cam,cam1)

    X,y = generateXYspeed20(df)

    speed20.append(train_model_and_get_metrics(X,y))

    X,y = generateXYspeedUnder(df)

    speedUnder.append(train_model_and_get_metrics(X,y))

    X,y = generateXYspeedAndFlowUnder(df)

    speedAndFlowUnder.append(train_model_and_get_metrics(X,y))

speed20df = pd.DataFrame(speed20, columns = ['Accurancy', 'precision','recall'])

speedUnderdf = pd.DataFrame(speedUnder, columns = ['Accurancy', 'precision','recall'])

speedAndFlowUnderdf = pd.DataFrame(speedAndFlowUnder, columns = ['Accurancy', 'precision','recall'])
for col in ['Accurancy', 'precision','recall']:

    plt.plot(speed20df[col], label='speed20')    

    plt.plot(speedUnderdf[col], label='speedUnder')

    plt.plot(speedAndFlowUnderdf[col], label='speedAndFlowUnder')

    plt.title(col)

    plt.legend()

    plt.show()
def getSequences(sequence, backward, forward=1):

    X, y = list(), list()

    for i in range(len(sequence)-(backward+forward-1)):

        if forward > 1:

            seq_x, seq_y = sequence[i:i+backward], sequence[i+backward:i+backward+forward]

        else:

            seq_x, seq_y = sequence[i:i+backward], sequence[i+backward]

        X.append(seq_x)

        y.append(seq_y)

    

    return np.array(X), np.array(y)
# Parameters

pindexMin = '2000-01-01 00:00:00+0000'

pindexMax = '2030-01-01 00:00:00+0000'



cam = 'A3.VM.8246' 

cam1 = 'A3.VM.7280'



WINDOW = 10

FORECAST = 3
# Build features dataframe

df0 = getFilteredData(data,cam,indexMin=pindexMin,indexMax=pindexMax)

df0=df0[['avgVehicleSpeed', 'vehicleFlowRate']]

df1 = getFilteredData(data,cam1,indexMin=pindexMin,indexMax=pindexMax)

df1=df1[['avgVehicleSpeed', 'vehicleFlowRate']]

    

df1.rename(columns={'avgVehicleSpeed' : 'pre_avgVehicleSpeed', 'vehicleFlowRate' : 'pre_vehicleFlowRate'},inplace=True)

df = df0.join(df1,how='inner')

df.dropna(inplace=True)



scaler = MinMaxScaler()

scaled = scaler.fit_transform(df)



X, _ = getSequences(scaled, backward = WINDOW , forward= FORECAST )

_ , ySpeed = getSequences(scaled[:,0], backward = WINDOW , forward= FORECAST )

_ , yFlow = getSequences(scaled[:,1], backward = WINDOW , forward= FORECAST )



print("X shape", X.shape," - y shape ", y.shape)



X_train = X[1000:]

X_test = X[:1000]

ySpeed_train = ySpeed[1000:]

ySpeed_test = ySpeed[:1000]

yFlow_train = yFlow[1000:]

yFlow_test = yFlow[:1000]
# Train the model

def trainTheModel(X_train,y_train):

    model = keras.Sequential()

    model.add(keras.layers.LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))

    #model.add(keras.layers.Dense(100, activation='relu'))

    model.add(keras.layers.Dense(FORECAST))

    model.compile(loss='mse', optimizer='adam')

    early_stop = keras.callbacks.EarlyStopping(

            monitor='val_loss',

            patience=10

    )   

    history = model.fit(

        X_train,y_train, 

        epochs=10, 

        batch_size=64, 

        validation_split=0.05,

        shuffle=True,

        callbacks=[early_stop],

        verbose=0

        )            



    plt.plot(history.history['loss'], label='train')

    plt.plot(history.history['val_loss'], label='test')

    plt.legend()

    plt.show()

    return model



modelSpeed = trainTheModel(X_train,ySpeed_train)

modelFlow = trainTheModel(X_train,yFlow_train)
# Evaluate the results with test data

ySpeed_pred = modelSpeed.predict(X_test)

yFlow_pred = modelFlow.predict(X_test)



for i in [0,1,2]:

    print("RMSE for Speed Prediction + {} minutes= {}".format(i*5+5,measure_rmse(ySpeed_test[:,i], (ySpeed_pred[:,i]))))



for i in [0,1,2]:

    print("RMSE for Flow Prediction + {} minutes= {}".format(i*5+5,measure_rmse(yFlow_test[:,i], (yFlow_pred[:,i]))))
iStart=100

iStop=200

for i in [0,1,2]:

    plt.plot(ySpeed_pred[:iStop,i], label='prediction')    

    plt.plot(ySpeed_test[:iStop,i], label='true')

    plt.title('Speed prediction + {} minutes'.format(i*5+5))

    plt.legend()

    plt.show()

for i in [0,1,2]:

    plt.plot(yFlow_pred[:iStop,i], label='prediction')    

    plt.plot(yFlow_test[:iStop,i], label='true')

    plt.title('Flow prediction + {} minutes'.format(i*5+5))

    plt.legend()

    plt.show()