# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential

import matplotlib.patches as mpatches

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.callbacks import ModelCheckpoint

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Import the library

import seaborn as sns

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import numpy as np

import time

import os

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

df = pd.read_csv('../input/testset.csv')

df.head()

data2 = df.copy()

# Any results you write to the current directory are saved as output.
df.head()
df= df.rename(index=str, columns={' _tempm': 'temprature'})

df= df[62304:]

df.head()

data2 = df.copy()
data2 = data2[['datetime_utc','temprature' ]]
data2['year'] = data2.datetime_utc.str.slice(stop =4)

data2.head()
data2 = data2.rename(index=str, columns={' _tempm': 'temprature'})
data2['datetime_utc'] = pd.to_datetime(data2['datetime_utc'])
data = []

colours = ['brown','grey', 'purple', 'black', 'pink', 'orange', 'yellow', 'green','blue', 'violet', 'red' ]

for (i,j) in zip(data2.year.unique(), colours):

    trace = go.Scatter(

        x = data2[data2.year == i].datetime_utc,

        y = data2[data2.year == i].temprature,

        mode = 'lines',

        name = i,

        marker=dict(

            color=j,

            size=6,

            symbol='circle',

            line=dict(

                color='rgb(254, 204, 204)',

                width=1

            ),

            opacity=0.9

        )

        )

    data.append(trace)

# data = [trace_4,trace_3,trace_2,trace_1,trace0,trace1,trace2,trace3,trace4,trace5,trace6,trace7]

# Define step as a list

steps = []



# We loop through the length of the dataset and try to define steps at each interval

for i in range(len(data)):

    # Defining step

    step = dict(

        # Using restyle method as we are changing underlying data

        method = 'restyle',

        label = 'black',

        # Setting all traces to invisible mode - visibility set to false

        args = ['visible', [False] * len(data)],

    )

    step['args'][1][i] = True # Toggle i'th trace to "visible"

    # Append step to the 'steps' list

    steps.append(step)



steps[0]['label'] = '2005'

steps[1]['label'] = '2006'

steps[2]['label'] = '2007'

steps[3]['label'] = '2008'

steps[4]['label'] = '2009'

steps[5]['label'] = '2010'

steps[6]['label'] = '2011'

steps[7]['label'] = '2012'

steps[8]['label'] = '2013'

steps[9]['label'] = '2014'

steps[10]['label'] = '2015'



# Defining the slider

sliders = [dict(

    active = 1,

    currentvalue = {"prefix": "Year:"},

    pad = {"t": 50},

    # Assigning steps of the slider

    steps = steps

)]



# Creating a layout with the slider defined above

layout = dict(title='Delhi Temperature', showlegend=True,

            xaxis=dict(rangeslider=dict(

            visible = True

        )))



# Creating a figure using data and layout

# fig = dict(data=data, layout=layout)

fig = go.Figure(data=data, layout=layout)

# Visualizing the plot

# Visualizing the plot

ofl.iplot(fig, filename='Sphere colour Slider')
# removing outliers

data2 = data2[data2.temprature < 50]
data = []

colours = ['brown','grey', 'purple', 'black', 'pink', 'orange', 'yellow', 'green','blue', 'violet', 'red' ]

for (i,j) in zip(data2.year.unique(), colours):

    trace = go.Scatter(

        x = data2[data2.year == i].datetime_utc,

        y = data2[data2.year == i].temprature,

        mode = 'lines',

        name = i,

        marker=dict(

            color=j,

            size=6,

            symbol='circle',

            line=dict(

                color='rgb(254, 204, 204)',

                width=1

            ),

            opacity=0.9

        )

        )

    data.append(trace)

# data = [trace_4,trace_3,trace_2,trace_1,trace0,trace1,trace2,trace3,trace4,trace5,trace6,trace7]

# Define step as a list

steps = []



# We loop through the length of the dataset and try to define steps at each interval

for i in range(len(data)):

    # Defining step

    step = dict(

        # Using restyle method as we are changing underlying data

        method = 'restyle',

        label = 'black',

        # Setting all traces to invisible mode - visibility set to false

        args = ['visible', [False] * len(data)],

    )

    step['args'][1][i] = True # Toggle i'th trace to "visible"

    # Append step to the 'steps' list

    steps.append(step)



steps[0]['label'] = '2005'

steps[1]['label'] = '2006'

steps[2]['label'] = '2007'

steps[3]['label'] = '2008'

steps[4]['label'] = '2009'

steps[5]['label'] = '2010'

steps[6]['label'] = '2011'

steps[7]['label'] = '2012'

steps[8]['label'] = '2013'

steps[9]['label'] = '2014'

steps[10]['label'] = '2015'



# Defining the slider

sliders = [dict(

    active = 1,

    currentvalue = {"prefix": "Year:"},

    pad = {"t": 50},

    # Assigning steps of the slider

    steps = steps

)]



# Creating a layout with the slider defined above

layout = dict(title='Delhi Temperature', showlegend=True,

            xaxis=dict(rangeslider=dict(

            visible = True

        )))



# Creating a figure using data and layout

# fig = dict(data=data, layout=layout)

fig = go.Figure(data=data, layout=layout)

# Visualizing the plot

# Visualizing the plot

ofl.iplot(fig, filename='Sphere colour Slider')
data2.set_index('datetime_utc', inplace= True)

data2.dropna(inplace=True)

data2.head()
data2.drop('year', axis =1, inplace = True)
data_array = data2.values

data_array = data_array.astype('float32')
scaler= MinMaxScaler(feature_range=(0,1))

sc = scaler.fit_transform(data2)
timestep = 48                                                                                                                                                                                                                                                                                                    

X= []

Y=[]
for i in range(len(sc)- (timestep)):

    X.append(sc[i:i+timestep])

    Y.append(sc[i+timestep])
X=np.asanyarray(X)

Y=np.asanyarray(Y)
k = 35000

Xtrain = X[:k,:,:]

Xtest = X[k:,:,:]    

Ytrain = Y[:k]    

Ytest= Y[k:]
print(Xtrain.shape)

print(Xtest.shape)

print(Ytrain.shape)

print(Ytest.shape)
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0],  Xtrain.shape[1],1))

Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1],1))
model = Sequential()

model.add(LSTM(128, batch_input_shape=(None,timestep,1),return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(32,return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(16,return_sequences=False))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer='adam')
history = model.fit(Xtrain,Ytrain,epochs=10,verbose=1,batch_size=32) 
# summarize history for loss

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()
preds = model.predict(Xtest)

preds = scaler.inverse_transform(preds)





Ytest = np.asanyarray(Ytest)  

Ytest = Ytest.reshape(-1,1) 

Ytest = scaler.inverse_transform(Ytest)





Ytrain=np.asanyarray(Ytrain)  

Ytrain=Ytrain.reshape(-1,1) 

Ytrain = scaler.inverse_transform(Ytrain)
y_index  = data1.index

Ytrain_index = y_index[:k]    

Ytest_index = y_index[k:-48]



preds = pd.DataFrame(preds)

pred= preds.rename(index=str, columns={0: 'Pred_Temp'})



Ytest = pd.DataFrame(Ytest)

Ytest= Ytest.rename(index=str, columns={0: 'Actual_Temp'})

pred_actual = pred.join(Ytest)

pred_actual.index = Ytest_index





mean_squared_error(Ytest,preds)
Actual_temp = go.Scatter(

    x = pred_actual.index,

    y = pred_actual['Actual_Temp'],

    mode = 'lines',

    name = 'Actual_Temp',

    marker = dict(

        size = 10,

        color =  np.random.randn(500),

        line = dict(

            width = 1,

        )

    )

)





Predicted_Temp = go.Scatter(

    x = pred_actual.index,

    y = pred_actual['Pred_Temp'],

    mode = 'lines',

    name = 'Pred_Temp',

    marker = dict(

        size = 10,

        color =  np.random.randn(500),

        line = dict(

            width = 1,

        ) 

    )

)



menu_var = [

    dict(active=1,

         buttons=list([   

            dict(label = 'Actual Temp',

                 method = 'update',

                 args = [{'visible': [True, False]},

                         {'title': 'Delhi Actual Temp', 'showlegend':False}]),



            dict(label = 'Pred Temp',

                 method = 'update',

                 args = [{'visible': [False, True]},

                         {'title': 'Delhi Predicted Temp', 'showlegend':True}])

        ]),

    )

]



layout= go.Layout(title='Delhi Temprature plot :: Comparision for actual and predicted values', xaxis={'title':'date'}, yaxis={'title':'Temprature'}, showlegend=False,

              updatemenus=menu_var)



figure=go.Figure(data=[Actual_temp,Predicted_Temp],layout=layout)

ofl.iplot(figure)