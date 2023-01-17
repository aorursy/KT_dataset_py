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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from shapely.geometry import Point, Polygon
virusData = pd.read_csv('../input/west-nile-virus-wnv-mosquito-test-results.csv')



print("Dimension of West Nile Virus data: {}".format(virusData.shape))
virusData.info()
sns.countplot(virusData['RESULT'], label="Count")

print("West Nile Virus Test Results:")

print(virusData.groupby('RESULT').size())
sns.catplot(x="RESULT", y="NUMBER OF MOSQUITOES", data=virusData, kind="boxen");
virusData.head()
species_data = pd.crosstab(virusData['SPECIES'], virusData['RESULT'])

print(species_data)



g = sns.countplot(x="SPECIES", hue="RESULT", data=virusData)

g.set_xticklabels(g.get_xticklabels(), rotation=90)
newData = virusData[virusData.RESULT != 'negative']

sns.countplot(x="WEEK", hue="RESULT", data=newData)
import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



virusData = virusData[virusData.RESULT == 'positive']

virus_data1 = virusData[virusData.SPECIES == 'CULEX PIPIENS/RESTUANS']

virus_data2 = virusData[virusData.SPECIES == 'CULEX RESTUANS']

virus_data3 = virusData[virusData.SPECIES == 'CULEX TERRITANS']

virus_data4 = virusData[virusData.SPECIES == 'CULEX PIPIENS']

virus_data5 = virusData[virusData.SPECIES == 'CULEX SALINARIUS']

mapbox_access_token = 'pk.eyJ1Ijoic2FuYXRuYXlhciIsImEiOiJjanphMmJjd2owNjhvM29rd3h0eGtnanE5In0.8vGE_l--loOQEEGwCtsDWQ'

fig = go.Figure([go.Scattermapbox(

        lat=virus_data4['LATITUDE'],

        lon=virus_data4['LONGITUDE'],

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=6, 

            color= "#00FFFF"

        ),

        text=virus_data4['SPECIES'],

        name='CULEX PIPIENS',

    ), go.Scattermapbox(

        lat=virus_data2['LATITUDE'],

        lon=virus_data2['LONGITUDE'],

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=6

        ),

        text=virus_data2['SPECIES'],

        name='CULEX RESTUANS',

    ), go.Scattermapbox(

        lat=virus_data1['LATITUDE'],

        lon=virus_data1['LONGITUDE'],

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=6,

            color='#008080'

        ),

        text=virus_data1['SPECIES'],

        name='CULEX PIPIENS/RESTUANS',

        

    ), go.Scattermapbox(

        lat=virus_data3['LATITUDE'],

        lon=virus_data3['LONGITUDE'],

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=6

        ),

        text=virus_data3['SPECIES'],

        name='CULEX TERRITANS',

    ), go.Scattermapbox(

        lat=virus_data5['LATITUDE'],

        lon=virus_data5['LONGITUDE'],

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=6

        ),

        text=virus_data5['SPECIES'],

        name='CULEX SALINARIUS'

    )])



fig.update_layout(

    autosize=True,

    hovermode='closest',

    title= 'Positive Case Analysis Across Chicago',

    mapbox=go.layout.Mapbox(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=go.layout.mapbox.Center(

            lat=41.88,

            lon=-87.63

        ),

        pitch=0,

        zoom=8,

    ),



)



fig.show()
import matplotlib.pyplot as plt

ctdf = (newData.reset_index()

          .groupby(['SEASON YEAR','RESULT'], as_index=False)

          .count()

          # rename isn't strictly necessary here, it's just for readability

          .rename(columns={'index':'Count of Positive Cases'})

       )

plt.figure(figsize=(16, 10))

g = sns.lineplot(x="SEASON YEAR", y="Count of Positive Cases", data=ctdf)



newDf = (newData.reset_index()

          .groupby(['TEST DATE','RESULT'], as_index=False)

          .count()

          # rename isn't strictly necessary here, it's just for readability

          .rename(columns={'index':'Count of Positive Cases'})

       )



plt.figure(figsize=(16, 10))

g = sns.lineplot(x="TEST DATE", y="Count of Positive Cases", data=newDf)

newDf['TEST DATE'] = pd.to_datetime(newDf['TEST DATE'],format='%Y-%m-%d')



for i in newDf.columns:

    if i != 'TEST DATE' and i != 'Count of Positive Cases':

        newDf = newDf.drop(i, axis=1)

print(newDf)        

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM



    

#setting index



#converting dataset into x_train and y_train

scaler = MinMaxScaler(feature_range=(0, 1)) ## training data is scaled from 0 to 1 (0 is min value and 1 is max)

x = []

y = []



newDf.index = newDf['TEST DATE']

newDf.drop('TEST DATE', axis=1, inplace=True)



def ltsmForecasting(p, scaler, newDf, units):

    #creating train and test sets

    dataset = newDf.values ## adds positive cases to dataset

    

    train = dataset[0:150]

    valid = dataset[150:]

    

    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(p,len(train)):

        x_train.append(scaled_data[i-p:i,0])

        y_train.append(scaled_data[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)



    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))





    # create and fit the LSTM network

    model = Sequential()

    model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1],1)))

    model.add(LSTM(units=units))

    model.add(Dense(1))



    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)



    #predicting 246 values, using past 60 from the train data

    inputs = newDf[len(newDf) - len(valid) - p:].values

    inputs = inputs.reshape(-1,1)

    inputs  = scaler.transform(inputs)



    X_test = []

    for i in range(p,inputs.shape[0]):

        X_test.append(inputs[i-p:i,0])

    X_test = np.array(X_test)



    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    closing_price = model.predict(X_test)

    closing_price = scaler.inverse_transform(closing_price)



    rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))

    print (rms)

    return [rms, closing_price]

x = []

y = []

hue = []

for p in range(5, 20):

    for units in range (10,60,20):

        result = ltsmForecasting(p, scaler, newDf, 50)

        x.append(p)

        y.append(result[0])

        hue.append(units)

    

    

sns.lineplot(x=x, y=y, hue=hue)
bestIndex = y.index(min(y))

print(bestIndex)

bestValue = x[bestIndex]

closing_price = ltsmForecasting(bestValue, scaler, newDf, hue[bestIndex])[1]

train = newDf[:150]

valid = newDf[150:]

valid['Predictions'] = closing_price

plt.plot(train['Count of Positive Cases'])

plt.plot(valid[['Count of Positive Cases','Predictions']])

plt.plot(valid[['Count of Positive Cases','Predictions']])