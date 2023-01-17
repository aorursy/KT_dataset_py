import pandas as pd

import plotly.express as px

import math

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from collections import deque

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random

import time

import os
virusdata = "/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv"

virus_data = pd.read_csv(virusdata)
virus_data
import plotly.graph_objects as go

grouped_multiple = virus_data.groupby(['Date']).agg({'Confirmed': ['sum']})

grouped_multiple.columns = ['Confirmed ALL']

grouped_multiple = grouped_multiple.reset_index()

fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 

                         y=grouped_multiple['Confirmed ALL'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='orange', width=2)))

fig.show()
china_vs_rest = virus_data.copy()

china_vs_rest.loc[china_vs_rest.Country == 'Mainland China', 'Country'] = "China"

china_vs_rest.loc[china_vs_rest.Country != 'China', 'Country'] = "Not China"

china_vs_rest = china_vs_rest.groupby(['Date', 'Country']).agg({'Confirmed': ['sum']})

china_vs_rest.columns = ['Confirmed ALL']

china_vs_rest = china_vs_rest.reset_index()

fig = px.line(china_vs_rest, x="Date", y="Confirmed ALL", color="Country",

              line_group="Country", hover_name="Country")

fig.update_layout(template='plotly_dark')

fig.show()
china_vs_rest = virus_data.copy()

china_vs_rest = china_vs_rest[china_vs_rest.Country != 'Mainland China']

china_vs_rest = china_vs_rest[china_vs_rest.Country != 'China']

china_vs_rest = china_vs_rest.groupby(['Date', 'Country']).agg({'Confirmed': ['sum']})

china_vs_rest.columns = ['Confirmed ALL']

china_vs_rest = china_vs_rest.reset_index()

fig = px.line(china_vs_rest, x="Date", y="Confirmed ALL", color="Country",

              line_group="Country", hover_name="Country")

fig.update_layout(template='plotly_dark')

fig.show()
grouped_multiple = virus_data.groupby(['Date']).agg({'Deaths': ['sum'],'Recovered': ['sum']})

grouped_multiple.columns = ['Deaths ALL','Recovered ALL']

grouped_multiple = grouped_multiple.reset_index()



fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 

                         y=grouped_multiple['Deaths ALL'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='red', width=2)))



fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 

                         y=grouped_multiple['Recovered ALL'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='green', width=2)))

fig.show()
grouped_multiple = virus_data.groupby(['Date']).agg({'Deaths': ['sum'],'Recovered': ['sum']})

grouped_multiple.columns = ['Deaths_ALL','Recovered_ALL']

grouped_multiple = grouped_multiple.reset_index()



grouped_multiple['Deaths_ALL_%'] = grouped_multiple.apply(lambda row: (row.Deaths_ALL*100)//

                                               (row.Deaths_ALL + row.Recovered_ALL) 

                                               if row.Deaths_ALL  else 0, axis=1)



grouped_multiple['Recovered_ALL_%'] = grouped_multiple.apply(lambda row: (row.Recovered_ALL*100)//

                                               (row.Deaths_ALL + row.Recovered_ALL) 

                                               if row.Deaths_ALL  else 0, axis=1)





fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 

                         y=grouped_multiple['Deaths_ALL_%'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='red', width=2)))



fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 

                         y=grouped_multiple['Recovered_ALL_%'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='green', width=2)))

fig.show()
grouped_multiple = virus_data.groupby(['Date']).agg({'Deaths': ['sum'],'Recovered': ['sum'], 'Confirmed': ['sum']})

grouped_multiple.columns = ['Deaths_ALL','Recovered_ALL', 'All']

grouped_multiple = grouped_multiple.reset_index()

grouped_multiple['Deaths_Revocered'] = grouped_multiple.apply(lambda row: row.Deaths_ALL + row.Recovered_ALL, axis=1)



fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 

                         y=grouped_multiple['Deaths_ALL'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='red', width=2)))



fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 

                         y=grouped_multiple['Recovered_ALL'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='green', width=2)))



fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 

                         y=grouped_multiple['All'],

                         mode='lines+markers',

                         name='All',

                         line=dict(color='orange', width=2)))



fig.add_trace(go.Scatter(x=grouped_multiple['Date'], 

                         y=grouped_multiple['Deaths_Revocered'],

                         mode='lines+markers',

                         name='Deaths + Recovered',

                         line=dict(color='white', width=2)))



fig.show()
grouped_multiple = virus_data.groupby(['Date']).agg({'Deaths': ['sum'],'Recovered': ['sum'], 'Confirmed': ['sum']})

grouped_multiple.columns = ['Deaths_ALL','Recovered_ALL', 'All']

grouped_multiple = grouped_multiple.reset_index()

grouped_multiple['Deaths_Revocered'] = grouped_multiple.apply(lambda row: row.Deaths_ALL + row.Recovered_ALL, axis=1)

grouped_multiple
grouped_multiple['infections_perc'] = grouped_multiple['All'].pct_change(1)

grouped_multiple['recovered_perc'] = grouped_multiple['Recovered_ALL'].pct_change(1)

grouped_multiple['death_perc'] = grouped_multiple['Deaths_ALL'].pct_change(1)

grouped_multiple = grouped_multiple.replace([np.inf, -np.inf], np.nan)

main_df=grouped_multiple.fillna(0)

main_df
fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=main_df['Date'], 

                         y=main_df['infections_perc'],

                         mode='lines+markers',

                         name='infections_perc',

                         line=dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=main_df['Date'], 

                         y=main_df['recovered_perc'],

                         mode='lines+markers',

                         name='recovered_perc',

                         line=dict(color='green', width=2)))

fig.add_trace(go.Scatter(x=main_df['Date'], 

                         y=main_df['death_perc'],

                         mode='lines+markers',

                         name='death_perc',

                         line=dict(color='red', width=2)))

fig.show()
def IncreaseData(dflist):

    

    NewList=[]

    add_this=0

    for split_value in dflist:

        increment_by = int((split_value-add_this)//24)

        for new_val in range(24):

            add_this=increment_by+add_this

            NewList.append(add_this)

    return NewList



inc_total = IncreaseData(grouped_multiple['All'])

inc_death = IncreaseData(grouped_multiple['Deaths_ALL'])

inc_rec = IncreaseData(grouped_multiple['Recovered_ALL'])

df = pd.DataFrame(list(zip(inc_total, inc_death, inc_rec)), 

               columns =['inc_total', 'inc_death', 'inc_rec']) 

df
fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=df.index, 

                         y=df['inc_total'],

                         mode='lines',

                         name='inc_total',

                         line=dict(color='orange', width=2)))

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=df.index, 

                         y=df['inc_rec'],

                         mode='lines',

                         name='inc_rec',

                         line=dict(color='green', width=2)))

fig.add_trace(go.Scatter(x=df.index, 

                         y=df['inc_death'],

                         mode='lines',

                         name='inc_death',

                         line=dict(color='red', width=2)))



fig.show()
inc_total_perc = df['inc_total'].pct_change(1)

inc_death_perc = df['inc_death'].pct_change(1)

inc_rec_perc = df['inc_rec'].pct_change(1)

dff = pd.DataFrame(list(zip(inc_total_perc, inc_death_perc, inc_rec_perc)), 

              columns =['inc_total_perc', 'inc_death_perc', 'inc_rec_perc']) 

dff=dff.replace([np.inf, -np.inf], np.nan)

dff=dff.fillna(0)

dff
class TrainLSTM():

    def create_dataset(self, dataset, look_back=1, column = 0):

        dataX, dataY = [], []

        for i in range(len(dataset)-look_back-1):

            a = dataset[i:(i+look_back), column]

            dataX.append(a)

            dataY.append(dataset[i + look_back, column])

        return np.array(dataX), np.array(dataY)



    def TrainModel(self, dframe, column):

        df = dframe.values

        df = df.astype('float32')

        train_size = int(len(df) * 0.90)

        test_size = len(df) - train_size

        Train, Validate = df[0:train_size,:], df[train_size:len(df),:]

        look_back = 24

        trainX, trainY = self.create_dataset(Train, look_back, column)

        testX, testY = self.create_dataset(Validate, look_back, column)

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        model = Sequential()

        model.add(LSTM(4, input_shape=(1, look_back)))

        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(trainX, trainY, epochs=300, batch_size=1, verbose=2)

        self.trainPredict = model.predict(trainX)

        self.testPredict = model.predict(testX)

        trainScore = math.sqrt(mean_squared_error(trainY, self.trainPredict[:,0]))

        print('Train Score: %.2f RMSE' % (trainScore))

        testScore = math.sqrt(mean_squared_error(testY, self.testPredict[:,0]))

        print('Test Score: %.2f RMSE' % (testScore))

        Model_Prediciton_Resolved=[]

        lastDT=testX[0][0]

        print(lastDT)

        for i in range(168):

            predi = model.predict(np.array([[lastDT]]))

            Model_Prediciton_Resolved.append(predi[0][0])

            lastDT = lastDT[:-1]

            lastDT = np.append(predi, lastDT)



        return Model_Prediciton_Resolved
NeuralNets = TrainLSTM()

result_total = NeuralNets.TrainModel(dff,0)

result_death = NeuralNets.TrainModel(dff,1)

result_rec = NeuralNets.TrainModel(dff,2)
fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=list(range(len(result_total))), 

                         y=result_total,

                         mode='lines+markers',

                         name='result_total',

                         line=dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=list(range(len(result_death))), 

                         y=result_death,

                         mode='lines+markers',

                         name='result_death',

                         line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=list(range(len(result_rec))), 

                         y=result_rec,

                         mode='lines+markers',

                         name='result_rec',

                         line=dict(color='green', width=2)))

fig.show()
def FinalChartCalc(df,startVal):

    finalist=[]

    start=startVal

    for item in df:

        percent = start*item

        start = start+percent

        finalist.append(start)

    return finalist
growth_total = FinalChartCalc(result_total,1)

growth_death =FinalChartCalc(result_death,1)

growth_rec =FinalChartCalc(result_rec,1)

fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=list(range(len(growth_total))), 

                         y=growth_total,

                         mode='lines+markers',

                         name='result_total',

                         line=dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=list(range(len(growth_death))), 

                         y=growth_death,

                         mode='lines+markers',

                         name='growth_death',

                         line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=list(range(len(growth_rec))), 

                         y=growth_rec,

                         mode='lines+markers',

                         name='growth_rec',

                         line=dict(color='green', width=2)))

fig.show()

real_total = FinalChartCalc(result_total,df['inc_total'].tail(1).values[0])

real_death = FinalChartCalc(result_death, df['inc_death'].tail(1).values[0])

real_rec = FinalChartCalc(result_rec, df['inc_rec'].tail(1).values[0])

fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=list(range(len(real_total))), 

                         y=real_total,

                         mode='lines+markers',

                         name='real_total',

                         line=dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=list(range(len(real_rec))), 

                         y=real_rec,

                         mode='lines+markers',

                         name='real_rec',

                         line=dict(color='green', width=2)))

fig.add_trace(go.Scatter(x=list(range(len(real_death))), 

                         y=real_death,

                         mode='lines+markers',

                         name='real_death',

                         line=dict(color='red', width=2)))



fig.show()