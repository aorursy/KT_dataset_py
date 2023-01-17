import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime



import plotly.graph_objects as go

import plotly.express as px

df = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")

print(df.shape)

df.head()
import missingno as msno

msno.matrix(df)
    

df['Date'] = pd.to_datetime(df['Date'])

def addYearMonthColumn(dataframe):



    dataframe['Year'] = dataframe['Date'].apply(lambda x : x.year)



    month_mapper = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun'

                   ,7:'Jul', 8:'Aug', 9:'Sep' ,10:'Oct', 11:'Nov', 12:'Dec'}

    dataframe['Month'] = dataframe['Date'].apply(lambda x : x.month).map(month_mapper)

    

    del month_mapper

    return dataframe



df = addYearMonthColumn(df)

df.sample(5)
df.Measure.unique()
def inboundTraffic(dataframe):

    resultDf = pd.DataFrame(dataframe.groupby(by='Measure')['Value'].sum().sort_values(ascending=False)).reset_index()

    fig = px.bar(resultDf, x='Measure', y='Value', height=400,color='Value', color_continuous_scale=px.colors.sequential.Magenta)

    fig.show()

    del resultDf

    

inboundTraffic(df)
import plotly.graph_objects as go



resultDf = df.groupby(by=['Border','Measure'])['Value'].sum().reset_index()

canada = resultDf.loc[resultDf['Border'] == 'US-Canada Border']['Value']

mexico = resultDf.loc[resultDf['Border'] == 'US-Mexico Border']['Value']



measures = resultDf['Measure']



fig = go.Figure()

fig.add_trace(go.Bar(

    x=measures,

    y=canada,

    name='Canada Border',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=measures,

    y=mexico,

    name='Mexico Border',

    marker_color='lightsalmon'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()

def pieChart(dataframe, feature):

    resultDf = dataframe.groupby(by=feature)['Value'].sum()

    fig = go.Figure(data=[go.Pie(labels = resultDf.index, values=resultDf.values)])

    fig.update_traces(textfont_size=15,  marker=dict(line=dict(color='#000000', width=2)))

    fig.show()

    del resultDf



pieChart(df, 'Border')

pieChart(df,'Measure')
def measure_values_by_years(dataframe, time_feature):

    

    plt.figure(figsize=(10,6))

    sns.lineplot(data=dataframe, x=time_feature, y='Value', hue='Measure',legend='full')

    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    plt.title('Measure Values Through ' + time_feature)

measure_values_by_years(df, 'Year')

measure_values_by_years(df, 'Month')
def crossing_by_post(data):

    resultDf = pd.DataFrame(data.groupby(by='Port Name')['Value'].sum().sort_values(ascending=False)).reset_index()

    fig = px.bar(resultDf, x='Port Name', y='Value', color='Value', color_continuous_scale=px.colors.sequential.Viridis) 

    fig.show()

crossing_by_post(df)
measure_size = {'Trucks' : 'Mid_Size', 'Rail Containers Full' : 'Mid_Size', 'Trains' : 'Big_Size',

       'Personal Vehicle Passengers':'Small_Size', 'Bus Passengers':'Small_Size',

       'Truck Containers Empty':'Mid_Size', 'Rail Containers Empty':'Mid_Size',

       'Personal Vehicles' : 'Small_Size', 'Buses' : 'Mid_Size', 'Truck Containers Full' : 'Mid_Size',

       'Pedestrians':'Small_Size', 'Train Passengers':'Small_Size'}



df['Size'] = df['Measure'].map(measure_size)



def crossing_by_measure_size(data):



    resultDf = data.groupby(by=['Size','State'])['Value'].sum().unstack()

    resultDf.fillna(0,inplace=True)



    plt.figure(figsize=(15,4))



    plt.subplot(131)

    resultDf.iloc[0].sort_values().plot(kind='bar', color='g')

    plt.xticks(rotation=90)

    plt.title('Big_Size')



    plt.subplot(132)

    resultDf.iloc[1].sort_values().plot(kind='bar')

    plt.xticks(rotation=90)

    plt.title('Mid_Size')



    plt.subplot(133)

    resultDf.iloc[2].sort_values().plot(kind='bar', color='red')

    plt.xticks(rotation=90)

    plt.title('Small_Size')



    del resultDf

    

crossing_by_measure_size(df)
def seasonality_check(data, timeLabel):

    plt.figure(figsize=(15,6))

    g = sns.FacetGrid(data=data, col='Size', sharey=False, height=5, aspect=1)

    g.map(sns.lineplot, timeLabel, 'Value')

    

seasonality_check(df,'Month')

seasonality_check(df,'Year')
from statsmodels.tsa.seasonal import seasonal_decompose

ts = df[['Date','Value']].groupby('Date').sum()

print(ts.shape)

print(ts.head(4))

ts = ts.loc['1997':]

ts.plot(figsize = (15,10))
# Multiplicative Decomposition 



ts_mult_decomposition = seasonal_decompose(ts, model='multiplicative', extrapolate_trend='freq')

ts_mult_decomposition.plot()
#ts_trend = ts_mult_decomposition.trend * ts_mult_decomposition.resid

from statsmodels.tsa.stattools import adfuller



result = adfuller(ts.Value.dropna())

print('p-value: %f' % result[1])



result = adfuller(ts.diff().Value.dropna())

print('p-value: %f' % result[1])
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 3, figsize=(16,10))



axes[0, 0].plot(ts.Value)

axes[0, 0].set_title('Original Series')

plot_pacf(ts, ax=axes[0, 1])

plot_acf(ts, ax=axes[0, 2])



# 1st Differencing

axes[1, 0].plot(ts.diff()); axes[1, 0].set_title('1st Order Differencing')

plot_pacf(ts.diff().dropna(), ax=axes[1, 1])

plot_acf(ts.diff().dropna(), ax=axes[1, 2])



def plot_trand_and_residual():

    trainingData = ts_mult_decomposition.trend * ts_mult_decomposition.resid

    trainingData.plot(figsize = (15,10))



    plt.show()

    return trainingData



training = plot_trand_and_residual()

training.shape
from statsmodels.tsa.arima_model import ARIMA



class Arima:

  def __init__(self, train_data, p,q,d=1):

    self.train_data = train_data

    self.p = p

    self.q=q

    self.d = d

    self.best_aic = np.Inf

    self.best_bic =np.Inf

    self.best_hqic = np.Inf

    self.best_order = (0,0,0)

    self.models = []

    self.model = 0

  

  def is_current_best_model(self):

    no_of_lower_metrics = 0

    if self.model.aic <= self.best_aic: no_of_lower_metrics+=1

    if self.model.bic <= self.best_bic: no_of_lower_metrics+=1

    if self.model.hqic <= self.best_hqic:no_of_lower_metrics+=1

    return no_of_lower_metrics >= 2



  def best_selection(self):

    for p_ in self.p:

        for q_ in self.q:

            

            currentOrder = (p_,q_)

            print("Current Order (p,q): "+ str(currentOrder))



            self.model = ARIMA(des, order=(1,0,1)).fit(disp=0)

            self.models.append(self.model)





            if self.is_current_best_model() == True:

                self.best_aic = np.round(self.model.aic,0)

                self.best_bic = np.round(self.model.bic,0)

                self.best_hqic = np.round(self.model.hqic,0)

                self.best_order = (p_,self.d,q_)

                current_best_model = self.model

                self.models.append(self.model)

                print('========================================================================')

                print("Best model so far: ARIMA" +  str(self.best_order) + 

                      " AIC:{} BIC:{} HQIC:{}".format(self.best_aic,self.best_bic,self.best_hqic)+

                      " resid:{}".format(np.round(np.exp(current_best_model.resid).mean(),3)))

                print('========================================================================')

                print()





    print('\n')

    print(current_best_model.summary())                

    return current_best_model, self.models 



x=range(2)

arima =Arima(training, x,x)

best_model, models = arima.best_selection()

best_model.plot_predict()

plt.title('Best Model')

plt.show()
def make_seasonal(ts,tms) :

    seasonal_series = ts * tms # Include the seasonality

    seasonal_series = seasonal_series[~seasonal_series.isnull()] # trim extra values

    return seasonal_series



def create_seasonal_component():

    seasonal = ts_mult_decomposition.seasonal.loc['2016-01-01':'2016-12-01'].values # seasonal component, we take the 2016 ones, but they are all the same.

    seasonal = pd.Series(np.tile(seasonal.flatten(),11), index = pd.date_range(start='2019-01-01', end = '2029-12-01', freq='MS'))  # This is just a very long series with the seasonality.

    return seasonal



def create_forecast():

    tms = create_seasonal_component()

    model = ARIMA(training, order=(1,0,1))

    model_fit = model.fit(disp=0)



    fc_series, se_series, conf_series = model_fit.forecast(n_forecast)  # 2 sigma Confidence Level (95,55% conf)



    # Make as pandas series and include seasonality

    fc_series = make_seasonal(pd.Series(fc_series, index = date_rng),tms)

    lower_series = make_seasonal(pd.Series(conf_series[:, 0], index = date_rng),tms)

    upper_series = make_seasonal(pd.Series(conf_series[:, 1], index = date_rng),tms)

    

    return fc_series,lower_series,upper_series



def plot_forecast(fc_series,lower_series,upper_series):

    plt.figure(figsize=(12,5), dpi=100)



    plt.plot(training * ts_mult_decomposition.seasonal, label='Time Series Data', color='g')

    plt.plot(fc_series , label='Forecast',color='r')



    # Confidence level intervals

    plt.fill_between(lower_series.index,lower_series, upper_series, 

                     color='k', alpha=.15, label='2$\sigma$ Confidence level (95%)')

    plt.title('Forecast 2019/20')

    plt.legend(loc='upper left', fontsize=8)

    plt.xlim('2000', '2021')

    plt.show()

# Forecast



def draw_forecast_main():



    date_start = training.tail(1).index[0]

    date_end = '2020-12-01'

    print(date_start)

    date_rng = pd.date_range(start=date_start, end=date_end, freq='MS', closed = 'right') # range for forecasting

    n_forecast = len(date_rng) # number of steps to forecast

    print('range of dates: '+str(n_forecast))



    

    fc_series,lower_series,upper_series = create_forecast()



    

    plot_forecast(fc_series,lower_series,upper_series)

    

draw_forecast_main()