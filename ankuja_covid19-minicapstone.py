import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import warnings

import matplotlib.dates as mdates

import datetime

import plotly.graph_objs as go

from scipy.integrate import solve_ivp

from scipy.optimize import minimize

from scipy.integrate import odeint

%config IPCompleter.greedy=True

warnings.filterwarnings('ignore')

%matplotlib inline 
#Reading datasets

df_age = pd.read_csv("/kaggle/input/covid19-in-india/AgeGroupDetails.csv",index_col = 0)

df_statetests = pd.read_csv("/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv")

df_hosp = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv',index_col=0)

df_pop = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv',index_col= 0)

df_individual = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv',index_col=0)

df_testlabs = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingLabs.csv',index_col= 0)

df_india = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv",index_col= 0)
df_india.head()
df_india.info()
df_india['Date'] = pd.to_datetime(df_india['Date'], dayfirst=True)
#Checking for missing values in all the dataframes

df_india.isnull().sum()
#Visualizing the trend in India for infected,confirmed, recovered

df_india['Date'] = pd.to_datetime(df_india['Date'], dayfirst=True)

Covid_df = df_india[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]

Covid_df = Covid_df.groupby('Date')[['Confirmed', 'Cured','Deaths']].sum().reset_index()

Covid_df['Active'] = Covid_df['Confirmed'] - Covid_df['Cured'] - Covid_df['Deaths']

Covid_df['new_case/day'] = Covid_df['Confirmed'] - Covid_df['Confirmed'].shift(1)

Covid_df['growth_ratio'] = Covid_df['new_case/day'] / Covid_df['new_case/day'].shift(1)

Covid_df['new_case/day'] = Covid_df['new_case/day'].replace(np.nan,'0.0')

Covid_df['growth_ratio'] = Covid_df['growth_ratio'].replace(np.nan,'0.0')

covid_melt_df = pd.melt(Covid_df, id_vars=['Date'], value_vars=['Confirmed','Active','Cured','new_case/day','Deaths'])

target_date = covid_melt_df['Date'].max()

fig = px.line(covid_melt_df, x="Date", y="value", color='variable', 

              title=f'All-India Cases as of {target_date}')

fig.show()
state_report = df_india.groupby(by = 'State/UnionTerritory').max().reset_index()

import IPython

IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1977187" data-url="https://flo.uri.sh/visualisation/1977187/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
max_date = df_india['Date'].max()

states_df = df_india.query('(Date == @max_date) & (Confirmed > 35000)').sort_values('Confirmed', ascending=False)

states_df['Active'] = states_df['Confirmed'] - states_df['Cured'] - states_df['Deaths']

states_melt_df = pd.melt(states_df, id_vars='State/UnionTerritory', value_vars=['Confirmed','Active', 'Cured','Deaths'])

fig = px.bar(states_melt_df.iloc[::-1],

             x='value', y='State/UnionTerritory', color='variable', barmode='group',

             title=f'Confirmed/Cured/Deaths as on {max_date}', text='value', height=800, orientation='h')

fig.show()
latest = df_india[df_india["Date"] > pd.to_datetime('2020-04-01')]



latest2 = latest.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured',"Date"].max().reset_index()



latest2['Active'] = latest2['Confirmed'] - (latest2['Deaths'] - latest2['Cured'])



state_list = list(latest2.sort_values('Active',ascending = False)['State/UnionTerritory'])[0:15]



states_confirmed = {}

states_deaths = {}

states_recovered = {}

states_active = {}

states_dates = {} 
fig = plt.figure(figsize=(25,20),dpi = 250)



import matplotlib.dates as mdates





def movingaverage(values,N):

    cumulativesum = [0]

    movingav = []

    

    for i,x in enumerate(values,1):

        cumulativesum.append(cumulativesum[i-1] + x)

        

        if i >= N:

            movingav.append((cumulativesum[i] - cumulativesum[i-N]) / N)

        else:

            movingav.append(0)

    return movingav





def percentchange(values):

    

    pctchange = []

    

    for i in range(0,len(values)):

        pastsum = 0

        

        for j in range(i):

            pastsum = pastsum + values[j]

        

        pctchange.append(int((values[i]/pastsum)*100))

        

    return pctchange



axno = 1

sns.set_style('darkgrid')



for state in state_list:

    df1 = latest[latest['State/UnionTerritory'] == state].reset_index(drop = True)



    new_cases = [0]



    state_dates = latest[latest['State/UnionTerritory'] == state]['Date']

    

    for i in range (1,len(df1)):

        cases_delta = (df1['Confirmed'][i] - df1['Confirmed'][i-1])

        new_cases.append(cases_delta)



    df1['New Cases'] = new_cases

    

    

    ax = fig.add_subplot(5,3,axno)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    

    ax.bar(state_dates.values,df1['New Cases'], label = 'Day wise confirmed cases',color = 'yellow')

    plt.title(state,fontsize = 20)

    

    plt.xlim(pd.to_datetime('2020-04-01'),pd.to_datetime('2020-08-01'))

    moving_aves = movingaverage(df1['New Cases'],10)

    

    ax.plot(state_dates.values,moving_aves,color='red',lw = 2.5,label = 'Moving Average')

    ax.text(datetime.date(2020,4, 15), 28, "*",fontsize = 20)

    ax.text(datetime.date(2020,5,4), 31, "*",fontsize = 20)

    ax.text(datetime.date(2020,6, 18), 120, "*",fontsize = 20)



    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper left',)

    axno = axno + 1



plt.suptitle('Confirmed Cases with a 10 day moving average',fontsize = 30,y=1.0)

plt.tight_layout(pad = 4.0)

plt.savefig("State")

df_statetests.head()
df_statetests.info()
#Converting the Date column's datatype to date 

df_statetests['Date']= pd.to_datetime(df_statetests['Date'], dayfirst=True)

#State wise testing

state_test = pd.pivot_table(df_statetests, values=['TotalSamples','Negative','Positive'], index='State', aggfunc='max')

state_names = list(state_test.index)

state_test['State'] = state_names



plt.figure(figsize=(15,10))

sns.set_color_codes("pastel")

sns.barplot(x="TotalSamples", y= state_names, data=state_test,label="Total Samples", color = '#9370db')

#sns.barplot(x="Negative", y=state_names, data=state_test,label='Negative', color= '#ff9999')

sns.barplot(x="Positive", y=state_names, data=state_test,label='Positive', color='#87479d')

plt.title('Testing statewise insight',fontsize = 20)

plt.legend(ncol=2, loc="lower right", frameon=True);
values = list(df_testlabs['state'].value_counts())

names = list(df_testlabs['state'].value_counts().index)



plt.figure(figsize=(15,10))

sns.set_color_codes("pastel")

plt.title('ICMR Testing Centers in each State', fontsize = 20)

sns.barplot(x= values, y= names,color = '#00FFFF');
#Prediction using fbprophet



model_pro_df = df_india.groupby("Date")["Confirmed"].sum().reset_index()

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot





#convert data to natural log as for some cases values might be rreally small while for others its quite large. Therefore to reduce the impact of such

#outliers of sort we take log to normalize distribution of data.



model_pro_df['Confirmed']= np.log(model_pro_df['Confirmed'])



#making dataset prophet compliant

model_pro_df.columns = ['ds','y']

model_pro_df.head()
m1 = Prophet(daily_seasonality=True)

m1.fit(model_pro_df)

future=m1.make_future_dataframe(periods=90)

prop_forecast=m1.predict(future)

forecast = prop_forecast[['ds','yhat']].tail(30)

prop_forecast.tail().T



fig = plot_plotly(m1, prop_forecast)

fig = m1.plot(prop_forecast,xlabel='Date',ylabel='Confirmed Cases')
prop_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
np.exp(prop_forecast[['yhat', 'yhat_lower', 'yhat_upper']].tail())
m1.plot_components(prop_forecast);
#Performance Metric



from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(m1, initial='30 days', period='15 days', horizon = '120 days')

df_cv.head()
from fbprophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(df_cv, metric='rmse')
from fbprophet.diagnostics import performance_metrics

performance_metrics_results = performance_metrics(df_cv)

performance_metrics_results.describe()
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



#merging the predicted values and the original one

Final_df = prop_forecast.set_index('ds')[['yhat']].join(model_pro_df.set_index('ds').y).reset_index()

Final_df.dropna(inplace=True)

print("R2 SCORE")

print(r2_score(Final_df.y, Final_df.yhat))



print("Mean Squared Error")

print(mean_squared_error(Final_df.y, Final_df.yhat))



print("Mean Absolute Error")

print(mean_absolute_error(Final_df.y, Final_df.yhat))
fig = plt.figure(figsize=(15, 10))

ax = plt.subplot() 

plt.title("Actual Vs Predicted Covid Cases")

ax.plot(prop_forecast['ds'],prop_forecast['yhat'],color='green',linestyle='dashed')

ax.plot(model_pro_df['ds'],model_pro_df['y'],color='red')


