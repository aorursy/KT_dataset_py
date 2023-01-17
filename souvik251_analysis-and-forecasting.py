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



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



from pathlib import Path

data_dir = Path('../input/covid19-global-forecasting-week-1')



import os

os.listdir(data_dir)

data = pd.read_csv(data_dir/'train.csv')

data.head()
data2= pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

data2.head()
cleaned_data=data2.rename(columns={'Country/Region':'country','Province/State':'state','Date':'date'})
cleaned_data['active']= cleaned_data['Confirmed']- cleaned_data['Deaths']- cleaned_data['Recovered']



cleaned_data['country']=cleaned_data['country'].replace('Mainland China','china')



cleaned_data.head(2)
cases=['Deaths','Confirmed','Recovered','active']

cleaned_data[cases]=cleaned_data[cases].fillna(0)



cleaned_data['state']=cleaned_data[['state']].fillna('')
data=cleaned_data
print("External Data")

print(f"Earliest Entry: {data['date'].min()}")

print(f"Last Entry:     {data['date'].max()}")
grouped = data.groupby('date')['date', 'Confirmed', 'Deaths'].sum().reset_index()



fig = px.line(grouped, x="date", y="Confirmed", 

              title="Worldwide Confirmed Cases Over Time")

fig.show()



fig = px.line(grouped, x="date", y="Confirmed", 

              title="Worldwide Confirmed Cases (Logarithmic Scale) Over Time", 

              log_y=True)

fig.show()


us = data[data['country'] == "US"].reset_index()

us_date = us.groupby('date')['date', 'Confirmed', 'Deaths'].sum().reset_index()

fig = px.line(us_date, x="date", y="Confirmed", 

              title="us's Confirmed Cases Over Time")

fig.show()
china=data[data['country']=='China'].reset_index()

china_date=china.groupby('date')['date','Confirmed','Deaths'].sum().reset_index()



fig=px.line(china_date,x='date',y='Confirmed',title="china's confirmed cases over time")

fig.show()
Italy=data[data['country']=='Italy'].reset_index()

Italy_date=Italy.groupby('date')['date','Confirmed','Deaths'].sum().reset_index()



fig=px.line(Italy_date,x='date',y='Confirmed',title="Italy's confirmed cases over time")

fig.show()
India=data[data['country']=='India'].reset_index()

India_date=India.groupby('date')['date','Confirmed','Deaths'].sum().reset_index()



fig=px.line(India_date,x='date',y='Confirmed',title="India's confirmed cases over time")

fig.show()
rest = data[~data['country'].isin(['China', 'US','India'])].reset_index()

grouped_rest_date = rest.groupby('date')['date', 'Confirmed', 'Deaths'].sum().reset_index()

fig=px.line(grouped_rest_date,x='date',y='Confirmed',title="other's confirmed cases over time")

fig.show()
data_latest = data[data['date'] == max(data['date'])]

flg = data_latest.groupby('country')['Confirmed', 'Deaths', 'Recovered', 'active'].sum().reset_index()



flg['mortalityRate'] = round((flg['Deaths']/flg['Confirmed'])*100, 2)

temp = flg[flg['Confirmed']>100]

temp = temp.sort_values('mortalityRate', ascending=False)



fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:20][::-1],

             x = 'mortalityRate', y = 'country', 

             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',

             color_discrete_sequence=['darkred']

            )

fig.show()
flg['recoveryRate'] = round((flg['Recovered']/flg['Confirmed'])*100, 2)

temp = flg[flg['Confirmed']>100]

temp = temp.sort_values('recoveryRate', ascending=False)





fig = px.bar(temp.sort_values(by="recoveryRate", ascending=False)[:10][::-1],

             x = 'recoveryRate', y = 'country', 

             title='Recoveries per 100 Confirmed Cases', text='recoveryRate', height=800, orientation='h',

             color_discrete_sequence=['#2ca02c']

            )

fig.show()


print("Countries with Worst Recovery Rates")

temp = flg[flg['Confirmed']>100]

temp = temp.sort_values('recoveryRate', ascending=True)[['country', 'Confirmed','Recovered']][:20]

temp.sort_values('Confirmed', ascending=False)[['country', 'Confirmed','Recovered']][:20].style.background_gradient(cmap='Reds')
#try to fit a linear regression model
import numpy as np



grouped["Days Since"]=grouped.index-grouped.index[0]

grouped["Days Since"]=np.array(grouped["Days Since"])
train=grouped.iloc[:int(grouped.shape[0]*0.85)]

valid=grouped.iloc[int(grouped.shape[0]*0.85):]
from sklearn import linear_model

reg=linear_model.LinearRegression(normalize=True)
reg.fit(np.array(train["Days Since"]).reshape(-1,1),np.array(train["Confirmed"]).reshape(-1,1))


prediction_valid_reg=reg.predict(np.array(valid["Days Since"]).reshape(-1,1))
import matplotlib.pyplot as plt

plt.figure(figsize=(11,6))

prediction_reg=reg.predict(np.array(grouped["Days Since"]).reshape(-1,1))

plt.plot(grouped["Confirmed"],label="Actual Confirmed Cases")

plt.plot(grouped.index,prediction_reg, linestyle='--',label="Predicted Confirmed Cases using Linear Regression",color='black')

plt.xlabel('Time')

plt.ylabel('Confirmed Cases')

plt.xticks(rotation=90)

plt.legend()
data.dtypes
no_country = data['country'].nunique()

no_province = data['state'].nunique()

no_country_with_prov = (data[data['state'].isna()==False]['country'].nunique())

total_forecasting_number = (no_province + no_country - no_country_with_prov+2)

no_days = data['date'].nunique()

print('there are ', no_country, 'unique countries, each with ', no_days, 'days of data, all of them having the same dates.There are also',

      no_province, 'Provinces/States which can be found on ',

      no_country_with_prov, 'countries/ regions.' )


confirmed_total_date = data.groupby(['date'])['Confirmed'].sum().reset_index()

df=confirmed_total_date.copy()

df = pd.DataFrame({'date': [df.index[i] for i in range(len(df))] , 'cases': df['Confirmed'].values.reshape(1,-1)[0].tolist()})

dfog = df.copy()

def model(x,y):

    model = reg.fit(x, y)

    return model



x = df['cases']

x = x.drop(x.index[-1]).values.reshape((-1, 1))

y = df['cases']

y = y.drop(y.index[0])

ex_slope = reg.fit(x,y).coef_



d = 0



for i in range(1,5):

    plt.plot(df['cases'])

    plt.show()

    plt.close()

    df['prev_cases'] = df['cases'].shift(1)

    df['cases'] = (df['cases'] - df['prev_cases'])

    df = df.drop(['prev_cases'],axis=1)

    df = df.drop(df.index[0])

    x = df['cases']

    x = x.drop(y.index[-1]).values.reshape((-1, 1))

    y = df['cases']

    y = y.drop(y.index[0])

    model = reg.fit(x,y)

    if( abs(model.coef_) > ex_slope):

        print('this is it! ', ex_slope)

        break

    d += 1

    ex_slope = model.coef_

    print(model.coef_)
!pip install pyramid-arima

import pyramid.arima

import warnings

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import mean_squared_error

X = dfog['cases'].values

size = int(len(X) * 0.80)

Atrain, Atest = X[0:size], X[size:len(X)]

history = [x for x in Atrain]

predictions = list()

for t in range(len(Atest)):

    model = ARIMA(history, order=(1,2,0))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = Atest[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(Atest, predictions)

print('Test MSE: %.3f' % error)

# plot

plt.plot(Atest)

plt.plot(predictions, color='red')

plt.show()
from tqdm import tqdm

index = 1

cases_pred= []

fatalities_pred = []

pbar = tqdm(total=total_forecasting_number)

while index < total_forecasting_number+1:

    x = data['Confirmed'].iloc[[i for i in range(no_days*(index-1),no_days*index)]].values

    z = data['Deaths'].iloc[[i for i in range(no_days*(index-1),no_days*index)]].values

    index += 1

    no_nul_cases = pd.DataFrame(x)

    no_nul_cases = no_nul_cases[no_nul_cases.values != 0]

    if(not no_nul_cases.empty):

        X = [xi for xi in no_nul_cases.values]

        try:

            model = pyramid.arima.auto_arima(X,seasonal=True, m=12)

            pred = model.predict(31)

            pred = pred.astype(int)

            pred = pred.tolist()

        except:

            model = reg.fit(np.array([i for i in range(len(X))]).reshape(-1, 1),X)

            pred = [(model.coef_*(len(X)+i) + model.intercept_).astype('int')[0][0] for i in range(1,32)]

                

    else:

        pred = [0] * 31

    pred = x[-12:].astype(int).tolist() + pred

    cases_pred+=pred

    

    no_nul_fatalities = pd.DataFrame(z)

    no_nul_fatalities = no_nul_fatalities[no_nul_fatalities.values != 0]

    if(not no_nul_fatalities.empty):

        Z = [zi for zi in no_nul_fatalities.values]

        try:

            model = pyramid.arima.auto_arima(Z, seasonal=False, m=12)

            pred = model.predict(31)

            pred = pred.astype(int)

            pred = pred.tolist()

        except:

            model = reg.fit(np.array([i for i in range(len(Z))]).reshape(-1, 1),Z)

            pred = [(model.coef_*(len(Z)+i) + model.intercept_).astype('int')[0][0] for i in range(1,32)]

    else:

        pred = [0] * 31

    pred = z[-12:].astype(int).tolist() + pred

    fatalities_pred+=pred

    pbar.update(1)

pbar.close()



    
submission = pd.DataFrame({'ForecastId': [i for i in range(1,len(cases_pred)+1)] ,'Confirmed': cases_pred, 'Fatalities': fatalities_pred})

filename = 'submission.csv'

submission.to_csv(filename,index=False)