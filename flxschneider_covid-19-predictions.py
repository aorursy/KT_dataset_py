import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

train = train.fillna(0)

train.Date = pd.to_datetime(train.Date)



test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

test = test.fillna(0)

test.Date = pd.to_datetime(test.Date)

sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')



def only_nth_tick(n):

    ax = plt.gca()

    temp = ax.xaxis.get_ticklabels()

    temp = list(set(temp) - set(temp[::n]))

    for label in temp:

        label.set_visible(False)
train.head(5)

data = train.groupby(by='Date').sum()['ConfirmedCases']



# plotting

fig = plt.figure(figsize=(10,5))

plt.plot(data,'--x')

plt.ylabel('# of people')

plt.xticks(rotation=40)

only_nth_tick(10)

plt.grid()

plt.title('Covid-19 - Confirmed infections worldwide')
# Countries with most confirmed cases

most_conf_cases = train.groupby('Country_Region').max()['ConfirmedCases'].sort_values()[::-1]

most_conf_cases
china = train[train.Country_Region == 'Korea, South']

italy = train[train.Country_Region == 'Italy']



plt.figure(figsize = (12,8))

plt.subplot(2,1,1)

plt.title('South Korea')

plt.ylabel('# confirmed cases')

plt.xticks(rotation=20)

only_nth_tick(5)

plt.plot(china.ConfirmedCases)

plt.subplot(2,1,2)

plt.ylabel('# confirmed cases')

plt.title('Italy')

plt.xticks(rotation=20)

only_nth_tick(5)

plt.plot(italy.ConfirmedCases)

plt.tight_layout()
country = 'Germany'

df = train[train.Country_Region == country]

df = df[df.ConfirmedCases > 100]['ConfirmedCases'].reset_index(drop=True)



X = df.index.values

y = df.values
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



def rmse(y_pred, y_true):

    return np.sqrt(mean_squared_error(y_pred, y_true))



lm = LinearRegression()

lm.fit(X.reshape(-1,1), np.log(y))

print(f'RMSE = {rmse(np.exp(lm.predict(X.reshape(-1,1))), y)}')
fig = plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.plot(np.exp(lm.predict(X.reshape(-1,1))), label='Model')

plt.plot(y,label='Data')

#plt.vlines(split,0,np.max(y),linestyles='dashed',color='red',label='Train/Test-Split')

plt.legend()

plt.title(f'Log-transformed Linear Fit - Confirmed cases in {country}')

plt.xlabel('Days')



plt.grid()



plt.subplot(1,2,2)

plt.plot(np.exp(lm.predict(X.reshape(-1,1))), label='Model')

plt.plot(y,label='Data')

#plt.vlines(split,0,np.max(y),linestyles='dashed',color='red',label='Train/Test-Split')

plt.legend()

plt.title(f'Log-transformed Linear Fit - Confirmed cases in {country} (Logscale)')

plt.yscale('log')

plt.xlabel('Days')



plt.grid()



plt.tight_layout()
train
fig = plt.figure(figsize=(20,15))



for i, c in enumerate(['Germany', 'Iran', 'Italy', 'Spain','Korea, South']):

    plt.subplot(3,3,i+1)

    df = train[train.Country_Region == c]

    df = df[df.ConfirmedCases > 200]['ConfirmedCases'].reset_index(drop=True)

    

    X = df.index.values

    y = df.values

    

    lm = LinearRegression()

    lm.fit(X.reshape(-1,1), np.log(y))





    plt.plot(np.exp(lm.predict(X.reshape(-1,1))), label='Model')

    plt.plot(y,label='Data')

    plt.legend()

    plt.title(f'Log-transformed Linear Fit - Confirmed cases in {c}')

    plt.grid()

    plt.ylabel('# confirmed cases')

    plt.xlabel('Days')

    print(f'Coountry = {c}, RMSE = {rmse(np.exp(lm.predict(X.reshape(-1,1))), y)}')

plt.tight_layout()
train['region_id'] = train.apply(lambda x: x['Country_Region'] +"_"+ str(x['Province_State']), axis=1)



data = {'ForecastId':[], 'ConfirmedCases':[],'Fatalities':[]}
for c in train.region_id.unique():

    df = train[train.Country_Region == c]

    max_conf = np.max(df.ConfirmedCases)

    start_date_conf = df[df.ConfirmedCases > 0.1*max_conf].head(1).Date.values[0]

    df = df[df.ConfirmedCases > 0.1*max_conf]

    df['days'] = (df.Date - start_date_conf).dt.days

    model_conf = LinearRegression()

    model_conf.fit(df['days'].values.reshape(-1,1), np.log(df['ConfirmedCases'].values))

    

    

    df = train[train.Country_Region == c]

    start_date_fatal = df[df.ConfirmedCases > 200].head(1).Date.values[0]

    df = df[df.Fatalities > 5]

    df['days'] = (df.Date - start_date_fatal).dt.days

    model_fatal = LinearRegression()

    model_fatal.fit(df['days'].values.reshape(-1,1), np.log(df['Fatalities'].values))

    

    

    predict = test[test.Country_Region == c]

    predict['days'] = (predict.Date - start_date_conf).dt.days

    conf_pred = np.exp(model_conf.predict(predict.days.values.reshape(-1,1)))

    

    predict = test[test.Country_Region == c]

    predict['days'] = (predict.Date - start_date_conf).dt.days

    conf_pred = np.exp(model_conf.predict(predict.days.values.reshape(-1,1)))

    

    

    predict = test[test.Country_Region == c]

    predict['days'] = (predict.Date - start_date_conf).dt.days

    conf_pred = np.exp(model_conf.predict(predict.days.values.reshape(-1,1)))
predict = test[test.Country_Region == 'Germany']

predict['days'] = (predict.Date - start_date_conf).dt.days

conf_pred = np.exp(model_conf.predict(predict.days.values.reshape(-1,1)))
predict = test[test.Country_Region == 'Germany']

predict['days'] = (predict.Date - start_date_fatal).dt.days

fatal_pred = np.exp(model_fatal.predict(predict.days.values.reshape(-1,1)))
data['ForecastId'].extend(predict.ForecastId.values)

data['ConfirmedCases'].extend(conf_pred)

data['Fatalities'].extend(fatal_pred)