from pandas import *

from matplotlib.pyplot import *

from sklearn import *

from fbprophet import *

from sklearn.model_selection import *

from sklearn.linear_model import *

from numpy import * 

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import statsmodels.api as sm

import warnings

import itertools

from fbprophet import *

warnings.filterwarnings("ignore")
covid = read_csv('../input/covid-19-dataset-world-wide/owid-covid-data.csv')
covid.isnull().sum()
# for data exploration we only need 5 columns:Continent,Country,Date,Total Death,Total cases

covid_req = DataFrame({'Continent':covid['continent'],'Country':covid['location'],'Date':covid['date'],'Death':covid['total_deaths'],'Cases':covid['total_cases']})
covid_req = covid_req.fillna(0)
covid_req['Date'] = to_datetime(covid_req['Date']).dt.strftime('%d-%m-%Y')
tot_df = covid[(covid['location']=='World')&(covid['date']=='2020-08-30')]

tot_case = tot_df['total_cases'].max()

tot_deth = covid['total_deaths'].max()

print("Total cases reported until 30th of August 2020 worldwide is {}\n".format(tot_case))

print("Total daths reported until 30th of August 2020 worldwide is {}".format(tot_deth))
covid_min_max = covid_req[(covid_req['Date']=='30-08-2020')&(covid_req['Country']!='International')&(covid_req['Country']!='World')]

min_deth = covid_min_max['Death'].min()

max_deth = covid_min_max['Death'].max()

min_deth_contry = covid_min_max.iloc[covid_min_max['Death'].argmin()]['Country']

max_deth_contry = covid_min_max.iloc[covid_min_max['Death'].argmax()]['Country']

print("The country which has reported lowest number of death due to covid 19({}) until 30th of August 2020 is {}\n".format(min_deth,min_deth_contry))

print("The country which has reported highest number of death due to covid 19({}) until 30th of August 2020 is {}".format(max_deth,max_deth_contry))
min_case = covid_min_max['Cases'].min()

max_case = covid_min_max['Cases'].max()

min_case_contry = covid_min_max.iloc[covid_min_max['Cases'].argmin()]['Country']

max_case_contry = covid_min_max.iloc[covid_min_max['Cases'].argmax()]['Country']

print("The country which has reported lowest number of covid 19 cases({}) until 30th of August 2020 is {}\n".format(min_case,min_case_contry))

print("The country which has reported highest number of covid 19 cases({}) until 30th of August 2020 is {}".format(max_case,max_case_contry))
def find_contry(contry,date):

    find_row = covid_req[(covid_req['Country']==contry)&(covid_req['Date']==date)]

    deth = find_row['Death'].max()

    case = find_row['Cases'].max()

    print("{} has reported {} covid 19 cases and {} deaths due to covid 19 on {}".format(contry,case,deth,date))
find_contry('India','30-08-2020')
def by_date(date):

    df_by_date = covid_req[(covid_req['Date']==date)&(covid_req['Country']!='International')&(covid_req['Country']!='World')].sort_values(by='Country').reset_index().drop(columns='index')

    print(df_by_date)
by_date('30-08-2020')
covid_vis = DataFrame({'Continent':covid['continent'],'Country':covid['location'],'Date':covid['date'],'Death':covid['total_deaths'],'Cases':covid['total_cases'],'New cases':covid['new_cases'],'New death':covid['new_deaths']})

covid_vis['month'] = to_datetime(covid_vis['Date'],format='%Y-%m-%d')

covid_vis['Date'] = to_datetime(covid_vis['Date'],format='%Y-%m-%d')

covid_vis = covid_vis.fillna(0)
_, ax = subplots(1,1, figsize=(25,15))

covid_vis_cont = covid_vis[covid_vis['Country']=='India']

covid_vis_cont['month'] = to_datetime(covid_vis_cont['month'],format='%Y-%m-%d')

ax.plot(covid_vis_cont.month,

        covid_vis_cont.Cases,  

        linestyle='',

        marker='o'

        

       )

xticks(fontsize=22)

yticks(fontsize=22)

xlabel('Date')

ylabel('Cases')

title('Covid 19 cases vs date',fontsize=22)
_, ax = subplots(1,1, figsize=(25,15))

ax.plot(covid_vis_cont.month,

        covid_vis_cont['New cases'],  

        linestyle='',

        marker='o'

        

       )

xticks(fontsize=22)

yticks(fontsize=22)

xlabel('Date')

ylabel('Cases')

title('Covid 19 New cases vs date',fontsize=22)
_, ax = subplots(1,1, figsize=(25,15))

ax.plot(covid_vis_cont.month,

        covid_vis_cont.Death,  

        linestyle='',

        marker='o'

        

       )

xticks(fontsize=22)

yticks(fontsize=22)

xlabel('Date')

ylabel('Deaths')

title('Covid 19 deaths vs date',fontsize=22)
_, ax = subplots(1,1, figsize=(25,15))

ax.plot(covid_vis_cont.month,

        covid_vis_cont['New death'],  

        linestyle='',

        marker='o'

        

       )

xticks(fontsize=22)

yticks(fontsize=22)

xlabel('Date')

ylabel('Deaths')

title('Covid 19 deaths vs date',fontsize=22)
_, ax = subplots(1,1, figsize=(25,15))

ax.plot(covid_vis_cont.month,

        covid_vis_cont['New death'],   

        label='daeth'

       )

ax.plot(covid_vis_cont.month,

        covid_vis_cont['New cases'], 

        label='cases'

       )

xticks(fontsize=22)

yticks(fontsize=22)

ax.legend()

xlabel('Date')

ylabel('Cases')

title('Covid 19 deaths vs date',fontsize=22)
_, ax = subplots(1,1, figsize=(25,15))

ax.plot(covid_vis_cont['New cases'],

        covid_vis_cont['New death'],  

        linestyle='',

        marker='o'

        

       )

xticks(fontsize=22)

yticks(fontsize=22)

xlabel('Cases')

ylabel('Death')

title('Covid 19 deaths vs date',fontsize=22)
#data of cases

covid_india = covid[(covid['location']=='India')&(covid['date']!='2019-12-31')]

covid_model = covid_india[['total_cases','date']]

covid_model.reset_index(inplace= True)

covid_model = covid_model.drop(columns='index')

covid_model['date'] = to_datetime(covid_model['date'], format='%Y-%m-%d')

covid_model.index = covid_model.date

covid_model = covid_model.drop(columns='date')

covid_model = covid_model.fillna(0)



#data of deaths

covid_deth = covid_india[['total_deaths','date']]

covid_deth['date'] = to_datetime(covid_deth['date'], format='%Y-%m-%d')

covid_deth.reset_index(inplace= True)

covid_deth = covid_deth.drop(columns='index')

covid_deth.index = covid_deth.date

covid_deth = covid_deth.drop(columns='date')

covid_deth = covid_deth.fillna(0)



train,test = train_test_split(covid_model, test_size=.3, shuffle=False)

def graph(pred,col,titl):

    _, ax = subplots(1,1, figsize=(25,15))

    ax.plot(test.index,

            test.total_cases,

            label='test',

            

           )

    ax.plot(test.index,

            pred[col],

            label='predicted'

           )

    ax.fontsize=22

    title(titl,fontsize=22)

    ax.legend()

    xticks(fontsize=22)

    yticks(fontsize=22)

rms = []

def rms_find(df,pred,title):

    rms_finded = sqrt(mean_squared_error(df['total_cases'], df[pred]))

    rms_sub = [title,rms_finded]

    if rms_sub not in rms:

        rms.append(rms_sub)

    print('The root mean squared error fot {} is {}'.format(title,rms_finded))
#copying test dataset to the variable tst_nive

tst_nive = test.copy()

#creating an array of the repoted cases for finding the last observation

lst = asarray(train.total_cases)

last_obsr = lst[len(lst)-1]

#adding the predicted value to the column of the dataset

tst_nive['naive'] = last_obsr

#plotting the graph for comparison

graph(tst_nive,'naive','Naive forecast')

rms_find(tst_nive,'naive','Naive')
tst_avrg = test.copy()

avg_obsr = train['total_cases'].mean()

tst_avrg['simple_average'] = avg_obsr

graph(tst_avrg,'simple_average','Simple Average Forecast')

rms_find(tst_avrg,'simple_average','Simple average')
tst_mvng_avrg = test.copy()

mvng_avg_obsr = train['total_cases'].rolling(60).mean().iloc[-1]

tst_mvng_avrg['mvng_average'] = mvng_avg_obsr

graph(tst_mvng_avrg,'mvng_average','Moving Average Forecast')

rms_find(tst_mvng_avrg,'mvng_average','Moving Average')
tst_ses = test.copy()

ses_arr = asarray(train['total_cases'])

ses_fit = SimpleExpSmoothing(ses_arr).fit(smoothing_level=.8,optimized=False)

tst_ses['SES_pred'] = ses_fit.forecast(len(test)) 

graph(tst_ses,'SES_pred','Simple Exponential Smoothing Forecast')

rms_find(tst_ses,'SES_pred','Simple Exponential Smoothing')
sm.tsa.seasonal_decompose(train.total_cases).plot()

result = sm.tsa.stattools.adfuller(train.total_cases)

show()

tst_holt_lin = test.copy()

holt_lin_arr = asarray(train['total_cases'])

holt_lin_fit = Holt(holt_lin_arr).fit(smoothing_level = .8,smoothing_slope = 1)

tst_holt_lin['holt_linear'] = holt_lin_fit.forecast(len(test)) 

graph(tst_holt_lin,'holt_linear',"Holt's linear trend Forecast")

rms_find(tst_holt_lin,'holt_linear','Holts linear')
tst_holt_add = test.copy()

holt_add_arr = asarray(train['total_cases'])

holt_add_fit = ExponentialSmoothing(holt_add_arr, trend='add').fit()

tst_holt_add['holt_additive'] = holt_add_fit.forecast(len(test)) 

graph(tst_holt_add,'holt_additive',"Holt's Winter trend Forecast-Additive")

rms_find(tst_holt_add,'holt_additive','Holts Additive')
p = d = q = range(0, 4)

#creating an array of different p,d,q values between 0 and 4

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(0,0,0,0)]

params = []

rms_arimas =[]

#checking the rmse value using each p,d,q values by the loop and appending those remse values to a list

for param in pdq:

    params.append(param)  

    for param_seasonal in seasonal_pdq:

        try:

            tst_arima = test.copy()

            tst_arima_model = sm.tsa.statespace.SARIMAX(train.total_cases,

                                                        order=param,

                                                        seasonal_order=param_seasonal,

                                                        enforce_stationarity=False,

                                                        enforce_invertibility=False)

            

            tst_arima_fit = tst_arima_model.fit()

            tst_arima['SARIMA'] = tst_arima_fit.predict(start="2020-06-19", end="2020-08-31", dynamic=True)

            rms_arimas.append(sqrt(mean_squared_error(test.total_cases, tst_arima['SARIMA'])))

        except Exception as e:

            continue  



data_tuples = list(zip(params,rms_arimas))

rms_df = DataFrame(data_tuples, columns=['Parameters','RMS value'])

#finding the p,d,q value with lowest rmse value

minimum = int(rms_df[['RMS value']].idxmin())

parameters = params[minimum]
tst_sarima = test.copy()

tst_sarima_model = sm.tsa.statespace.SARIMAX(train.total_cases, order=parameters,seasonal_order=(0,0,0,0),enforce_stationarity=False,

                                            enforce_invertibility=False)

tst_sarima_fit = tst_sarima_model.fit()

tst_sarima['SARIMA'] = tst_sarima_fit.predict(start="2020-06-19", end="2020-08-31", dynamic=True)

graph(tst_sarima,'SARIMA',"SARIMAX Forecast")

rms_find(tst_sarima,'SARIMA','SARIMAX')
trin_data_prpht = covid_model.copy()

trin_data_prpht['date'] = trin_data_prpht.index

trin_data_prpht = trin_data_prpht.rename(columns={'date':'ds','total_cases':'y'})
tst_prophet = Prophet(changepoint_prior_scale=.25)

tst_prophet.fit(trin_data_prpht)



tst_prpht_futr = tst_prophet.make_future_dataframe(periods=0, freq='D')

tst_prpht_pred = tst_prophet.predict(tst_prpht_futr)



rms_prophet = sqrt(mean_squared_error(trin_data_prpht['y'], tst_prpht_pred['yhat']))

rms_sub_prpht = ['FBProphet',rms_prophet]

rms.append(rms_sub_prpht)
tst_prophet.plot(tst_prpht_pred, xlabel = 'Date', ylabel = 'Cases')
for i in rms:

    print('The root mean squared eroor for {} is {}'.format(i[0],i[1]))
rms_df = DataFrame(rms,columns=['models','rmse'])

rms_df
model = rms_df.iloc[rms_df['rmse'].argmin()]['models']

rmse = rms_df['rmse'].min()

print("The suitbale model for our dataset is {} model with a RMSE value of {}".format(model,rmse))
trin_data_deth = covid_deth.copy()

trin_data_deth['date'] = trin_data_deth.index

trin_data_deth = trin_data_deth.rename(columns={'date':'ds','total_deaths':'y'})





tst_deth_prophet = Prophet(changepoint_prior_scale=.25)

tst_deth_prophet.fit(trin_data_deth)

tst_deth_futr = tst_deth_prophet.make_future_dataframe(periods=0, freq='D')

tst_deth_pred = tst_deth_prophet.predict(tst_deth_futr)

tst_deth_prophet.plot(tst_deth_pred, xlabel = 'Date', ylabel = 'Death')

title('FBProphet Forecast')



rms_deth = sqrt(mean_squared_error(trin_data_deth['y'], tst_deth_pred['yhat']))

print('The root mean squared error fot {} is {}'.format('Death forecast by Prophet model',rms_deth))
def pred_case_prpht(days):

    tst_pred_futr = tst_prophet.make_future_dataframe(periods=days, freq='D')

    tst_pred_fun = tst_prophet.predict(tst_pred_futr)

    test_pred_fun_df = DataFrame({'Date':tst_pred_fun['ds'],'Predicted':tst_pred_fun['yhat']})

    test_pred_fun_df = test_pred_fun_df[test_pred_fun_df['Date']>'2020-08-30']

    print(test_pred_fun_df)

    tst_prophet.plot(tst_pred_fun, xlabel = 'Date', ylabel = 'Cases')

    title('FBProphet Forecast') 

    

def pred_deth_prpht(days):

    tst_deth_futr = tst_deth_prophet.make_future_dataframe(periods=days, freq='D')

    tst_deth_pred = tst_deth_prophet.predict(tst_deth_futr)

    test_deth_fun_df = DataFrame({'Date':tst_deth_pred['ds'],'Predicted':tst_deth_pred['yhat']})

    test_deth_fun_df = test_deth_fun_df[test_deth_fun_df['Date']>'2020-08-30']

    print(test_deth_fun_df)

    tst_deth_prophet.plot(tst_deth_pred, xlabel = 'Date', ylabel = 'Death')

    title('FBProphet Forecast') 
pred_case_prpht(30)
pred_deth_prpht(30)