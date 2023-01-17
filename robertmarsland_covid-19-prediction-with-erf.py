from covid_functions import *

folder = '/kaggle/input/covid19-global-forecasting-week-3/'



submission = pd.read_csv(folder+'submission.csv',index_col=0)

test = pd.read_csv(folder+'test.csv',index_col=0)

test['Date'] = pd.to_datetime(test['Date'],format='%Y-%m-%d')

test['Province_State'] = test['Province_State'].fillna(value='NaN')

data_cases = format_kaggle(folder,'ConfirmedCases')

data_deaths = format_kaggle(folder,'Fatalities')
params_cases = fit_all(data_cases,plot=True,ylabel='Confirmed Cases',p0=1e2,prior=(8,500))

params_cases.to_csv('params_cases.csv')
params_deaths = fit_all(data_deaths,plot=True,ylabel='Fatalities',p0=10,prior=(8,200))

params_deaths.to_csv('params_deaths.csv')
for item in submission.index:

    #Extract location and time

    country = test.loc[item,'Country_Region']

    province = test.loc[item,'Province_State']

    t_abs = test.loc[item,'Date']

    t = (t_abs-tref)/pd.to_timedelta(1,unit='days')

    

    #Predict cases

    th, logK, sigma = params_cases[['th','logK','sigma']].loc[country,province]

    if not np.isnan(th):

        tau = (t-th)/(np.sqrt(2)*sigma)

        submission.loc[item,'ConfirmedCases'] = np.exp(logK)*(1+erf(tau))/2

    else:

        if t_abs in data_cases.index:

            submission.loc[item,'ConfirmedCases'] = data_cases[country,province].loc[t_abs]

        else:

            submission.loc[item,'ConfirmedCases'] = data_cases[country,province].max()



    #Predict fatalities

    th, logK, sigma = params_deaths[['th','logK','sigma']].loc[country,province]

    if not np.isnan(th):

        tau = (t-th)/(np.sqrt(2)*sigma)

        submission.loc[item,'Fatalities'] = np.exp(logK)*(1+erf(tau))/2

    else:

        if t_abs in data_deaths.index:

            submission.loc[item,'Fatalities'] = data_deaths[country,province].loc[t_abs]

        else:

            submission.loc[item,'Fatalities'] = data_deaths[country,province].max()

    

submission.to_csv('submission.csv')
country = 'Italy'

province = 'NaN'

info = test.reset_index().set_index(['Country_Region','Province_State']).sort_index().loc[country,province]

t = info['Date']

idx = info['ForecastId']

plt.plot(data_cases.index,data_cases[country,province],'o',label='Data')

plt.plot(t,submission['ConfirmedCases'].loc[idx],label='Prediction')

plt.gca().set_ylabel('Confirmed Cases')

plt.gca().set_yscale('log')

plt.show()



plt.plot(data_deaths.index,data_deaths[country,province],'o',label='Data')

plt.plot(t,submission['Fatalities'].loc[idx],label='Prediction')

plt.gca().set_ylabel('Fatalities')

plt.gca().set_yscale('log')

plt.show()
country='Denmark'

province='NaN'



daymin=-50

daymax=20



t = pd.to_datetime([datetime.today()+timedelta(days=k) for k in range(daymin,daymax)])

ax = plot_predictions(data_cases[country,province],params_cases[['th','logK','sigma']].loc[country,province].values,t_pred = t)

ax.set_ylabel('Confirmed Cases')

ax.set_title(country)

plt.show()



ax = plot_predictions(data_deaths[country,province],params_deaths[['th','logK','sigma']].loc[country,province].values,t_pred = t)

ax.set_ylabel('Fatalities')

ax.set_title(country)

plt.show()