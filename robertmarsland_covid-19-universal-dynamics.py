from sklearn.linear_model import LinearRegression

from matplotlib.backends import backend_pdf as bpdf

from covid_functions import *



#Load data from JH repository

base_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'

cases_global = format_JH(base_url+'time_series_covid19_confirmed_global.csv',['Lat','Long'],['Country/Region','Province/State'])

deaths_global = format_JH(base_url+'time_series_covid19_deaths_global.csv',['Lat','Long'],['Country/Region','Province/State'])

cases_US = format_JH(base_url+'time_series_covid19_confirmed_US.csv',['UID','iso2','iso3','code3','FIPS','Lat','Long_','Combined_Key'],['Country_Region','Province_State','Admin2'])

deaths_US = format_JH(base_url+'time_series_covid19_deaths_US.csv',['UID','iso2','iso3','code3','FIPS','Lat','Long_','Combined_Key','Population'],['Country_Region','Province_State','Admin2'])

cases_US = cases_US.T.groupby(level=[0,1]).sum().T

deaths_US = deaths_US.T.groupby(level=[0,1]).sum().T

#Join US and global data into single table

cases = cases_global.join(cases_US)

deaths = deaths_global.join(deaths_US)



#Load full prediction tables, with confidence bounds

pred_date = datetime(2020,4,15)

predictions_deaths = format_predictions('/kaggle/input/covid19-april-15-predictions/predictions_deaths_apr15.csv')

predictions_cases = format_predictions('/kaggle/input/covid19-april-15-predictions/predictions_cases_apr15.csv')
country = 'Spain'

region = 'NaN'

daymax = 50



fig,ax=plt.subplots(2,figsize=(10,12),sharex=True)

t = pd.to_datetime([data.index[0]+timedelta(days=k) for k in range(daymax)])



data = deaths[country,region].copy()

data = data.loc[data>5]

pred = predictions_deaths.loc[country,region]

tau = ((t-pred['th'])/timedelta(days=1))/pred['sigma']



ax[0].plot(data.index,data.values,marker='o',label='data')

ax[0].plot(t,pred['Nmax']*norm.cdf(tau),label='prediction')

ax[0].set_yscale('log')

ax[0].set_title(', '.join([country,region]))

ax[0].plot([pred_date,pred_date],[data.min(),3*data.max()],'k--',label='prediction date')

ax[0].legend()

ax[0].set_ylim((10,None))

ax[0].set_ylabel('Cumulative fatalities')



data = cases[country,region].copy()

data = data.loc[data>5]

pred = predictions_cases.loc[country,region]

tau = ((t-pred['th'])/timedelta(days=1))/pred['sigma']



ax[1].plot(data.index,data.values,marker='o',label='data')

ax[1].plot(t,pred['Nmax']*norm.cdf(tau),label='prediction')

ax[1].set_yscale('log')

ax[1].set_ylabel('Cumulative cases')

ax[1].set_ylim((10,None))

ax[1].plot([pred_date,pred_date],[data.min(),3*data.max()],'k--',label='prediction date')

plt.show()