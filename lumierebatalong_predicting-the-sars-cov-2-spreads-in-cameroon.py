# import package

import matplotlib.pyplot as plt

import seaborn as sns 

import statsmodels as sm

import folium as fl

from pathlib import Path

from sklearn.impute import SimpleImputer

import geopandas as gpd

import mapclassify as mpc

import warnings

import cufflinks
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
%matplotlib inline

pd.options.plotting.backend

#pd.plotting.register_matplotlib_converters()

gpd.plotting.plot_linestring_collection

sns.set()

warnings.filterwarnings('ignore')
covidfile = '/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv'
covid19 = pd.read_csv(covidfile, parse_dates=True)
covid19.head()
covid19.isnull().sum()[covid19.isnull().sum()>0]
covid19.info()
covid19['ObservationDate'] = pd.DataFrame(covid19['ObservationDate'])

covid19['currentCase'] = covid19['Confirmed'] - covid19['Recovered'] - covid19['Deaths']
replace = ['Dem. Rep. Congo', "Côte d'Ivoire", 'Congo', 'United Kingdom', 'China','Central African Rep.',

          'Eq. Guinea','eSwatini','Bosnia and Herz.', 'S. Sudan', 'Dominican Rep.', 'W. Sahara',

          'United States of America']



name = ['Congo (Kinshasa)', 'Ivory Coast', 'Congo (Brazzaville)', 'UK', 'Mainland China', 

        'Central African Republic', 'Equatorial Guinea', 'Eswatini', 'Bosnia and Herzegovina', 'South Sudan',

       'Dominica', 'Western Sahara','US']
covid_data = covid19.drop(columns=['Province/State'])

covid_data = covid_data.replace(to_replace=name, value=replace)

#END Cleaning
covid_data.head()
kmerCovid = covid_data[covid19['Country/Region'] == 'Cameroon']

cameroon = kmerCovid[['ObservationDate', 'Confirmed', 'Deaths', 'Recovered','currentCase']]
cameroon.plot(x='ObservationDate',figsize=(15,5), title='SARS Cov 2 over time in Cameroon')

plt.ylabel('Cummulative')
print('========= COVID-19 Cameroon ==============================')

print("======== Daily report {} ===============\n".format(cameroon.ObservationDate.max()))

print('1- Total Confirmed: {}'.format(cameroon['Confirmed'][cameroon.ObservationDate == cameroon.ObservationDate.max()].values[-1]))

print('2- Total Deaths: {}'.format(cameroon['Deaths'][cameroon.ObservationDate == cameroon.ObservationDate.max()].values[-1]))

print('3- Total Recovered: {}'.format(cameroon['Recovered'][cameroon.ObservationDate == cameroon.ObservationDate.max()].values[-1]))

print('4- Total CurrentCase: {}'.format(cameroon['currentCase'][cameroon.ObservationDate == cameroon.ObservationDate.max()].values[-1]))

print('============================================================')
def piechart(data, xplod, lab, filename):

    ''' This fonction have 3 arguments: data, explode, labels and filename in string '''

    

    kolors = ['red', 'green', 'yellow', 'blue', 'cyan', 'tan', 'wheat']

    

    n= len(lab)

    colrs =  kolors[:n]

    

    fig, ax = plt.subplots(figsize=(8, 3.5))

    

    ax.pie(data, explode=xplod, labels=lab, autopct='%1.1f%%', startangle=270, colors= colrs)

    ax.axis('equal')

    

    fig.suptitle(filename)

    #fig.savefig(filename+'.png', dpi=125)
daily = cameroon[cameroon.ObservationDate == cameroon.ObservationDate.max()]

daily_data = daily.drop(columns='ObservationDate').copy()
daily_data['Deaths'] = (daily_data['Deaths']/daily_data.Confirmed)*100

daily_data['Recovered'] = (daily_data['Recovered']/daily_data.Confirmed)*100

daily_data['currentCase'] = (daily_data['currentCase']/daily_data.Confirmed)*100
x = daily_data.drop(columns='Confirmed')
piechart(x, (0,0,0), x.columns,  'Daily {} SARS Cov 2 in Cameroon'.format( cameroon.ObservationDate.max()))
cameroon.describe()
cameroon.corr()
from scipy import stats, linalg



def partial_corr(C):

    """

    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 

    for the remaining variables in C.

    Parameters

    ----------

    C : array-like, shape (n, p)

        Array with the different variables. Each column of C is taken as a variable

    Returns

    -------

    P : array-like, shape (p, p)

    P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling

        for the remaining variables in C.

    """

    

    C = np.asarray(C)

    p = C.shape[1]

    P_corr = np.zeros((p, p), dtype=np.float)

    for i in range(p):

        P_corr[i, i] = 1

        for j in range(i+1, p):

            idx = np.ones(p, dtype=np.bool)

            idx[i] = False

            idx[j] = False

            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]

            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]



            res_j = C[:, j] - C[:, idx].dot( beta_i)

            res_i = C[:, i] - C[:, idx].dot(beta_j)



            corr = stats.pearsonr(res_i, res_j)[0]

            P_corr[i, j] = corr

            P_corr[j, i] = corr

            

        return P_corr
need_feature = ['Confirmed','currentCase','Recovered','Deaths']

pcoray = cameroon[need_feature].values 

corrpartial = pd.DataFrame(partial_corr(pcoray), columns=need_feature, index=need_feature)

corrpartial.head()
key_feat =  need_feature[0]

fig = plt.figure(figsize=(15, 5))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

cols = ['currentCase', 'Recovered', 'Deaths']

for i in range(1,4):

    ax = fig.add_subplot(1, 3, i)

    ax.scatter(cameroon[key_feat], cameroon[cols[i-1]])

    ax.set_xlabel(key_feat)

    ax.set_ylabel(cols[i-1])

    ax.set_title('Phase Plane')
def polyRegression(x=None, y=None, degree=1):

    """

        params: x array-like predictor

        params: y array-like target

    

    """

    

    # importing libraries for polynomial transform

    from sklearn.preprocessing import PolynomialFeatures

    # for creating pipeline

    from sklearn.pipeline import Pipeline

    # creating pipeline and fitting it on data

    

    # Importing Linear Regression

    from sklearn.linear_model import LinearRegression

    from sklearn.ensemble import RandomForestRegressor

    from sklearn.multioutput import MultiOutputRegressor

    from sklearn.model_selection import GridSearchCV

    

    # many output

    multi_rgr = MultiOutputRegressor(RandomForestRegressor(random_state=0, n_estimators=100))

    

    Input = [('polynomial',PolynomialFeatures(degree=degree)),('modal',multi_rgr)]

    pipe=Pipeline(Input)

    pipe.fit(x.reshape(-1, 1), y)

    

    poly_pred=pipe.predict(x.reshape(-1, 1))

    

    #sorting predicted values with respect to predictor

    pred = []

    for i in range(y.shape[1]):

        sorted_zip = sorted(zip(x, poly_pred[:,i]))

        _, poly_pred1 = zip(*sorted_zip)

        pred.append(poly_pred1)

    

    

    return np.asfarray(pred).T, pipe
prediction, model = polyRegression(x=cameroon[key_feat].values,

                                  y=cameroon[['Deaths','Recovered','currentCase']].values, degree=7)
df_predict = pd.DataFrame(prediction, columns=['Deaths','Recovered','currentCase'],

                          index=cameroon.ObservationDate)
df_predict.head()
fig= plt.figure(figsize=(15.5,5.5))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

cols = ['Recovered', 'currentCase', 'Deaths']

for i in range(1,4):

    ax=fig.add_subplot(1, 3, i)

    ax.scatter(cameroon[key_feat], cameroon[cols[i-1]], s=20, label='Actual')

    ax.plot(cameroon[key_feat], df_predict[cols[i-1]] ,'r', label='Prediction')

    ax.set_xlabel(key_feat,fontsize=16)

    ax.set_ylabel(cols[i-1],fontsize=16)

    ax.legend(loc='best')

    ax.set_title('Polynomial regression phase plane')
from sklearn.metrics import mean_squared_error, mean_absolute_error
for c in cols:

    score = np.sqrt(mean_squared_error(cameroon[c] ,df_predict[c]))

    print('{}: RMSE for Polynomial Regression => {}\n'.format(c, score))
xplot = cameroon.currentCase.copy()

xplot.index = cameroon.ObservationDate

ag = xplot.plot(legend=True,label='Actual', figsize=(15,5))

df_predict.currentCase.plot(legend=True, label='prediction', ax=ag, title='plotting Current Case')

plt.ylabel('cummulative')
yplot = cameroon.Recovered.copy()

yplot.index = cameroon.ObservationDate

ah = yplot.plot(legend=True,label='Actual', figsize=(15,5))

df_predict.Recovered.plot(legend=True, label='prediction', ax=ah, title='plotting Recovered')

plt.ylabel('cummulative')
#importing package

from fbprophet import Prophet
confirm = cameroon[['ObservationDate', key_feat]]
confirm.head(3)
prec = confirm.rename(columns={'ObservationDate':'ds', key_feat:'y'})
prec.head(3)
m = Prophet(interval_width=0.95,changepoint_prior_scale=1.25, yearly_seasonality=False, 

            daily_seasonality=True)

m.fit(prec)
futureDays = m.make_future_dataframe(periods=5)

futureDays.tail(7)
confirmed_forecast = m.predict(futureDays)
confirmed_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
graph = m.plot(confirmed_forecast)

plt.title('Confirmed Cameroon forecasting')
graph1 = m.plot_components(confirmed_forecast)
from fbprophet.diagnostics import cross_validation

from fbprophet.diagnostics import performance_metrics
#for cross validation we are taking the range of our data 

df_cv = cross_validation(m, initial='30 days', period='1 days', horizon = '7 days')

df_cv.head(3)
df_p = performance_metrics(df_cv)

df_p.head(3)
from fbprophet.plot import plot_cross_validation_metric

ufig = plot_cross_validation_metric(df_cv, metric='mape')
pred_data = confirmed_forecast[confirmed_forecast['ds'].isin(futureDays.ds)]
pforecast_lower = pd.DataFrame(model.predict(pred_data.yhat_lower.values.reshape(-1,1)), 

                               columns=['Deaths','Recovered','currentCase'],

                         index=futureDays.ds)
pforecast = pd.DataFrame(model.predict(pred_data.yhat.values.reshape(-1,1)),

                         columns=['Deaths','Recovered','currentCase'],

                         index=futureDays.ds)
pforecast_upper = pd.DataFrame(model.predict(pred_data.yhat_upper.values.reshape(-1,1)), 

                               columns=['Deaths','Recovered','currentCase'],

                         index=futureDays.ds)
# function for plotting

def viewing_forecast(actual=None, lower=None, forecast=None, upper=None, title=None):

    

    plt.figure(figsize=(15, 5.5))

    ax = forecast.plot(color='blue')

    

    nd = len(actual)

    time = forecast.index[:nd]

    ax.scatter(time, actual, label='Actual', color='black')

    

    ax.fill_between(forecast.index, lower, upper, alpha=0.9, color='orange')

    

    a = actual.index.to_list()

    b = actual.index.max()

    n = a.index(b)

    pmax = forecast.max()

    pmin = forecast.min()

    ax.vlines(actual.index[n], actual.min(), pmax, linestyles='dashdot', colors='black',

              label='stop actual')

    

    bbox = dict(boxstyle="round", fc='0.8')

    arrowprops = dict(arrowstyle='->', connectionstyle='angle, angleA=0, angleB=100, rad=10', 

                      facecolor='black')

    

    offset = 72

    

    ax.annotate('Actual-Prediction', (actual.index.max(), actual.max()), xytext=(-2*offset, offset), 

                textcoords='offset points', bbox=bbox, arrowprops=arrowprops)

    

    disp = ax.annotate('Forecasting', (actual.index.max(), actual.max()), xytext=(0.5*offset, -offset),

                textcoords='offset points', bbox=bbox, arrowprops=arrowprops)

    

    

    ax.set_xlabel('Date')

    ax.set_ylabel('Cumulative')

    ax.set_title('{} Africa Forecasting'.format(title))

    plt.legend(loc='best')  
datac = cameroon.set_index('ObservationDate')
viewing_forecast(actual=datac['Deaths'] ,lower=pforecast_lower['Deaths'],

                 forecast=pforecast['Deaths'] , 

                 upper=pforecast_upper['Deaths'] ,

                 title='Deaths')
viewing_forecast(actual=datac['Recovered'],lower=pforecast_lower['Recovered'], 

                 forecast=pforecast['Recovered'] , 

                 upper=pforecast_upper['Recovered'] ,

                 title='Recovered')
viewing_forecast(actual=datac['currentCase'],lower=pforecast_lower['currentCase'], 

                 forecast=pforecast['currentCase'] , 

                 upper=pforecast_upper['currentCase'] ,

                 title='CurrentCase')
def determinate_beta_gamma_delta(data=None):

    '''

        this function compute transmission rate, recovered rate and fatalities rate over time

        params: data

        return: beta, gamma, delta

    '''

    

    beta = []

    gamma = []

    delta = []

    

    for t in range(len(data.ObservationDate.values)):

        

        x = data.Confirmed.iloc[t]

        y = data.Deaths.iloc[t]

        z = data.Recovered.iloc[t]

        w = data.currentCase.iloc[t]

        

        if x == 0.0:

            beta.append(0)

            gamma.append(0)

            delta.append(0)

        else:

            beta_t = w/x

            gamma_t = z/x

            delta_t = y/x

            

            beta.append(beta_t)

            gamma.append(gamma_t)

            delta.append(delta_t)

            

    return np.array(beta), np.array(gamma), np.array(delta)        
transmission, recovery, fatality = determinate_beta_gamma_delta(data=cameroon)
parameter_dynamic = pd.DataFrame()

parameter_dynamic['beta'] = transmission

parameter_dynamic['gamma'] = recovery

parameter_dynamic['delta'] = fatality

parameter_dynamic.index = cameroon.ObservationDate
parameter_dynamic.head()
def find_R0(data=None):

    '''

        This function compute R0 over time

        params: data

        return: R0

    '''

    return data.beta.values/(data.gamma.values + data.delta.values)
#Compute R0

parameter_dynamic['R0'] = find_R0(data=parameter_dynamic)

n_max = len(parameter_dynamic.index)
parameter_dynamic[['beta','gamma','delta']].plot(figsize=(15,7))

plt.hlines(0.4, 0, n_max, linestyles='dashdot', label='lower buffer zone')

plt.hlines(0.56, 0, n_max, linestyles='dashdot', label='upper buffer zone')

plt.hlines(0.48, 0, n_max, linestyles='dashdot', label='middle buffer zone')

plt.legend(loc='best')

plt.title('parameter dynamics for spreading of SARS Cov 2 in Cameroon')
# Plot R0

parameter_dynamic['R0'].plot(figsize=(15,7))

plt.hlines(1, 10, n_max, linestyles='dashdot', label='Threshold(R0 = 1)')

plt.legend(loc='best')

plt.title('ratio reproductive number for SARS Cov 2 in Cameroon')
def growth_rate(data=None):

    """

        This function compute a growth rate of one variable

        params: data

        return: growth rate x

    

    """

    x = []

    x.append(0)

    for i in range(data.shape[0]-1):

        a = data.iloc[i+1]-data.iloc[i]

        b = a/data.iloc[i]

        x.append(b)

        

    return np.array(x)
growth_rate_currentCase = pd.DataFrame(growth_rate(data=cameroon.currentCase), columns=['currentCase'],

                                       index=cameroon.ObservationDate)
growth_rate_currentCase.plot(figsize=(15,7))

plt.hlines(0, 0, n_max, linestyles='dashdot', label='0')

plt.legend(loc='best')

plt.title('Infective growth rate for Covid 19 disease in Cameroon')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.ar_model import AutoReg, ar_select_order
fg = plt.figure(figsize=(15, 5))

fg.subplots_adjust(hspace=0.4, wspace=0.4)

axis1 = fg.add_subplot(1, 2, 1)

axis2 = fg.add_subplot(1, 2, 2)

_ = plot_acf(growth_rate_currentCase.values, ax=axis1)

_ = plot_pacf(growth_rate_currentCase.values, ax=axis2)
sel_seasonal = ar_select_order(growth_rate_currentCase, 13,  glob=True, seasonal=True)

sel_seasonal.ar_lags
res_seasonal = sel_seasonal.model.fit()

res_seasonal.summary()
_ =  res_seasonal.plot_predict(start='04-06-2020', end='05-12-2020', figsize=(15,5))
_ = res_seasonal.plot_diagnostics(lags=30, figsize=(15,10))
fig0 = plt.figure(figsize=(15, 5))

fig0.subplots_adjust(hspace=0.4, wspace=0.4)

axis01 = fig0.add_subplot(1, 2, 1)

axis02 = fig0.add_subplot(1, 2, 2)

_ = plot_acf(parameter_dynamic.iloc[16:,-1], ax=axis01)

_ = plot_pacf(parameter_dynamic.iloc[16:,-1], ax=axis02)
r0 = ar_select_order(parameter_dynamic.iloc[16:,-1], 13,  glob=True, seasonal=True)

r0.ar_lags
res_r0 = r0.model.fit()

res_r0.summary()
_ =  res_r0.plot_predict(start='04-30-2020', end='05-12-2020', figsize=(15,5))
_ = res_r0.plot_diagnostics(lags=30, figsize=(15,10))