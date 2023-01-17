# imports

import plotly.express as px

import plotly.graph_objects as go

from scipy import stats, special

import numpy as np

import matplotlib.pyplot as plt

from scipy import interpolate

import json



import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import lognorm

from scipy.optimize import curve_fit

import string

from scipy.integrate import quad



from sklearn import mixture

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.linear_model import SGDRegressor, LinearRegression, Lasso, Ridge, LogisticRegression

from sklearn.base import clone

from sklearn.pipeline import Pipeline, make_pipeline





# https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf

# https://www.apsnet.org/edcenter/disimpactmngmnt/topc/EpidemiologyTemporal/Pages/ModellingProgress.aspx

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
# https://stackoverflow.com/questions/56176657/how-to-fit-data-with-log-normal-cdf

# y = a * log10(b * x + h) + k



def fgaussian(x,a,x0,sigma):

    return a*np.exp(-(x-x0)**2/(2*sigma**2))



def ferf(x, a, b, c, d):

    return d + 0.5*c*(1 + special.erf(a*(x-b)))



def ftriangle (z, a, b, c):

    y = np.zeros(z.shape)

    y[z <= a] = 0

    y[z >= c] = 0

    first_half = np.logical_and(a < z, z <= b)

    y[first_half] = (z[first_half]-a) / (b-a)

    second_half = np.logical_and(b < z, z < c)

    y[second_half] = (c-z[second_half]) / (c-b)

    return y



def flog10(x, a, b, h, k):

    return a*np.log10(b*x + h) + k



def fsigmoid(x, a, b):

    return 1.0 / (1.0 + np.exp(-a*(x-b)))



def fexp(t, a, b, alpha):

    return a - b * np.exp(-alpha * t)



def fexp2(x, a, b, c, d):

    return a * np.exp(b * x) + c * np.exp(d * x)



def fpow(x, a, b, c):

    return c * pow(a, x) + b



def flog(x, a, b, c):

    return c * np.log(x * a) + b



def poly_transform_to_ln(x):

    """polycyclic disease model transform x is within 0.11111 to 0.99999"""

    return np.log( x / (1-x) )



def fline(x, a, b):

    """polycyclic disease model transform x is within 0.11111 to 0.99999"""

    return a*x + b



def get_country_sum(df, country='Italy'):

    df_country = df[df['Country_Region']==country]

    df_country = df_country.groupby('Date').agg({'ConfirmedCases':['sum'], 'Fatalities':['sum']}).reset_index()

    df_country.columns = ['Date','ConfirmedCases','Fatalities']

    df_country['Country_Region'] = country

    

    return df_country



def fit(fun,x,y,params=None):

#     params=[100,80,5]

    return curve_fit(fgaussian,x,y,maxfev=2000,method='lm',p0=params)

#     return curve_fit(fun,x,y,maxfev=2000,method='trf',loss='huber',p0=params)



def country_slice(df, country='China', province=None):

    if province is None or pd.isna(province):

        return df[(df['Country_Region']==country) & (pd.isna(df['Province_State']) == True) ]

    else:

        return df[(df['Country_Region']==country) & (df['Province_State']==province)]



    

def do_fit_fun(x, y, fun=fgaussian, params=None):

    x = np.array(x)

    y = np.array(y)

    

    filter_nan_inf = lambda y: (np.isnan(y)==False) & (np.isinf(y)==False)

    x = x[filter_nan_inf(y)]

    y = y[filter_nan_inf(y)]

    

#     print(f'x={x}')

#     print(f'y={y}')

#     print(f'params={params}')

#     print(f'fun={fun}')

    

    if params:

#         popt, pcov = curve_fit(fun, x, y, p0=params)

        popt, pcov = fit(fun, x, y, params=params)

    else:

#         popt, pcov = curve_fit(fun, x, y)

        popt, pcov = fit(fun, x, y)

    

    perr = mean_squared_error(y, fun(x,*popt))



    label = 'fit: ' + ' '.join([f'{a} = %.2f' for a in string.ascii_lowercase[:len(popt)]]) % tuple(popt)

#     label = 'fit: a = %.2f b = %.2f' % tuple(popt) #locl = %.2f scale = %.2f

                                    

#     print(f'noparams! popt={popt}, pcov={pcov}, err={perr}, label={label}')



    return popt, perr



def do_fit(df, y_label, gaussian_params=[1000, 60, 2]):

    y = df[y_label]

    x = list(range(1,len(y)+1))

    

    ret = {'Country_Region':df['Country_Region'].drop_duplicates().tolist()[0], 

           'Province_State':df['Province_State'].drop_duplicates().tolist()[0],

           'total_cases':y.sum(), 'max_cases':y.max()

          }

    try:

        popt, perr = do_fit_fun(x, y, fun=fgaussian, params=gaussian_params)

        ret['fgaussian_popt_a'] = popt[0]

        ret['fgaussian_popt_x0'] = popt[1]

        ret['fgaussian_popt_sd'] = popt[2]

        ret['fgaussian_perr'] = perr

    except:

        ret['fgaussian_popt_a'] = None

        ret['fgaussian_popt_x0'] = None

        ret['fgaussian_popt_sd'] = None

        ret['fgaussian_perr'] = None

        

    return ret





def nyc_scale(df_fit_confirmed, country, province=None, ref_country='US', ref_province='New York', metric='DailyConfirmedCases'):

# def nyc_scale(df_fit_confirmed, country, province=None, ref_country='France', ref_province=None, metric='DailyConfirmedCases'):

# def nyc_scale(df_fit_confirmed, country, province=None, ref_country='Italy', ref_province=None, metric='DailyConfirmedCases'):

    """

    Use the NYC gaussian distribution as reference and scale by max_cases

    

    Returns popt and scaling_factor

    """

#     nyc = country_slice(df_fit_confirmed,'US','New York')  # NYC is the index

    nyc = country_slice(df_fit_confirmed, ref_country, ref_province)

    nyc_opts = np.array(nyc[['fgaussian_popt_a','fgaussian_popt_x0','fgaussian_popt_sd']])[0]

    print(f"nyc_opts={nyc_opts}")



#     country = 'Australia'

#     province = 'Tasmania'

    # province = 'Virginia'

    # province = 'Massachusetts'

    tmp = country_slice(df_fit_confirmed,country,province)

    if metric == 'DailyConfirmedCases':

        scaling_factor = np.array(tmp['max_cases']) /  np.array(nyc['max_cases'])

    else:

        scaling_factor = 2.  # DailyFatalities is small, so make it bigger

        

#     print(f"province={province}, tmp.shape={tmp.shape}, province_max={tmp['max_cases']}, nyc_max={nyc['max_cases']}, scaling_factor={scaling_factor}")



    # new_york_popts = [5093.862731, 63.169327, 3.791986 ]



    f = fgaussian

    tmp = country_slice(train, country=country, province=province)

    y = np.array(tmp[ metric ])

    x = np.array(list(range(len(y))))

    # popt, pcov = curve_fit(f, x, y, p0=[100, 60, 1])

    popt = nyc_opts

#     print(popt)

#     plt.plot(x,y,'b.')



#     plt.plot(x,scaling_factor*f(x,*popt),'r.')

    

    return popt, scaling_factor[0]





def get_popts_scaling_factory(fits, df_fit, country, province, x, fun=fgaussian, min_cases=50, max_x0=150, max_a=1000, metric='DailyConfirmedCases'):

    """

    Handle countries with special cases.

    

    - Colombia was predicted to have a mean of over 130 which is too far out.

    - Zimbabwe has fewer than 10 cases

    

    """

    scaling_factor = 1.

    try:

        popts = np.array(fits[['fgaussian_popt_a','fgaussian_popt_x0','fgaussian_popt_sd']])[0]

#         if pd.isna(popts[0]) or (np.array(fits['max_cases']) < min_cases) or (popts[1] > 50 and popts[1] < 70):

#             raise Exception

#         if pd.isna(popts[0]) or (metric=='DailyConfirmedCases' and (np.array(fits['max_cases']) < min_cases) or (np.array(fits['fgaussian_popt_x0']) > max_x0)):

        if pd.isna(popts[0]) or (metric=='DailyConfirmedCases' and (np.array(fits['max_cases']) < min_cases) ):

            print(f'found no popts or cases less than {min_cases}, OLD scaling factor {scaling_factor}, OLD popts={popts}')

            popts, scaling_factor = nyc_scale(df_fit, country, province, metric=metric)

            print(f'NEW scaling factor {scaling_factor}, NEW popts={popts}')

        elif (np.array(fits['fgaussian_popt_x0']) > max_x0):

            print(f'mean is over {max_x0}, OLD scaling factor {scaling_factor}, OLD popts={popts}')

            popts, scaling_factor = nyc_scale(df_fit, country, province, metric=metric)

            print(f'NEW scaling factor {scaling_factor}, NEW popts={popts}')

        elif (metric=='DailyConfirmedCases' and (np.array(fits['max_cases']) < max_a) and (np.array(fits['fgaussian_popt_a']) > max_a)):

            print(f'cases less than {max_a} and amplitude is over {max_a}, OLD scaling factor {scaling_factor}, OLD popts={popts}')

            popts, scaling_factor = nyc_scale(df_fit, country, province, metric=metric)

            print(f'NEW scaling factor {scaling_factor}, NEW popts={popts}')

        else:

#             x = np.array(list(range(tmp.shape[0])))

            if metric == 'DailyConfirmedCases':

                scaling_factor = np.array(fits['max_cases']) / np.array(max(fun(x,*popts)))

            else:

                scaling_factor = 2.  # DailyFatalities is small, so make it bigger



    except Exception as e:

        print(f'Error {e}')

        popts, scaling_factor = nyc_scale(df_fit, country, province)

    

    # make it wider

    if metric=='DailyConfirmedCases':

        scaling_factor = 1.1 * scaling_factor

        sigma_scaling_factor = 2.5

        popts[2] = sigma_scaling_factor * popts[2]

        print(f'Scaling factors sigma={sigma_scaling_factor}, amplitude={scaling_factor})')

    

    return popts, scaling_factor
# plotting functions



def plot_fit(x, y, params=None, fun=fsigmoid, title=''):

#     params = [ 0.21766133, 18.87933821]

    popt, perr = do_fit_fun(x, y, fun, params)         

    

    plt.plot(x,y,'r.')

    plt.title(title)

    plt.plot(x, fun(x, *popt), 'k--', label=label)

    plt.legend(loc = 'lower right')

    # plt.xscale('log')

#     plt.show()



    print(f'RMSE={mean_squared_error(y, fun(x,*popt))}')

    

    return plt



def plot_fit_daily(df, country='China', transform=None, fun=fpow, params=None, y_label='DailyConfirmedCases'):

#     df = get_country_sum(df, country)

#     df = df[df['Country_Region']==country]

#     y = df[y_label] / max(df[y_label])

    y = df[y_label]

    x = list(range(1,len(y)+1))

    # y = np.array(y)

    # x = np.array(x)



    if transform:

        y = transform(y)

        

    plot_fit(x, y, fun=fun, title=country, params=params)

    print(f'X={x}, y={y}')

    

    

def plot_fit_country(train, df_fit, country='China', province=None, fun=fgaussian, metric='DailyConfirmedCases', scaling_factor=None):

    # plot gausian parameters

    # country = 'Canada'

    # province = 'Quebec'

    # country = 'China'

    # province = 'Hubei'

#     country = 'US'

#     province = 'New Mexico'

    # country = 'Australia'

    # province = 'South Australia'

    # province = 'Washington'

    # province = 'Louisiana'

    # province = 'Florida'



    # province = 'California'

    tmp = country_slice(train, country, province)

    # print(tmp)

    fits = country_slice(df_fit, country, province)

    

#     try:

#         scaling_factor = np.array(fits[['scaling_factor']])[0]

#     except:

#         scaling_factor = 1.

        

    x = np.array(list(range(tmp.shape[0])))

    popts, _scaling_factor = get_popts_scaling_factory(fits, df_fit, country, province, x, metric=metric)

    

    if scaling_factor is None:

        scaling_factor = _scaling_factor

        

    print(f'{province} {country} scaling_factor={scaling_factor} popts={popts}')

    

    y = np.array(tmp[ metric ])

    x = np.array(list(range(len(y))))

    

    y_pred = scaling_factor * fun(x,*popts)

    

    plt.plot(x,y,'b-+', label='Actual')

    # popts, pcov = curve_fit(fgausian,x,y, p0=[400,23,2])

#     popts = fits[['fgaussian_popt_a','fgaussian_popt_x0','fgaussian_popt_sd']].transpose().iloc[:,0].tolist()

    plt.plot(x,y_pred,'r.', label='Predicted')

    plt.title(f'Actual vs fitted gaussian for {province} {country}')

    plt.ylabel(metric)

    plt.xlabel('Time')

    ax = plt.gca()

    ax.legend()
# load datasets

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

train['Date'] = pd.to_datetime(train['Date'])

train = train.sort_values(by=['Country_Region','Province_State','Date'])

print(f'train min_date={min(train["Date"])}, max_date={max(train["Date"])}')

train['DailyConfirmedCases'] = train['ConfirmedCases'].diff()

train['DailyFatalities'] = train['Fatalities'].diff()

train_bak = train  # make a backup



# replace negatives with a 0

filter = train['DailyConfirmedCases']<0

train.loc[filter,'DailyConfirmedCases'] = 0

train.loc[filter,'DailyFatalities'] = 0

filter = np.isnan(train['DailyConfirmedCases'])

train.loc[filter,'DailyConfirmedCases'] = 0

train.loc[filter,'DailyFatalities'] = 0



train.to_csv('train_daily.csv',index=False)



test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

test['Date'] = pd.to_datetime(test['Date'])

print(f'test min_date={min(test["Date"])}, max_date={max(test["Date"])}')

test



# filter training data upto the test date

train = train[train['Date']<min(test['Date'])]

print(f"max_train={max(train['Date'])}")



min(test['Date'])



submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')

submission



# countries with a Province

train[train['Province_State'].isna()==False]
tmp = country_slice(train,'Korea, South')

y = np.array(tmp['DailyConfirmedCases'])

x = np.array(list(range(len(y))))

popts, pcov = curve_fit(fgaussian,x,y,p0=[10,50,100])

print(popts)

plt.plot(x,y)

plt.plot(x,fgaussian(x,*popts)*1.3)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html

tmp = country_slice(train,'Algeria')

y = np.array(tmp['DailyConfirmedCases'])

x = np.array(list(range(len(y))))

# popts, pcov = curve_fit(fgaussian,x,y,absolute_sigma=False,maxfev=2000,method='trf',loss='huber',p0=[100,80,5])

popts, pcov = fit(fgaussian,x,y,params=[100,80,5])

# popts, pcov = curve_fit(fgaussian,x,y,absolute_sigma=False,maxfev=2000,method='lm',p0=[100,60,1])

print(popts)

plt.plot(x,y)

plt.plot(x,fgaussian(x,*popts))
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html

tmp = country_slice(train,'Algeria')

y = np.array(tmp['DailyFatalities'])

x = np.array(list(range(len(y))))

# popts, pcov = curve_fit(fgaussian,x,y,absolute_sigma=False,maxfev=2000,method='trf',loss='huber',p0=[100,80,5])

# popts, pcov = fit(fgaussian,x,y,params=[1,80,5])

popts, pcov = curve_fit(fgaussian,x,y,absolute_sigma=False,maxfev=2000,method='lm',p0=[100,60,1])

print(popts)

plt.plot(x,y)

plt.plot(x,fgaussian(x,*popts))
tmp = country_slice(train,'Korea, South')

y = np.array(tmp['DailyConfirmedCases'])

x = np.array(list(range(len(y))))

popts, pcov = curve_fit(fgaussian,x,y,absolute_sigma=True)

print(popts)

plt.plot(x,y)

plt.plot(x,fgaussian(x,*popts))
tmp = country_slice(train,'Korea, South')

y = np.array(tmp['DailyFatalities'])

x = np.array(list(range(len(y))))

popts, pcov = curve_fit(fgaussian,x,y,p0=[1,60,1])

print(popts)

plt.plot(x,y)

plt.plot(x,fgaussian(x,*popts))
tmp = country_slice(train,'China', province='Hubei')

y = np.array(tmp['DailyFatalities'])

x = np.array(list(range(len(y))))

popts, pcov = curve_fit(fgaussian,x,y,p0=[1,60,1])

print(popts)

plt.plot(x,y)

plt.plot(x,fgaussian(x,*popts)*2)
# plot_fit_country(train, df_fit_confirmed, country='China', province='Shanghai')
# plot_fit_country(train, df_fit_confirmed, country='US', province='New Mexico')
# # covid

# import requests

# import io



# def get_df_from_url(url):

#     s = requests.get(url).content

#     return pd.read_csv(io.StringIO(s.decode('utf-8')))



# covid_url_prefix = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/'

# df_covid_confirmed = get_df_from_url(covid_url_prefix + 'time_series_covid19_confirmed_global.csv')

# df_covid_deaths = get_df_from_url(covid_url_prefix + 'time_series_covid19_deaths_global.csv')

# df_covid_recovered = get_df_from_url(covid_url_prefix + 'time_series_covid19_recovered_global.csv')

tmp = train[train['Country_Region']=='Korea, South']

plt.plot(list(range(tmp.shape[0])), tmp['DailyConfirmedCases'])

plt.title('Korea, South daily confirmed cases')
tmp = train[(train['Country_Region']=='China')&(train['Province_State']=='Hubei')]

plt.plot(list(range(tmp.shape[0])), tmp['DailyConfirmedCases'])

plt.title('Hubei, China daily confirmed cases')

# perform curve fitting

df_country_state_max = train.groupby(['Country_Region','Province_State']).max().reset_index()

df_country_state_max

fit_confirmed = []

fit_fatality = []

# confirmed_cases_gaussian_params = [1000, 70, 60] # initial params determined empirically 

# fatalities_gaussian_params = [1,60,40]

confirmed_cases_gaussian_params = [1000, 70, 40] # initial params determined empirically 

fatalities_gaussian_params = [10,70,20]

print(f'Confirmed cases Gaussian params={confirmed_cases_gaussian_params}, fatalities_gaussian_params={fatalities_gaussian_params}')

country_state_df = train[['Country_Region','Province_State']].drop_duplicates()

# for i, row in list(df_country_state_max.iterrows())[:20]:

for i, row in list(country_state_df.iterrows()):

    country = row[0]

    province = row[1]

    if province is None or pd.isna(province):

        tmp = train[(train['Country_Region']==country)&(pd.isna(train['Province_State']))]

    else:

        tmp = train[(train['Country_Region']==country)&(train['Province_State']==province)]

#     print(f'tmp={tmp}')

    # filter out cumulative days with 0 cases so that all the graphs are shifted to the left and start the same

    # then compare the fitted means

#     tmp = tmp[tmp['ConfirmedCases']>=5]

    if tmp.shape[0] == 0:

        print(f'No confirmed cases found for {country} {province}')

        continue

    ret = do_fit(tmp, y_label='DailyConfirmedCases', gaussian_params=confirmed_cases_gaussian_params)

    fit_confirmed.append(ret)

    ret = do_fit(tmp, y_label='DailyFatalities', gaussian_params=fatalities_gaussian_params)

    fit_fatality.append(ret)



df_fit_confirmed = pd.DataFrame(fit_confirmed)

df_fit_fatality = pd.DataFrame(fit_fatality)



df_fit_confirmed.to_csv('fit_confirmed.csv')

df_fit_fatality.to_csv('fit_fatality.csv')



df_fit_confirmed
print(f"df_fit_confirmed error={np.sum(df_fit_confirmed['fgaussian_perr'])}")

print(f"df_fit_fatality error={np.sum(df_fit_fatality['fgaussian_perr'])}")
print(f"df_fit_confirmed={df_fit_confirmed.shape}")

print(f"nan={df_fit_confirmed[np.isnan(df_fit_confirmed['fgaussian_popt_x0'])].shape}")

df_fit_confirmed[np.isnan(df_fit_confirmed['fgaussian_popt_x0'])].sort_values(by='total_cases', ascending=False)
def model_correction(df_fit, country, province=None):

    """

    Returns new popts (model parameter options) and scaling factor

    """

    scaling_factor = 1.

    

#     try:

#         popts = np.array(fits[['fgaussian_popt_a','fgaussian_popt_x0','fgaussian_popt_sd']])[0]

#         if pd.isna(popts[0]) or (np.array(fits['max_cases']) < 50):

#             raise Exception

#     except:

        

    # because the means are bimodal

    fits = country_slice(df_fit, country, province)

    popts = np.array(fits[['fgaussian_popt_a','fgaussian_popt_x0','fgaussian_popt_sd']])[0]

    print(f"popts[1]={popts[1]}")

    if (popts[1] <= 30):

        popts, scaling_factor = nyc_scale(df_fit, country, province, ref_country='China', ref_province='Hubei')

    else:

#               (popts[1] > 50 and popts[1] < 70)

        popts, scaling_factor = nyc_scale(df_fit, country, province, ref_country='US', ref_province='New York')

        

        

    return popts, scaling_factor
df_fit_confirmed
df_fit_fatality_old = df_fit_fatality

df_fit_confirmed_old = df_fit_confirmed
# apply correction cases

# metric = 'DailyConfirmedCases'

# df_all = df_fit_confirmed

# df_all_out = []

# for i, row in list(df_all.iterrows()):

#     country = row[0]

#     province = row[1]

#     fits = row

#     popts, scaling_factor = model_correction(df_all, country, province)

#     row['scaling_factor'] = scaling_factor

#     row['fgaussian_popt_a'] = popts[0]

#     row['fgaussian_popt_x0'] = popts[1]

#     row['fgaussian_popt_sd'] = popts[2]

#     y = np.array(train[[metric]])

#     x = np.array(list(range(len(y))))

#     y_pred = np.array(fgaussian(x, *popts))

#     row['fgaussian_perr'] = mean_squared_error(y, y_pred)

#     df_all_out.append(row)

# df_fit_confirmed = pd.concat(df_all_out,axis=1)

# df_fit_confirmed = df_fit_confirmed.transpose()

# df_fit_confirmed



# # apply correction cases

# metric = 'DailyFatalities'

# df_all = df_fit_fatality

# df_all_out = []

# for i, row in list(df_all.iterrows())[:5]:

#     country = row[0]

#     province = row[1]

#     fits = row

#     popts, scaling_factor = model_correction(df_all, country, province)

#     row['scaling_factor'] = scaling_factor

#     row['fgaussian_popt_a'] = popts[0]

#     row['fgaussian_popt_x0'] = popts[1]

#     row['fgaussian_popt_sd'] = popts[2]

#     y = np.array(train[[metric]])

#     x = np.array(list(range(len(y))))

#     y_pred = np.array(fgaussian(x, *popts))

#     row['fgaussian_perr'] = mean_squared_error(y, y_pred)

#     df_all_out.append(row)

# df_fit_fatality = pd.concat(df_all_out,axis=1)

# df_fit_fatality = df_fit_fatality.transpose()

# df_fit_fatality
df_fit_fatality = df_fit_fatality.astype({'total_cases':'int64', 'max_cases':'int64', 'fgaussian_perr':'float64'})

df_fit_confirmed = df_fit_confirmed.astype({'total_cases':'int64', 'max_cases':'int64', 'fgaussian_perr':'float64' })
df_fit_fatality
df_fit_confirmed[df_fit_confirmed['Country_Region']=='Denmark']
# plot model mean and sd

plt.plot(df_fit_confirmed['fgaussian_popt_x0'],df_fit_confirmed['fgaussian_popt_sd'],'b.')

plt.title('Confirmed Gaussian mean vs stdev')

plt.xlabel('Mean')

plt.ylabel('Stdev')
df_fit_confirmed.sort_values(by='fgaussian_popt_sd',ascending=False).head(10)
# plot model amplitude and sd

plt.plot(df_fit_confirmed['fgaussian_popt_a'],df_fit_confirmed['total_cases'],'b.')

plt.title('Confirmed Gaussian amplitude vs total_cases')

plt.xlabel('amplitude (log)')

plt.ylabel('total_cases')

plt.xscale('log')

plt.yscale('log')
df_fit_confirmed.sort_values(by='fgaussian_popt_a',ascending=False)
# plot fit against number of cases

plt.plot(df_fit_fatality['fgaussian_popt_x0'],np.log10(df_fit_fatality['total_cases']+0.1),'b.')

plt.title('Fatalities Gaussian mean vs total confirmed cases')

plt.xlabel('Mean')

plt.ylabel('Total cases (log10)')
# plot fit against number of cases

plt.plot(df_fit_confirmed['fgaussian_popt_x0'],np.log10(df_fit_confirmed['total_cases']+0.1),'b.')

plt.title('Gaussian mean vs total confirmed cases')

plt.xlabel('Mean')

plt.ylabel('Total cases (log10)')
# show which countries

print('Mean > 80')

print(df_fit_confirmed[df_fit_confirmed['fgaussian_popt_x0']>80])



# mostly China, the index country, was the first country to flatten the curve

print('Mean < 20')

print(df_fit_confirmed[df_fit_confirmed['fgaussian_popt_x0']<20])
# plot fit against number of fatalities

plt.plot(np.log10(df_fit_fatality['fgaussian_perr']+0.00001),np.log10(df_fit_fatality['total_cases']+0.1),'b.')

plt.title('Fatalities Gaussian RMSE vs total confirmed cases')

plt.xlabel('RMSE (log10)')

plt.ylabel('Total cases (log10)')
# plot fit against number of cases

plt.plot(np.log10(df_fit_confirmed['fgaussian_perr']+0.00001),np.log10(df_fit_confirmed['total_cases']+0.1),'b.')

plt.title('Gaussian RMSE vs total confirmed cases')

plt.xlabel('RMSE (log10)')

plt.ylabel('Total cases (log10)')
# Hubei, China has the largest error

print(df_fit_confirmed[df_fit_confirmed['fgaussian_perr']>1e5])

print(country_slice(df_fit_confirmed, country='China', province='Shanghai'))

plot_fit_country(train, df_fit_confirmed, country='China', province='Shanghai')
from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.1)

clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])

# Lasso(alpha=0.1)

print(clf.coef_)

print(clf.intercept_)
plot_fit_country(train, df_fit_confirmed, country='Colombia')
plot_fit_country(train, df_fit_confirmed, country='Japan')
plot_fit_country(train, df_fit_fatality, country='Japan', metric='DailyFatalities')
plot_fit_country(train, df_fit_confirmed, country='US', province='New Mexico')
plot_fit_country(train, df_fit_confirmed, country='Australia', province='Queensland')
plot_fit_country(train, df_fit_confirmed, country='US', province='Texas')
plot_fit_country(train, df_fit_confirmed, country='US', province='New York')



# Use New York as the reference distribution for NaNs
plot_fit_country(train, df_fit_confirmed, country='Korea, South')
plot_fit_country(train, df_fit_confirmed, country='Italy')
plot_fit_country(train, df_fit_confirmed, country='Australia', province='Tasmania')
country = 'France'

province = None

country_slice(df_fit_confirmed,country,province).sort_values(by='total_cases',ascending=False)
country = 'Netherlands'

province = None

country_slice(df_fit_confirmed,country,province).sort_values(by='total_cases',ascending=False)
# fix NaNs using New York as the reference because the fit looks reasonable and it has a good number of cases

country = 'US'

province = 'New York'

country_slice(df_fit_confirmed,country,province)
train[['Country_Region','Province_State']].drop_duplicates()
def calc_cumsum_predict(train, df_fit, test, country,province,metric='DailyConfirmedCases', scaling_factor=None):

    """

    Returns the cumulative sum for submission

    """

    tmp_test = country_slice(test,country, province)

    tmp = country_slice(train,country, province)

    tmp_fit = country_slice(df_fit,country, province)

   

    x = np.array(list(range(tmp.shape[0])))

    popts, _scaling_factor = get_popts_scaling_factory(tmp_fit, df_fit, country, province, x, metric=metric)

    

    if scaling_factor is None:

        scaling_factor = _scaling_factor

        

    print(f'country={country}, province={province}, scaling_factor={scaling_factor}, popts={popts}')

    y = np.array(tmp[metric])

    x = np.array(list(range( len(y) )))



    x_pred = np.array(list(range( tmp.shape[0], tmp.shape[0]+tmp_test.shape[0] )))

    y_pred = np.array(scaling_factor * fgaussian(x_pred,*popts))

    

#     print(f'y_pred.shape={y_pred.shape}')



    concat_x = np.concatenate([x,x_pred])

    concat_y = np.ceil(np.concatenate([y,y_pred]))

    ret = np.cumsum(concat_y)[tmp.shape[0]:]

    

#     print(f'ret.shape={ret.shape}')

    

    return ret
out = []

country = 'US'

province = 'New York'



country_state_df = train[['Country_Region','Province_State']].drop_duplicates()



for i, row in list(country_state_df.iterrows()):

    country = row[0]

    province = row[1]

    

    tmp_test = country_slice(test,country, province)

    

    y_submit = calc_cumsum_predict(train, df_fit_confirmed, test, country, province, metric='DailyConfirmedCases')

    tmp_test['ConfirmedCases'] = y_submit

    

    y_submit = calc_cumsum_predict(train, df_fit_fatality, test, country, province, metric='DailyFatalities')

    tmp_test['Fatalities'] = y_submit

    

    out.append(tmp_test)



results = pd.concat(out)



# make sure there's no negative fatalities

results['Fatalities'] = results[['Fatalities','ConfirmedCases']].min(axis=1)  # Gambia
results.to_csv('results.csv',index=False)

results[submission.columns].to_csv('submission.csv',index=False)

print(f'Results saved to results.csv {results.shape}, submission_shape={submission.shape}, total_cases={results["ConfirmedCases"].sum()}, total_fatalities={results["Fatalities"].sum()}')
results.groupby(['Country_Region']).agg({'ConfirmedCases':'max'}).reset_index().sort_values(by='ConfirmedCases',ascending=False).head(10)
results.groupby(['Country_Region']).agg({'Fatalities':'max'}).reset_index().sort_values(by='Fatalities',ascending=False).head(10)
plot_fit_country(train, df_fit_confirmed, country='France')
plot_fit_country(train, df_fit_confirmed, country='Iran')
plot_fit_country(train, df_fit_confirmed, country='Spain')
# mean is too far

plot_fit_country(train, df_fit_confirmed, country='US', province='Texas')
# mean is too far

plot_fit_country(train, df_fit_confirmed, country='Colombia')
plot_fit_country(train, df_fit_confirmed, country='Italy')
plot_fit_country(train, df_fit_confirmed, country='Korea, South')
plot_fit_country(train, df_fit_fatality, country='Korea, South', metric='DailyFatalities')
plot_fit_country(train, df_fit_fatality, country='China', province='Hubei', metric='DailyFatalities')
plot_fit_country(train, df_fit_confirmed, country='Canada', province='British Columbia')
plot_fit_country(train, df_fit_fatality, country='US', province='New York', metric='DailyFatalities')
plot_fit_country(train, df_fit_confirmed, country='US', province='New York', metric='DailyConfirmedCases')
plot_fit_country(train, df_fit_fatality, country='Canada', province='British Columbia', metric='DailyFatalities')
plot_fit_country(train, df_fit_confirmed, country='China', province='Hubei', metric='DailyConfirmedCases')
plot_fit_country(train, df_fit_confirmed, country='Diamond Princess', metric='DailyConfirmedCases')
plot_fit_country(train, df_fit_confirmed, country='Switzerland', metric='DailyConfirmedCases')
plot_fit_country(train, df_fit_confirmed, country='Algeria', metric='DailyConfirmedCases')
plot_fit_country(train, df_fit_confirmed, country='US', province='New York', metric='DailyConfirmedCases')
plot_fit_country(train, df_fit_confirmed, country='US', province='New Jersey', metric='DailyConfirmedCases')
# Compare results

train_all = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

train_all['Date'] = pd.to_datetime(train_all['Date'])

df_ypred = pd.merge(results, train_all, on=['Country_Region','Province_State','Date'])

df_ypred.columns = [c.replace('_x','_pred') for c in df_ypred.columns]

df_ypred['ConfirmedCases_diff'] = abs(df_ypred['ConfirmedCases_pred'] - df_ypred['ConfirmedCases_y'])

df_ypred['Fatalities_diff'] = abs(df_ypred['Fatalities_pred'] - df_ypred['Fatalities_y'])

df_ypred_stats = df_ypred.groupby(['Country_Region']).agg({'ConfirmedCases_diff':'sum', 'Fatalities_diff':'sum'}).reset_index()

df_ypred_stats.sort_values(by='ConfirmedCases_diff', ascending=False).head(20)
cols = ['ForecastId','Province_State','Country_Region','ConfirmedCases_pred','ConfirmedCases_y','ConfirmedCases_diff','Fatalities_pred','Fatalities_y','Fatalities_diff']

df_ypred[df_ypred['Country_Region'].isin(['US'])].sort_values(by='ConfirmedCases_diff',ascending=False)[cols].head(5)
df_ypred[df_ypred['Country_Region'].isin(['Italy'])].sort_values(by='ConfirmedCases_diff',ascending=False)[cols].head(5)