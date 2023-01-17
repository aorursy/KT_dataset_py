# imports

import plotly.express as px

import plotly.graph_objects as go

from scipy import stats, special

import numpy as np

import matplotlib.pyplot as plt

from scipy import interpolate

import json



import traceback



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



import numpy as np

import pandas as pd

pd.options.display.float_format = '{:.2f}'.format



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
def country_slice(df, country='China', province=''):

    if province is None or pd.isna(province):

        return df[(df['Country_Region']==country) & (pd.isna(df['Province_State']) == True) ]

    else:

        return df[(df['Country_Region']==country) & (df['Province_State']==province)]

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

train['Date'] = pd.to_datetime(train['Date'])

train = train.sort_values(by=['Country_Region','Province_State','Date'])

train['DailyConfirmedCases'] = train['ConfirmedCases'].diff()

train['DailyFatalities'] = train['Fatalities'].diff()

train['Province_State'] = [ v if not pd.isna(v) else '' for v in train['Province_State'] ]

train['Days'] = (train['Date'] - min(train['Date'])).dt.days

train_bak = train  # make a backup



# replace negatives with a 0

# train.query('DailyConfirmedCases < 0')

# pd.isna( train.query('Country_Region=="Algeria" & Date=="2020-03-25"')['Province_State'] )

filter = train['DailyConfirmedCases']<0

train.loc[filter,'DailyConfirmedCases'] = 0

train.loc[filter,'DailyFatalities'] = 0

filter = np.isnan(train['DailyConfirmedCases'])

train.loc[filter,'DailyConfirmedCases'] = 0

train.loc[filter,'DailyFatalities'] = 0



train.to_csv('train_daily.csv',index=False)



test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test['Date'] = pd.to_datetime(test['Date'])

test['Province_State'] = [ v if not pd.isna(v) else '' for v in test['Province_State'] ]

test['Days'] = (test['Date'] - min(test['Date'])).dt.days

test



# filter training data upto the test date

train = train[train['Date']<min(test['Date'])]



min(test['Date'])



submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

submission



# countries with a Province

train[train['Province_State'].isna()==False]



print(f'train days={max(train["Days"])}, min_date={min(train["Date"])}, max_date={max(train["Date"])}')

print(f'test days={max(test["Days"])}, min_date={min(test["Date"])}, max_date={max(test["Date"])}')

def fpoly(x, a, b, c):

    return a*x**2 + b*x + c



def flinear(x, a, b, c):

    return a*(x+b)+c



def fgauss(x,a,x0,sigma):

    return a*np.exp(-(x-x0)**2/(2*sigma**2))



def predict(fun, x, popts=None, scale=1.):

    return scale * fun(x, *popts)
# example

# fun = fgauss

# fun = flinear

fun = fpoly

metric = 'DailyConfirmedCases'

df = country_slice(train, country='Afghanistan', province='')

y = np.array(df[metric])

x = np.array(range(len(y)))

plt.plot(x,y,label=metric)

# popts, pcov = curve_fit(fun,x,y,p0=[0.23709127,  32.24503999, -12.60506309])

popts, pcov = curve_fit(fun,x,y)

print(popts)

# plt.plot(x,fun(x,*popts))

plt.plot(x,predict(fun,x,popts,scale=1.),label='scale 1')

plt.plot(x,predict(fun,x,popts,scale=2.),label='scale 2')

ax = plt.gca()

ax.legend()
# example

# fun = fgauss

# fun = flinear

fun = fpoly

df = country_slice(train, country='Bahamas')

y = np.array(df['DailyConfirmedCases'])

x = np.array(range(len(y)))

plt.plot(x,y)

# popts, pcov = curve_fit(fun,x,y,p0=[3851.107624, 20.45838, 6.834536])

popts, pcov = curve_fit(fun,x,y)

print(popts)

plt.plot(x,fun(x,*popts))
# example

fun = fgauss

# fun = flinear

df = country_slice(train, country='China', province='Hubei')

y = np.array(df['DailyConfirmedCases'])

x = np.array(range(len(y)))

plt.plot(x,y)

# popts, pcov = curve_fit(fun,x,y,p0=[3851.107624, 20.45838, 6.834536])

popts, pcov = curve_fit(fun,x,y,p0=[5000, 20, 10])

print(popts)

plt.plot(x,fun(x,*popts))
# train.query('DailyConfirmedCases>0').quantile([0.1, 0.25, 0.5, 0.75, 0.9])

MEAN_MAX = 150

MEAN_MIN = 10

AMP_MAX_FATALITIES = 1000

AMP_MAX_CASES = 10 * AMP_MAX_FATALITIES

MIN_CASES = 100



try_params = [ None ]

for amp in [5000, 1000, 10]:

    for x0 in [100, 70, 40, 20]:

        for sd in [100, 40, 10]:

            try_params.append([amp, x0, sd])

            



def fit(x, y, metric):

    fun   = fgauss

    popts = None

    p0    = None

    x     = np.array(x)

    y     = np.array(y)



    try:

        for fun in [fgauss, fpoly]:

            for p0 in try_params:

                try:

                    amp = max(y)

                    popts, _ = curve_fit(fun, x, y, p0=p0)

                    if popts[0] < 1 or popts[1] < MEAN_MIN or popts[1] > MEAN_MAX:

                        raise Exception

                    elif metric == 'DailyConfirmedCases' and popts[0] > max(y) * 10:

                        raise Exception

                    elif metric == 'DailyFatalities' and popts[0] > max(y) * 10:

                        raise Exception

                    else:

                        return popts, fun

                except:

                    pass

    except Exception as e:

        print(f'Something went wrong in fit, country={country}, province={province}, metric={metric}. {e}')

        

    return popts, fun



models = dict()

model_means = []

model_amps = []

# for i, row in list(train[['Country_Region','Province_State']].drop_duplicates().iterrows())[:10]:

for i, row in list(train[['Country_Region','Province_State']].drop_duplicates().iterrows()):

    country = row[0]

    province = row[1]

    models[country + '_' + province] = dict()

    for metric in ['DailyConfirmedCases', 'DailyFatalities']:

        df = country_slice(train, country, province)

        if df.empty:

            raise Exception(f'Dataframe is empty')

        y  = df[metric]

        x  = range(len(y))

        popts, fun = fit(x, y, metric)  

        if popts is None:

            print(f'country={country}, province={province} has popts={popts}')

#         print(f'popts={popts}, fun={fun}, metric={metric}, country={country}, province={province}')

        models[country + '_' + province][metric] = {'fun':fun, 'popts':popts } 

        if fun == fgauss:

            model_amps.append(popts[0])

            model_means.append(popts[1])

    

# models

print(f'{len(model_means)} {len(list(models.keys()))}')

plt.plot(model_means, model_amps, 'b.')

plt.yscale('log')

plt.xlabel('Means')

plt.ylabel('Amplitude')
# SCALE = 2.  # scaling factor for predictions

SCALE = 1.5

MAX_RMSE = int(1e7)



def plot_fit_country(train, country='China', province='', fun=fgauss, popts=None, metric='DailyConfirmedCases'):

    

    tmp = country_slice(train, country, province)

    

    y = np.array(tmp[ metric ])

    x = np.array(list(range(len(y))))

    y_pred = predict(fun,x,popts,scale=SCALE)

    tmp_rmse = mean_squared_error(y, y_pred)

    

    if tmp_rmse > MAX_RMSE:

        print(f'RMSE more than {MAX_RMSE}! {country} {province} {fun} {popts}')

    

    plt.plot(x,y,'b-+', label='Actual')

    plt.plot(x,y_pred,'r.', label='Predicted')

    if len(province) > 0:

        title = f'Prediction for {province}, {country}'

    else:

        title = f'Prediction for {country}'

    title += f'\nScale={SCALE} RMSE={round(tmp_rmse,4)}'

    plt.title(title)

    plt.ylabel(metric)

    plt.xlabel(f'Time since {min(tmp["Date"])}')

    ax = plt.gca()

    ax.legend()

    

    

def plot_country(train_bak, models, country='Italy', province=''):

    nrows = 2

    ncols = 1

    index = 1

    

    plt.subplots_adjust(hspace=1.)

    

    for index, metric in enumerate(['DailyConfirmedCases','DailyFatalities']):

        model = models[country + '_' + province][metric]

        ax = plt.subplot(nrows,ncols,index+1)

    #     plot_fit_country(train_bak, country=country, province=province, fun=fun, popts=popts, metric=metric)

        plot_fit_country(train_bak, country=country, province=province, fun=model['fun'], popts=model['popts'], metric=metric)

        dates = sorted(train_bak['Date'].drop_duplicates().tolist())

        test_date_index = dates.index(test['Date'].min())

        test_date_str = str(test["Date"].min()).replace('00:00:00','')

        plt.axvline(test_date_index,0,10000,label=f'Test {test_date_str}',linestyle='--')

#         ax = plt.gca()

        ax.legend()

    

    

    

# country_province_to_show = {'China':'Hubei', 'Diamond Princess':None, 'Korea, South':None, 'Taiwan*':None, 'Japan':None,

#                             'US':'New York', 'US':'California', 'Spain':None, 'Italy':None, 'Germany':None, 

#                             'Turkey':None, 'Canada':'British Columbia', 'Colombia':None}

# for country, province in list(country_province_to_show.items()):

queries = ['Bahamas', 'Senegal', 'Afghanistan', 'Australia_New South Wales', 'Colombia', 'Japan', 

           'China_Hubei', 'Italy', 'Spain', 'Taiwan*', 'US_New York',

           'Canada_British Columbia'

          ]

for q in queries:

    if '_' not in q:

        province = ''

        country = q

    else:

        country, province = q.split('_')

    plot_country(train_bak, models, country=country, province=province)

    plt.show()
def calc_cumsum_predict(train, models, test, country, province):

    """

    Returns the cumulative sum for submission {'ConfirmedCases': []}, num_rows = test.shape[0]

    """

    tmp_test = country_slice(test,country, province)

    tmp = country_slice(train,country, province)

   

    ret = dict()

    for metric in ['DailyConfirmedCases', 'DailyFatalities']:

        metric_ret = metric.replace('Daily','')

        model = models[country + '_' + province][metric]

        popts, fun = model['popts'], model['fun']

#         print(metric, fun, popts)

        

        #calculate test

        x_pred = np.array(list(range( tmp.shape[0], tmp.shape[0]+tmp_test.shape[0] )))

        y_pred = np.array(predict(fun,x_pred,popts,scale=SCALE)).clip(0).round(4)

        

        #calculate cumsum for train+test

        y = np.array(tmp[metric])

        x = np.array(list(range( len(y) )))

        concat_x = np.concatenate([x,x_pred])

        concat_y = np.concatenate([y,y_pred])

        cumulative_y = np.cumsum(concat_y)

        

        #truncate to only test for submission

        ret[metric_ret] = cumulative_y[tmp.shape[0]:]

    

#         print(ret[metric_ret][:2] )



    return ret







out = []

country_state_df = train[['Country_Region','Province_State']].drop_duplicates()



for i, row in list(country_state_df.iterrows()):

    country = row[0]

    province = row[1]

    

#     if country != 'Senegal':

#     if country != 'Italy':

#         continue

        

    tmp_test = country_slice(test, country, province)

    

    y_submit = calc_cumsum_predict(train, models, test, country, province)

    tmp_test['ConfirmedCases'] = y_submit['ConfirmedCases']

    tmp_test['Fatalities'] = y_submit['ConfirmedCases'] if y_submit['Fatalities'][0] > y_submit['ConfirmedCases'][0] else y_submit['Fatalities']

#     print(f"{country} {province} tmp_test{len(tmp_test['ConfirmedCases'])} y_submit{len(y_submit['ConfirmedCases'])}")

    

    out.append(tmp_test)



results = pd.concat(out)



results.to_csv('results.csv',index=False)

results[submission.columns].to_csv('submission.csv',index=False)

print(f'Results saved to results.csv {results.shape}, submission_shape={submission.shape}, total_cases={results["ConfirmedCases"].sum()}, total_fatalities={results["Fatalities"].sum()}')



# results
results.describe()