# constants // data

MIN_CASES_FIT = 1e-5

"""threshold to start modeling, as confirmed cases over total population"""

MIN_DAYS_FIT = 45

"""total days of valid data needed for training/modeling"""

DAYS_TEST = 10

"""number of recent days held out for prediction/testing"""

MAX_MISSING_DATA = 0.15

"""acceptable portion of missing input data"""

DENOM_SZ = 1000

"""denominater size when computing rmse, proportion active cases, etc"""

# constants // sir-poly

SIR_POLY_BETA_A_MIN = -10

"""lower bound on first order coefficient of sir-poly"""



# constants // ML

RANDOM_SEED = 1234

"""fix seed for random number generator for reproducible results"""

Y_WIN_SZ = 15

"""window size when computing smoothed contact rate"""

MOBILITY_WIN_SZ = 21

"""windows size for rolling mean when replacing missing values"""

CROSS_FOLDS = 5

"""number of folds to use during cross validation"""

PARAMETERS_SVM = [

    {

        'kernel': ['rbf'],

        'C': [0.01, 0.1, 1.0, 10.0],

        'gamma': ['scale'],

        'epsilon': [0.001, 0.005, 0.01, 0.1]

    }

]

"""SVM parameter options to be tried during grid search"""

PARAMETERS_RANDOMFOREST = [

    {

        'n_estimators': [100, 500, 1000],

        'criterion': ['mse'],# 'mae'],

        'min_samples_leaf': [100, 200, 500],

        'max_features': ['auto', 'sqrt']

    }

]

"""Random forest options to be tried during grid search"""

pass
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.integrate import odeint

from scipy.optimize import curve_fit

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA

from bokeh.io import output_notebook, show

from bokeh.plotting import figure, ColumnDataSource

from bokeh.models import BoxAnnotation, Span, HoverTool, CrosshairTool

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor



# inline notebook viz

output_notebook()



# set random seed

np.random.seed(RANDOM_SEED)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
testing = pd.read_csv('../input/hackathon/task_2-owid_covid_data-21_June_2020.csv')

testing.head()
# load raw datasets

confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

countries = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv', decimal=',')

mobility = pd.read_csv('/kaggle/input/covid19-mobility-data/Global_Mobility_Report.csv')
# collapse counts by country

confirmed.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)

confirmed = confirmed.groupby('Country/Region').sum()



recovered.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)

recovered = recovered.groupby('Country/Region').sum()



deaths.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)

deaths = deaths.groupby('Country/Region').sum()
# flip for more traditional dataframe format

confirmed = confirmed.transpose()

confirmed.index.rename('Date', inplace=True)

confirmed.columns.rename('', inplace=True)



recovered = recovered.transpose()

recovered.index.rename('Date', inplace=True)

recovered.columns.rename('', inplace=True)



deaths = deaths.transpose()

deaths.index.rename('Date', inplace=True)

deaths.columns.rename('', inplace=True)
# normalize dates

confirmed.index = pd.to_datetime(confirmed.index)

recovered.index = pd.to_datetime(recovered.index)

deaths.index = pd.to_datetime(deaths.index)



# reindex country data

countries['Country'] = countries['Country'].str.strip()

countries.set_index('Country', inplace=True)
# map any missing country names we can

countries.rename(

    index={

        'Antigua & Barbuda': 'Antigua and Barbuda',

        'Bahamas, The': 'Bahamas',

        'Bosnia & Herzegovina': 'Bosnia and Herzegovina',

        'Cape Verde': 'Cabo Verde',

        'Central African Rep.': 'Central African Republic',

        'Czech Republic': 'Czechia',

        'Taiwan': 'Taiwan*',

        'Trinidad & Tobago': 'Trinidad and Tobago',

        # TODO // more here...

        'United States': 'US'

    },

    inplace=True

)

# validate matches between datasets

count = 0

for name in confirmed:

    if name not in countries.index:

        count += 1

        #print('Missing country data on: {}'.format(name))

print('Missing country data for {} countries'.format(count))

        

# only want country-level data

mobility = mobility[mobility["sub_region_1"].isnull()]

mobility.drop(columns=['country_region_code', 'sub_region_1', 'sub_region_2'], inplace=True)
# more usable names

mobility.rename(

    columns={

        'retail_and_recreation_percent_change_from_baseline': 'retail',

        'grocery_and_pharmacy_percent_change_from_baseline': 'grocery',

        'parks_percent_change_from_baseline': 'parks',

        'transit_stations_percent_change_from_baseline': 'transit',

        'workplaces_percent_change_from_baseline': 'work',

        'residential_percent_change_from_baseline': 'residential'

    },

    inplace=True

)



# reindex mobility data

mobility['date'] = pd.to_datetime(mobility['date'])

mobility.set_index(['country_region', 'date'], inplace=True)
# map any missing mobility country names we can

mobility.rename(

    index={

        'The Bahamas': 'Bahamas',

        "Côte d'Ivoire": "Cote d'Ivoire",

        'South Korea': 'Korea, South',

        'Taiwan': 'Taiwan*',

        # TODO // more here...

        'United States': 'US'

    },

    inplace=True

)



count = 0

mobility_country_names = set(mobility.index.get_level_values('country_region'))

for name in confirmed :

    if name not in mobility_country_names:

        count += 1

        #print('Missing mobility data on: {}'.format(name))

print('Missing mobility data for {} countries'.format(count))
# reindex testing data

testing['date'] = pd.to_datetime(testing['date'])

testing.drop(columns=['total_cases', 'new_cases', 'total_deaths', 'new_deaths'], inplace=True)

testing.set_index(['location', 'date'], inplace=True)
# map any missing mobility country names we can

testing.rename(

    index={

        'Cape Verde': 'Cabo Verde',

        'Czech Republic': 'Czechia',

        'South Korea': 'Korea, South',

        'Taiwan': 'Taiwan*',

        # TODO // more here...

        'United States': 'US'

    },

    inplace=True

)



count = 0

testing_country_names = set(testing.index.get_level_values('location'))

for name in confirmed :

    if name not in testing_country_names:

        count += 1

        #print('Missing testing data on: {}'.format(name))

print('Missing testing data for {} countries'.format(count))
# examine data sample

name = 'Italy'

idx = confirmed[name] > MIN_CASES_FIT * countries.loc[name]['Population']

sample_confirmed = confirmed[name][idx]

sample_recovered = recovered[name][idx]

sample_deaths = deaths[name][idx]



# plot

p = figure(x_axis_type="datetime", title=name, plot_height=400, plot_width=800)

p.xaxis.axis_label = 'Time'

p.yaxis.axis_label = 'Count'

p.line(sample_confirmed.index, sample_confirmed, color='blue', legend_label='confirmed')

p.line(sample_recovered.index, sample_recovered, color='green', legend_label='recovered')

p.line(sample_deaths.index, sample_deaths, color='red', legend_label='deaths')

p.legend.location = "top_left"

show(p)



# country data

countries.loc[name]
data = {}



for name in confirmed:

    if name not in mobility_country_names:

        continue

    if name not in testing_country_names:

        continue

    if name not in countries.index:

        continue



    # select data of interest

    pop = countries.loc[name]['Population']

    min_cases = MIN_CASES_FIT * pop

    idx = confirmed[name] > min_cases

    idx_dt = confirmed[idx].index

    

    # check for sufficient input data

    idx_overlap = idx_dt.intersection(mobility.loc[name].index)

    mob_missing_count = mobility.loc[name].loc[idx_overlap]['retail'].isnull().sum()

    if mob_missing_count > len(idx) * MAX_MISSING_DATA:

        #print('Insufficient mobility data {} {}'.format(name, mob_missing_count))

        continue

        

    idx_overlap = idx_dt.intersection(testing.loc[name].index)

    test_missing_count = testing.loc[name].loc[idx_overlap]['new_tests_smoothed'].isnull().sum()

    if test_missing_count > len(idx) * MAX_MISSING_DATA:

        #print('Insufficient testing data {} {}'.format(name, test_missing_count))

        continue

    

    # compute SIR values

    s = pop - confirmed[name][idx]  # susceptible

    r = recovered[name][idx] + deaths[name][idx]  # recovered

    i = confirmed[name][idx] - r  # infected

    t = range(len(s))

    

    if len(s) < MIN_DAYS_FIT:

        continue

    

    # format SIR data as numpy array

    sir = np.zeros((len(s), 3))

    sir[:, 0] = s

    sir[:, 1] = i

    sir[:, 2] = r

                        

    # store by country

    data[name] = {

        'sir': sir,

        'dt': s.index

    }



print('Prepared SIR data for {} countries'.format(len(data)))
# differential equations for SIR

def deriv_sir(y, t, n, beta, gamma):

    s, i, r = y

    dsdt = -beta * s * i / n

    didt = beta * s * i / n - gamma * i

    drdt = gamma * i

    return dsdt, didt, drdt

    

# compute SIR given candidate beta and gamma

def gen_compute_sir(s0, i0, r0, n):

    def compute_sir(t, beta, gamma):

        init = (s0, i0, r0)

        pop = n

        res = odeint(deriv_sir, init, t, args=(pop, beta, gamma))

        return res.flatten()

    return compute_sir
# helper function to visualize SIR

def plot_sir(

    name=None, t=None, s=None, i=None, r=None,

    t_fit=None, s_fit=None, i_fit=None, r_fit=None,

    t_pred=None, s_pred=None, i_pred=None, r_pred=None

):

    p = figure(title=name, x_axis_type="datetime", plot_height=400, plot_width=800)



    # plot actual data

    src = ColumnDataSource(data={

        'dt': t,

        's': s if s is not None else np.full(len(t), np.nan),

        'i': i if i is not None else np.full(len(t), np.nan),

        'r': r if r is not None else np.full(len(t), np.nan)

    })

    if s is not None:

        ls = p.line('dt', 's', source=src, color='blue', alpha=0.3, legend_label='susceptible (actual)')

        p.add_tools(HoverTool(renderers=[ls], tooltips=[('S', '@s{0,0}')], mode='vline'))

    if i is not None:

        li = p.line('dt', 'i', source=src, color='red', alpha=0.3, legend_label='infected (actual)')

        p.add_tools(HoverTool(renderers=[li], tooltips=[('I', '@i{0,0}')], mode='vline'))

    if r is not None:

        lr = p.line('dt', 'r', source=src, color='green', alpha=0.3, legend_label='recovered (actual)')

        p.add_tools(HoverTool(renderers=[lr], tooltips=[('R', '@r{0,0}')], mode='vline'))

        

    # plot fit data

    if t_fit is not None:

        src_fit = ColumnDataSource(data={

            'dt': t_fit,

            's': s_fit if s_fit is not None else np.full(len(t_fit), np.nan),

            'i': i_fit if i_fit is not None else np.full(len(t_fit), np.nan),

            'r': r_fit if r_fit is not None else np.full(len(t_fit), np.nan)

        })

        if s_fit is not None:

            lsf = p.line('dt', 's', source=src_fit, color='blue', legend_label='susceptible (fit)')

            p.add_tools(HoverTool(renderers=[lsf], tooltips=[('S (fit)', '@s{0,0}')], mode='vline'))

        if i_fit is not None:

            lif = p.line('dt', 'i', source=src_fit, color='red', legend_label='infected (fit)')

            p.add_tools(HoverTool(renderers=[lif], tooltips=[('I (fit)', '@i{0,0}')], mode='vline'))

        if r_fit is not None:

            lrf = p.line('dt', 'r', source=src_fit, color='green', legend_label='recovered (fit)')

            p.add_tools(HoverTool(renderers=[lrf], tooltips=[('R (fit)', '@r{0,0}')], mode='vline'))

            

    # plot prediction data

    if t_pred is not None:

        src_pred = ColumnDataSource(data={

            'dt': t_pred,

            's': s_pred if s_pred is not None else np.full(len(t_pred), np.nan),

            'i': i_pred if i_pred is not None else np.full(len(t_pred), np.nan),

            'r': r_pred if r_pred is not None else np.full(len(t_pred), np.nan)

        })

        if s_pred is not None:

            lsp = p.line('dt', 's', source=src_pred, color='blue', line_dash=[4, 4], legend_label='susceptible (predicted)')

            p.add_tools(HoverTool(renderers=[lsp], tooltips=[('S (pred)', '@s{0,0}')], mode='vline'))

        if i_pred is not None:

            lip = p.line('dt', 'i', source=src_pred, color='red', line_dash=[4, 4], legend_label='infected (predicted)')

            p.add_tools(HoverTool(renderers=[lip], tooltips=[('I (pred)', '@i{0,0}')], mode='vline'))

        if r_pred is not None:

            lrp = p.line('dt', 'r', source=src_pred, color='green', line_dash=[4, 4], legend_label='recovered (predicted)')

            p.add_tools(HoverTool(renderers=[lrp], tooltips=[('R (pred)', '@r{0,0}')], mode='vline'))



    # additional annotation

    p.add_tools(CrosshairTool(dimensions='height', line_alpha=0.3))

    p.legend.location = "top_left"

    if t_fit is not None and t_pred is not None:

        t_cutoff = t_fit[-1] + (t_pred[0] - t_fit[-1]) / 2.0

        p.add_layout(

            Span(location=t_cutoff, dimension='height', line_color='gray', line_dash='dotted')

        )

    show(p)
# test with dummy values

t = range(100)

res = odeint(deriv_sir, (9990, 10, 0), t, args=(10000, 0.4, 0.2))



# plot

plot_sir(name='Dumb Blonde Bokeh SIR', t=t, s=res[:,0], i=res[:,1], r=res[:,2])
for name in ['Italy', 'Japan', 'US']:

    sir = data[name]['sir']

    dt = data[name]['dt']

    s0, i0, r0 = sir[0, :]

    pop = countries.loc[name]['Population']

    t = range(sir.shape[0])



    # fit SIR model

    fx = gen_compute_sir(s0, i0, r0, pop)

    opt_params, opt_cov = curve_fit(fx, t, sir.flatten(), bounds=(0, 1))

    beta, gamma = opt_params

    

    print('{} (beta: {:.4f}, gamma: {:.4f})'.format(name, beta, gamma))

   

    # integrate

    fit = odeint(deriv_sir, (s0, i0, r0), t, args=(pop, beta, gamma))

    

    # plot

    plot_sir(

        t=dt, i=sir[:, 1], r=sir[:, 2],

        t_fit=dt, i_fit=fit[:, 1], r_fit=fit[:, 2]

    )
# differential equations for SIR w/ polynomial contact rate

def deriv_sir_poly(y, t, n, beta_a, beta_b, gamma):

    s, i, r = y

    

    # compute polynomial contact rate

    x = i / n * DENOM_SZ  # (1000)

    beta = beta_a * x + beta_b



    dsdt = -beta * s * i / n

    didt = beta * s * i / n - gamma * i

    drdt = gamma * i

    return dsdt, didt, drdt

    

# compute SIR w/ polynomial contact given candidate values

def gen_compute_sir_poly(s0, i0, r0, n):

    def compute_sir_poly(t, beta_a, beta_b, gamma):

        init = (s0, i0, r0)

        pop = n

        res = odeint(deriv_sir_poly, init, t, args=(pop, beta_a, beta_b, gamma))

        return res.flatten()

    return compute_sir_poly
for name in ['Italy', 'Japan', 'US']:

    # select data of interest

    sir = data[name]['sir']

    dt = data[name]['dt']

    s0, i0, r0 = sir[0, :]

    pop = countries.loc[name]['Population']

    t = range(sir.shape[0])

        

    # fit SIR model

    fx = gen_compute_sir_poly(s0, i0, r0, pop)

    opt_params, opt_cov = curve_fit(

        fx,

        t,

        sir.flatten(),

        bounds=([SIR_POLY_BETA_A_MIN, 0, 0], [0, 1, 1])

    )

    beta_a, beta_b, gamma = opt_params

    

    print('{} (beta A: {:.4f}, beta B: {:.4f}, gamma: {:.4f})'.format(

        name, beta_a, beta_b, gamma

    ))

        

    # integrate

    fit = odeint(

        deriv_sir_poly,

        (s0, i0, r0),

        t,

        args=(pop, beta_a, beta_b, gamma)

    )

    

    # plot

    plot_sir(

        t=dt, i=sir[:, 1], r=sir[:, 2],

        t_fit=dt, i_fit=fit[:, 1], r_fit=fit[:, 2]

    )
# setup

debug_countries = ['Italy', 'Japan', 'US']

rmse_normalized = []

for name in data:

    sir = data[name]['sir']

    dt = data[name]['dt']

    pop = countries.loc[name]['Population']

    

    # get training split

    train_sz = int(sir.shape[0] - DAYS_TEST)

    train_t = range(train_sz)

    train_data = sir[:train_sz]

    

    # fit SIR model against training split

    s0, i0, r0 = sir[0, :]

    fx = gen_compute_sir_poly(s0, i0, r0, pop)

    opt_params, opt_cov = curve_fit(

        fx,

        train_t,

        train_data.flatten(),

        bounds=([SIR_POLY_BETA_A_MIN, 0, 0], [0, 1, 1])

    )

    beta_a, beta_b, gamma = opt_params

        

    # integrate across train split

    fit = odeint(

        deriv_sir_poly,

        (s0, i0, r0),

        train_t,

        args=(pop, beta_a, beta_b, gamma)

    )



    # integrate across test split

    test_t = range(train_sz, sir.shape[0])

    predicted = odeint(

        deriv_sir_poly,

        (sir[train_sz, 0], sir[train_sz, 1], sir[train_sz, 2]),

        test_t,

        args=(pop, beta_a, beta_b, gamma)

    )

    

    # score RMSE against normalized values

    test_data = sir[train_sz:]

    rmse = mean_squared_error(test_data, predicted, squared=False)

    rmse_normalized.append(rmse / pop * DENOM_SZ)

    

    if name not in debug_countries:

        continue

        

    print('{}'.format(name))

    print('beta A: {:.4f}, beta B: {:.4f}, gamma: {:.4f}'.format(

        beta_a, beta_b, gamma

    ))

    print('RMSE: {:.2f}, RMSE per {}: {:.6f}'.format(

        rmse, DENOM_SZ, rmse / pop * DENOM_SZ

    ))

    

    # plot

    plot_sir(

        t=dt, i=sir[:, 1], r=sir[:, 2],

        t_fit=dt[:train_sz], i_fit=fit[:, 1], r_fit=fit[:, 2],

        t_pred=dt[train_sz:], i_pred=predicted[:, 1], r_pred=predicted[:, 2]

    )