# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

corona = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
def tidy(data, start, confirmed_key='Confirmed', recovered_key='Recovered', deaths_key='Deaths', date_key='Date'):

    data["Active"] = data[confirmed_key] - data[recovered_key] - data[deaths_key]

    data["Date"] = pd.to_datetime(data[date_key]) 

    data["Days"] = (data["Date"] - pd.to_datetime([start]*len(data))).astype('timedelta64[D]')

    assert('Active' in data)

    assert('Date' in data)

    assert('Days' in data)

tidy(corona, corona['Date'].iloc[0])
TODAY_DAY = (pd.to_datetime('today') - corona['Date'].iloc[0]).days
it = corona[corona["Country/Region"] == "Italy"]

_ = it.plot(x = "Days", y ="Active", kind='scatter', title="Active cases in Italy")
import pymc3 as pm
# De Nicolao's data are slightly different from John Hopkins' ones

dn_data = pd.DataFrame([(34, 322), (35, 400), (36, 650), (37, 821), (38, 1049), (39, 1577)], columns=["Days", "Active"])

dn_data
# De Nicolao considers data starting from day 34

DN_START = dn_data["Days"][0] -  1
import theano
observations = theano.shared(np.array(np.log(dn_data['Active'])))

variable = theano.shared(np.array(dn_data['Days'] - DN_START).astype('int'))
with pm.Model() as log_linear:

    alpha = pm.HalfNormal("alpha", sd=20)

    beta = pm.HalfNormal("beta", sd=20)

    sigma = pm.HalfNormal("sigma", sd=20)

    log_active = pm.Normal("log_active", mu=alpha*variable + beta, sigma=sigma, observed=observations)
with log_linear:

    dn_MAP = pm.find_MAP()


fig = plt.figure(figsize=(20, 10))

x = np.arange(dn_data['Days'][0], TODAY_DAY)

start, end = (i.strftime("%a %d %b %Y") for i in (it['Date'].iloc[0] + pd.to_timedelta(str(DN_START+1)+'d'), it['Date'].iloc[-1])) 

fig.suptitle("Generated data vs. observed from {} to {}".format(start, end))

ax = fig.add_subplot(121, xlabel='Days', ylabel='Active', title='Log scale'.format(start, end))

ax.scatter(dn_data['Days'], np.log(dn_data['Active']), label='Observed (De Nicolao data)')

ax.scatter(it[it['Days'] > DN_START+len(dn_data)]['Days'], np.log(it[it['Days'] > DN_START+len(dn_data)]['Active']), 

           label='Observed (John Hopkins data)')

ax.plot(x, dn_MAP['alpha']*(x  - DN_START) + dn_MAP['beta'], label='Generated')

ax.legend()

ax = fig.add_subplot(122, xlabel='Days', ylabel='Active', title='Active cases')

ax.scatter(dn_data['Days'], dn_data['Active'], label='Observed (De Nicolao data)')

ax.scatter(it[it['Days'] > DN_START+len(dn_data)]['Days'], it[it['Days'] > DN_START+len(dn_data)]['Active'], 

           label='Observed (John Hopkins data)')

ax.plot(x, np.exp(dn_MAP['alpha'].mean()*(x  - DN_START) + dn_MAP['beta'].mean()), label='Generated')

ax.legend()

plt.show()

plt.close()
h_data = it[(it['Days'] > DN_START) & (it['Date'] <= pd.to_datetime('2020-03-01'))]
observations.set_value(np.array(np.log(h_data['Active'])))

variable.set_value(np.array(h_data['Days'] - DN_START).astype('int'))
with log_linear:

    h_MAP = pm.find_MAP()
fig = plt.figure(figsize=(20, 10))

fig.suptitle("Generated data vs. observed (John Hopkins' data) from {} to {}".format(start, end))

x = np.arange(h_data['Days'].iloc[0], TODAY_DAY)

h_obs = it[it['Days'] > DN_START+len(h_data)]

ax = fig.add_subplot(121, xlabel='Days', ylabel='Active', title='Log scale')

ax.scatter(h_data['Days'], np.log(h_data['Active']), label='Observed')

ax.scatter(it[it['Days'] > DN_START+len(h_data)]['Days'], np.log(it[it['Days'] > DN_START+len(h_data)]['Active']), 

           label='Observed')

ax.plot(x, h_MAP['alpha']*(x  - DN_START) + h_MAP['beta'], label='Generated')

ax.legend()

ax = fig.add_subplot(122, xlabel='Days', ylabel='Active', title='Active cases')

ax.scatter(h_data['Days'], h_data['Active'], label='Observed')

ax.scatter(h_obs['Days'], h_obs['Active'], label='Observed')

ax.plot(x, np.exp(h_MAP['alpha']*(x  - DN_START) + h_MAP['beta']), label='Generated')

ax.legend()

plt.savefig("it.png")

plt.show()

plt.close()
from scipy.interpolate import BSpline
def mkB(n, data, degree=1):

    """Given the number of knots and data, 

       makes a matrix of B-splines basis, 

       with knots on quantiles. By default it uses linear splines."""

    knot_list = list(np.quantile(data, np.linspace(start=0, stop=1, num=n)))

    t = ([knot_list[0]]*degree)+knot_list+([knot_list[n-1]]*degree)

    c = [0]*(n+2) # zero weigths to get basis

    B = []

    for i in range(len(c)):

        c[i] = 1

        B.append(BSpline(t=t, c=c, k=degree)(data))

        c[i] = 0  

    

    return np.stack(B, axis=1)
def spline_country(data, country, ax_log, ax, knots=6, k_key='Country/Region', active_key='Active', days_key='Days', use_MAP=True):

    """Model the number of active cases in country, 

       with the given number of linear splines with knots on quantiles.

       By default it uses the much faster `find_MAP` method, 

       but sometimes this gives not sensible results, 

       and the slower MCMC `sample` might be better. 

       It returns the sampled model."""

    data_k = data[(data[k_key] == country) & (data[active_key] > 0)]

    active = sorted(data_k.groupby(days_key).sum()[active_key])

    days = sorted(data_k[days_key].unique())



    B_k = mkB(knots, days, 1)



    with pm.Model() as hopkins_k:

        alpha = pm.Normal("alpha",  mu=0, sd=1, shape=B_k.shape[1])

        beta = pm.HalfNormal("beta", sd=20)

        sigma = pm.HalfNormal("sigma", sd=20)

        mu=pm.math.dot(B_k, alpha) + beta

        log_active = pm.Normal("log_active", mu=mu, sigma=sigma, observed=np.log(active))

        if use_MAP:

            h_k = pm.find_MAP(return_raw=True)

        if not use_MAP or not h_k[1].success:

            h_k = pm.sample()

    

    if use_MAP:

        print('{}: using find_MAP'.format(country))

        a, b = h_k[0]['alpha'], h_k[0]['beta']

    else:

        print('{}: using MCMC sampling'.format(country))        

        a, b = h_k['alpha'].mean(axis=0), h_k['beta'].mean()

    generated = np.dot(B_k, a) + b

    ax_log.scatter(days, np.log(active), s=3, label='{}: Observed'.format(country))

    ax_log.plot(days, generated, label='{}: Generated'.format(country))

    ax_log.legend()

    ax.scatter(days, active, label='{}: Observed'.format(country))

    ax.plot(days, np.exp(generated), label='{}: Generated'.format(country))

    ax.legend()

    return h_k
COUNTRIES = ('Italy', 'Germany', 'Spain', 'France', 'United Kingdom', 'South Korea', 'US', 'Poland', 'Switzerland', 'China', 'India')
fig = plt.figure(figsize=(20, 10))

start, end = (corona['Date'].iloc[i].strftime("%a %d %b %Y") for i in (0, -1))

WEEKS = int((corona['Date'].iloc[-1] - corona['Date'].iloc[0]).days / 7) + 1

fig.suptitle("Generated data vs. observed (John Hopkins' data) from {} to {}".format(start, end))

ax_log = fig.add_subplot(121, xlabel='Days', ylabel='Active', title='Generated data vs. observed (Log scale)')

ax = fig.add_subplot(122, xlabel='Days', ylabel='Active', title='Generated data vs. observed')

for k in COUNTRIES:

    try:

        spline_country(corona, k, ax_log, ax, WEEKS)

    except Exception as e:

        print(e, k)

plt.savefig('countries.png')

plt.show()

plt.close()
# http://www.governo.it/it/approfondimento/coronavirus/13968 

it_events = [('2020-01-30', 'Flights to/from China blocked'),

             ('2020-02-21', 'Quarantine for active cases'),

             ('2020-02-23', 'Red zones'),

             ('2020-02-25', 'Schools closed in some regions'),

             ('2020-03-01', 'Partial lockdown in some regions'),

             ('2020-03-04', 'Schools closed'),

             ('2020-03-08', 'Strict lockdown in some regions'),

             ('2020-03-09', 'Strict lockdown'),

             ('2020-03-20', 'Stricter mobility restrictions'),

             ('2020-03-22', 'Halt all non-essential work activities'),

             ('2020-03-24', 'Higher fines for lockdown rebels'),

             ('2020-04-13', 'Preliminary unlocking in some regions'),

             ('2020-05-04', 'First unlocking steps nationwide'),

             ('2020-05-18', 'Many commercial activities are permitted to open'),

             ('2020-06-03', 'Interregional mobility'),

             ('2020-06-15', 'Wider reopening'),

             ('2020-08-17', 'Masks mandatory after 18:00'),

             ('2020-10-08', 'Masks mandatory all the day'),

            ]

START = pd.to_datetime(corona['Date'].iloc[0])

it_events = [((pd.to_datetime(d) - START).days, e) for d, e in it_events]
fig = plt.figure(figsize=(20, 10))

fig.suptitle("Generated data vs. observed (John Hopkins' data) from {} to {}".format(start, end))

ax_log = fig.add_subplot(121, xlabel='Days', ylabel='Active', title='Generated data vs. observed (Log scale)')

ax = fig.add_subplot(122, xlabel='Days', ylabel='Active', title='Generated data vs. observed')

spline_country(corona,'Italy', ax_log, ax, WEEKS)

for i, (d, e) in enumerate(it_events):

    ax_log.axvline(x=d, color='C{}'.format((i + 1) % 10), ls=':', label=e)

ax_log.legend()

plt.savefig('it-events.png')

plt.show()

plt.close()
dpc_region = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_region.csv')
tidy(dpc_region, corona['Date'].iloc[0], confirmed_key='TotalPositiveCases')
REGIONS = ('Lombardia', 'Veneto', 'Emilia-Romagna', 'Lazio', 'Marche', 'Toscana', 'Campania', 'Piemonte')
fig = plt.figure(figsize=(20, 10))

start, end = (dpc_region['Date'].iloc[i].strftime("%a %d %b %Y") for i in (0, -1))

fig.suptitle("Generated data vs. observed (Dipartimento Protezione Civile's data) from {} to {}".format(start, end))

ax_log = fig.add_subplot(121, xlabel='Days', ylabel='Active', title='Generated data vs. observed (Log scale)')

ax = fig.add_subplot(122, xlabel='Days', ylabel='Active', title='Generated data vs. observed')

for r in REGIONS:

    try: 

        spline_country(dpc_region, r, ax_log, ax, WEEKS, k_key='RegionName')

    except Exception as e:

        print(r, e)

for i, (d, e) in enumerate(it_events[2:]):

    ax_log.axvline(x=d, color='C{}'.format((i + 1) % 10), ls=':', label=e)

ax_log.legend()

plt.savefig('regions.png')

plt.show()

plt.close()
fig = plt.figure(figsize=(20, 10))

fig.suptitle("Generated data vs. observed (Dipartimento Protezione Civile's data) from {} to {}".format(start, end))

ax_log = fig.add_subplot(121, xlabel='Days', ylabel='Deaths', title='Generated data vs. observed (Log scale)')

ax = fig.add_subplot(122, xlabel='Days', ylabel='Deaths', title='Generated data vs. observed')

for r in REGIONS:

    try:

        spline_country(dpc_region, r, ax_log, ax, WEEKS, k_key='RegionName', active_key='Deaths')

    except Exception as e:

        print(r, e)

for i, (d, e) in enumerate(it_events[2:]):

    ax_log.axvline(x=d, color='C{}'.format((i + 1) % 10), ls=':', label=e)

ax_log.legend()

plt.savefig('regions-deaths.png')

plt.show()

plt.close()
fig = plt.figure(figsize=(20, 10))

fig.suptitle("Generated data vs. observed (Dipartimento Protezione Civile's data) from {} to {}".format(start, end))

ax_log = fig.add_subplot(121, xlabel='Days', ylabel='Hospitalized', title='Generated data vs. observed (Log scale)')

ax = fig.add_subplot(122, xlabel='Days', ylabel='Hospitalized', title='Generated data vs. observed')

for r in REGIONS:

    try:

        spline_country(dpc_region, r, ax_log, ax, WEEKS, k_key='RegionName', active_key='TotalHospitalizedPatients', use_MAP=(r != 'Veneto'))

    except Exception as e:

        print(r, e)

for i, (d, e) in enumerate(it_events[2:]):

    ax_log.axvline(x=d, color='C{}'.format((i + 1) % 10), ls=':', label=e)

ax_log.legend()

plt.savefig('regions-hospital.png')

plt.show()

plt.close()