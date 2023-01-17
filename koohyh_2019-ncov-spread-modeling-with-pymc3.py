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



from IPython.display import display, Markdown

import matplotlib.pyplot as plt

import matplotlib



import seaborn as sns

import arviz as az

import pymc3 as pm





train_df = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

train_df.head()

                       
def get_per_country_data(df, country='Italy',initial_number=100):

    """

    Will extract country and aggregate over dates (this is crucial for countries divided by states).

    The initial data well be date with number of cases greater than or equal to initial_number

    """

    this_country_df = df[df['Country/Region']==country]

    this_country_df = this_country_df[['ConfirmedCases', 'Fatalities', 'Date']].groupby('Date').sum()

    this_country_df = this_country_df[this_country_df['ConfirmedCases']>= initial_number]

    return(this_country_df)





def plot_newcases_and_fatalities(df, country):

    fig = plt.figure(figsize=(15,5))

    ax = df['ConfirmedCases'].plot.line(lw=3, color='blue', label='New Cases')

    df['Fatalities'].plot.line(lw=3, color='red', ax=ax, label='Fatalities')

    ax.legend( ['ConfirmedCases', 'Fatalities'], fontsize=20)

    ax.grid()

    ax.set_xlabel('Date', fontsize=20)

    ax.set_ylabel('Counts', fontsize=20)

    ax.set_title(country, fontsize=30)

    

    

def predict_value(trace, x):

    A = trace['A'].mean()

    B = trace['B'].mean()

    K = trace['K'].mean()

    y =   A / (1 + B*np.exp(-K*x))

    return(np.exp(y))



class Country:

    def __init__(self, name, n_passed_days, recent_date, model, trace, ppc, x, y):

        self.name = name

        self.n_passed_days_since100 = n_passed_days

        self.recent_date = recent_date

        self.model = model

        self.trace = trace

        self.ppc = ppc

        self.y = y

        self.x =x

        

def plot_predicted_mu(models_dic, country='Italy', n_future_days=30):

    name = models_dic[country].name

    x = models_dic[country].x

    y = models_dic[country].y

    trace = models_dic[country].trace

    mu = trace['mu'].mean(0)

    sigma = trace['sigma'].mean(0)

    

    yhat = np.exp( mu)

    

    xhat = np.arange(n_future_days) 

    y_pred = [np.rint(predict_value(trace, x)) for x in xhat]

    

    



    

    _, ax = plt.subplots(1, 3, figsize=(20, 6))

    ax[0].scatter(x, np.log(y), s=50, color='black')

    ax[0].plot(x, mu, c='k',lw=4, color='blue')

    ax[0].fill_between(x, mu+1*sigma, mu-1*sigma, alpha=0.6, color='C1')

    ax[0].fill_between(x, mu+2*sigma, mu-2*sigma, alpha=0.6,  color='C1')

    ax[0].set_xlabel('Days since 100 Confirmed Cases', fontsize=20)

    ax[0].set_ylabel('log(Confirmed Cases)', fontsize=20)

    ax[0].set_title(name, fontsize=30)

    ax[0].grid()

    

    ax[1].scatter(x, y, s=50, color='black')

    ax[1].plot(x, yhat, c='k',lw=4, color='blue')

    ax[1].set_xlabel('Days since 100 Confirmed Cases', fontsize=20)

    ax[1].set_ylabel('Confirmed Cases', fontsize=20)

    ax[1].set_title(name, fontsize=30)

    ax[1].grid()

    

    ax[2].scatter(xhat, y_pred, s=50, color='black')

    ax[2].axvline(x[-1], color='red', linestyle='--')

    ax[2].set_xlabel('Days since 100 Confirmed Cases', fontsize=20)

    ax[2].set_ylabel('Confirmed Cases', fontsize=20)

    ax[2].set_title( 'Future ', fontsize=30)

    ax[2].grid()

    



    
# China

country = 'China'

lower = 0

upper = 800000

china_df = get_per_country_data(train_df, country=country,initial_number=lower)

china_df = china_df[china_df['ConfirmedCases'] <=  upper]



print(china_df.shape)

china_df.head()

# china_df.tail()

plot_newcases_and_fatalities(china_df, country='China')


y = np.log(china_df.ConfirmedCases.values- china_df.ConfirmedCases.values[0]+1)

x = np.arange(1, len(y)+1)



plt.plot(x, y)

plt.axhline(12, color='red')





with pm.Model() as china_model:

    sigma = pm.Normal('sigma', mu=2, sd=10)

    C = pm.Normal('C', mu=5, sd=10)

    K = pm.Uniform('K', 0, 1)

    mu = pm.Deterministic('mu', C*(1 -  np.exp(-K*x)))

    

    y_pred = pm.Normal('y_pred', mu=mu, sd=sigma,  observed=y)

    china_trace = pm.sample(1000, tune=1000, chains=4) 
## PPC

china_ppc = pm.sample_posterior_predictive(china_trace, samples=5000, model=china_model)
_, ax = plt.subplots(figsize=(12, 6))

china_ppc['y_pred'].shape

ax.hist([y_pred.mean() for y_pred in china_ppc['y_pred']], bins=20, alpha=0.5)

ax.axvline(y.mean())

ax.set(title='China: Posterior Predictive of the mean', xlabel='mean', ylabel='Frequency')
pm.summary(china_trace, var_names=['C', 'K', 'sigma'])
pm.traceplot(china_trace, var_names=['C', 'K', 'sigma'])
C = china_trace['C'].mean()

saturation = np.exp(C) + china_df.ConfirmedCases.values[0]

saturation
_, ax = plt.subplots(1,1, figsize=(8,8))

plt.scatter(x, y, s=50, color='black')

plt.ylabel('log(ConfirmedCases)', fontsize=20)

plt.xlabel('dates', fontsize=20)

china_mu_m = china_trace['mu'].mean(0) 

sigma= china_trace['sigma'].mean(0)

print(sigma)

plt.plot(x, china_mu_m, c='k',lw=4, color='blue')

plt.fill_between(x, china_mu_m+1*sigma, china_mu_m-1*sigma, alpha=0.6, color='C1')

plt.fill_between(x, china_mu_m+2*sigma, china_mu_m-2*sigma, alpha=0.6,  color='C1')

plt.title(country)
yhat_china = np.exp(china_mu_m) + (china_df.ConfirmedCases.values[0]-1)

_, ax = plt.subplots(1,1, figsize=(8,8))



plt.scatter(x, china_df.ConfirmedCases.values,  s=40, color='blue')

plt.plot(x, yhat_china, lw=3, color='red')

plt.ylabel('ConfirmedCases', fontsize=20)

plt.xlabel('dates', fontsize=20)

plt.yscale('log')
N= 20

train_agg_df = train_df[['Country/Region', 'Date', 'ConfirmedCases', 'Fatalities']].groupby(['Country/Region', 'Date']).sum()

# train_agg_df.head()

top_countries = train_agg_df.groupby('Country/Region').sum().sort_values(by='ConfirmedCases').tail(N).index

top_countries
# class Country:

#     def __init__(self, name, n_passed_days, recent_date, model, trace, ppc, x, y):

#         self.name = name

#         self.n_passed_days_since100 = n_passed_days

#         self.recent_date = recent_date

#         self.model = model

#         self.trace = trace

#         self.ppc = ppc

#         self.y = y

#         self.x =x

    
LOWER = 100

countries_df = []

lower = 100

i = 0

for country in top_countries:

    if country =='China':

        continue ### China has been delt with a different model

    df = get_per_country_data(train_df, country, initial_number=100)

#     print(df.shape)

    if (df.shape[0] >= 15):

        i += 1



        df['country'] = country

        df['days_since_100'] = np.arange(1, df.shape[0]+1)

        print('Country: ', country, df.shape[0], df.index[0])

        countries_df.append(df)

    

count_since_100_df = pd.concat(countries_df, axis=0)

count_since_100_df = count_since_100_df.reset_index()

count_since_100_df.shape

count_since_100_df.head()

selected_countris = np.unique(count_since_100_df.country.values)

# selected_countris



countries_dic = {}



for a_country in selected_countris:

    print('********* ' , a_country, ' ********')

    df = count_since_100_df[count_since_100_df.country == a_country]

    last_date = df.Date.values[-1]

    days_since100 = df.shape[0]    

    y = df. ConfirmedCases

    x = np.arange(len(y))

    with pm.Model() as model:

        A = pm.Normal('A', mu=5, sd=10)

        B = pm.Normal('B', mu=2, sd=10)

        K = pm.HalfCauchy('K', 1)

        sigma = pm.Normal('sigma', mu=5, sd=10)

    

        mu = pm.Deterministic('mu', A / (1 + B*np.exp(-K*x)))

        y_pred = pm.Normal('y_pred', mu=mu, sd=sigma, observed=np.log(y)) ## 

    

        trace = pm.sample(2000, tune=2000, chains=4)

        ppc = pm.sample_posterior_predictive(trace, 2000)

        

        country_obj = Country(name=a_country, n_passed_days=days_since100, recent_date=last_date, model=model, trace=trace, ppc=ppc, x=x, y=y)

        countries_dic[a_country] = country_obj

        

        

    
countries_dic['Italy'].x
# def predict_future(models_dic, country='Italy', n_future_days=30):

#     n_passed_days_since100 = models_dic[country].n_passed_days_since100

#     x = models_dic[country].x

#     y = models_dic[country].y

#     trace = models_dic[country].trace

    

    

#     _, ax = plt.subplots(1, 1, figsize=(16, 8))

    

    
  

    

   



plot_predicted_mu(countries_dic, country='Italy', n_future_days=60)





pm.summary(countries_dic['Italy'].trace, var_names=['A', 'B', 'K', 'sigma'])
plot_predicted_mu(countries_dic, country='United Kingdom', n_future_days=30)
pm.summary(countries_dic['United Kingdom'].trace, var_names=['A', 'B', 'K', 'sigma'])
plot_predicted_mu(countries_dic, country='Spain', n_future_days=30)
pm.summary(countries_dic['Spain'].trace, var_names=['A', 'B', 'K', 'sigma'])
plot_predicted_mu(countries_dic, country='France', n_future_days=30)
plot_predicted_mu(countries_dic, country='Korea, South', n_future_days=60)
plot_predicted_mu(countries_dic, country='Iran', n_future_days=60)
## Italy

country = 'Italy'

lower = 100

upper = 800000

this_country_df = get_per_country_data(train_df, country=country,initial_number=lower)







_, ax = plt.subplots(figsize=(12, 6))

this_country_df['ConfirmedCases'].plot.line(ax=ax)

ax.set(title='Italy: Posterior Predictive of the mean', xlabel='date', ylabel='ConfirmedCases')







plot_newcases_and_fatalities(this_country_df, country=country)
_, ax = plt.subplots(figsize=(6, 6))

y = this_country_df.ConfirmedCases

x = np.arange(1, len(y)+1)

print('number of days', len(y))

sns.scatterplot(x, np.log(y))

ax.set_ylabel('log(ConfirmedCases)')
### Italy  Expony Decay (Increasing) y= a / (1 + b e-kx ), k > 0

with pm.Model() as italy_edi_model:

    A = pm.Normal('A', mu=5, sd=10)

    B = pm.Normal('B', mu=2, sd=10)

    K = pm.HalfCauchy('K', 1)

    sigma = pm.Normal('sigma', mu=5, sd=10)

    

    mu = pm.Deterministic('mu', A / (1 + B*np.exp(-K*x)))

    y_pred = pm.Normal('y_pred', mu=mu, sd=sigma, observed=np.log(y)) ## 

    

    italy_edi_trace = pm.sample(2000, tune=2000, chains=4)

    italy_ppc = pm.sample_posterior_predictive(italy_edi_trace, 2000)
pm.summary(italy_edi_trace, var_names=['A', 'B', 'K', 'sigma'])
future_dates = np.arange(30, 60)

[np.rint(predict_value(italy_edi_trace, x)) for x in future_dates]
italy_ppc['y_pred'].shape

_, ax = plt.subplots(figsize=(6, 6))

ax.hist([y_pred.mean() for y_pred in italy_ppc['y_pred']], bins=19, alpha=0.5)

ax.axvline(np.log(y).mean())

ax.set(title= country + ': Posterior predictive of the mean', xlabel='mean(x)', ylabel='Frequency');

plt.scatter(x, np.log(y), s=50, color='black')

italy_mu_m = italy_edi_trace['mu'].mean(0)

sigma= italy_edi_trace['sigma'].mean(0)

print(sigma)

plt.plot(x, italy_mu_m, c='k',lw=4, color='blue')

plt.fill_between(x, italy_mu_m+1*sigma, italy_mu_m-1*sigma, alpha=0.6, color='C1')

plt.fill_between(x, italy_mu_m+2*sigma, italy_mu_m-2*sigma, alpha=0.6,  color='C1')

plt.xlabel('Days from >= 10', fontsize=10)

plt.ylabel('log(Confirmed Cases)', fontsize=10)

plt.title(country, fontsize=20)
plt.scatter(x, y, s=50, color='black')

plt.yscale('log')

italy_yhat = np.exp( italy_edi_trace['mu'].mean(0)) 

plt.plot(x, italy_yhat, color='red', lw=3)

plt.xlabel('Days from >= 10', fontsize=10)

plt.ylabel('Confirmed Cases', fontsize=10)

plt.title('Italy', fontsize=20)

plt.axvline(23,  color='C1', linestyle='--')

### Italy Negative Binomial



### It doesnt look like a good fit. Since I log-transform the data, Normal likelihood seems to be a better option.

### I have left the following nex cells for just to illustration purpose, otherwise, I will be using normal distribution





with pm.Model() as italy_nb_model:

    A = pm.Normal('A', mu=5, sd=10)

    B = pm.Normal('B', mu=2, sd=10)

    K = pm.HalfCauchy('K', 1)

    sigma = pm.Normal('sigma', mu=5, sd=10)

    

    mu = pm.Deterministic('mu', A / (1 + B*np.exp(-K*x)))

    y_pred = pm.NegativeBinomial('y_pred', mu, sigma, observed=np.log(y))

   

    

    italy_nb_trace = pm.sample(2000, tune=2000, chains=4)

    italy_nb_ppc = pm.sample_posterior_predictive(italy_edi_trace, 2000)

pm.summary(italy_nb_trace, var_names=['A', 'B', 'K', 'sigma'])
italy_ppc['y_pred'].shape

_, ax = plt.subplots(figsize=(6, 6))

ax.hist([y_pred.mean() for y_pred in italy_nb_ppc['y_pred']], bins=19, alpha=0.5)

ax.axvline(np.log(y).mean())

ax.set(title= country + ': NB Posterior predictive of the mean', xlabel='mean(x)', ylabel='Frequency');

plt.scatter(x, np.log(y), s=50, color='black')

nb_italy_mu_m = italy_nb_trace['mu'].mean(0)

sigma= italy_nb_trace['sigma'].mean(0)

print(sigma)

plt.plot(x, nb_italy_mu_m, c='k',lw=4, color='blue')

plt.fill_between(x, nb_italy_mu_m+1*sigma, nb_italy_mu_m-1*sigma, alpha=0.6, color='C1')

plt.fill_between(x, nb_italy_mu_m+2*sigma, nb_italy_mu_m-2*sigma, alpha=0.6,  color='C1')

plt.xlabel('Days from >= 10', fontsize=10)

plt.ylabel('log(Confirmed Cases)', fontsize=10)

plt.title(country, fontsize=20)


### Iran

next_country = 'Iran'

lower = 10

upper = 800000

iran_df = get_per_country_data(train_df, country=next_country, initial_number=lower)







plot_newcases_and_fatalities(iran_df, country=next_country)



  
y_iran = iran_df.ConfirmedCases.values

y_iran = y_iran 

x = np.arange(1, len(y_iran)+1)

# plt.plot(x, np.log(y))



with pm.Model() as iran_edi_model:

    A = pm.Normal('A', mu=5, sd=10)

    B = pm.Normal('B', mu=2, sd=10)

    K = pm.HalfCauchy('K', 1)

    sigma = pm.Normal('sigma', mu=5, sd=10)

    

    mu = pm.Deterministic('mu', A / (1 + B*np.exp(-K*x)))

    irany_pred = pm.Normal('irany_pred', mu=mu, sd=sigma, observed=np.log(y_iran )) 

    

    iran_edi_trace = pm.sample(2000, tune=2000, chains=4)

    iran_ppc = pm.sample_posterior_predictive(iran_edi_trace, 2000)
pm.summary(iran_edi_trace, var_names=['A', 'B', 'K', 'sigma'])
future_dates = np.arange(30, 60)

[np.rint(predict_value(iran_edi_trace, x)) for x in future_dates]
### Mean of PPC doesnt match with mean of data reported!!!!



iran_ppc['irany_pred'].shape

_, ax = plt.subplots(figsize=(6, 6))

ax.hist([y_pred.mean() for y_pred in iran_ppc['irany_pred']], bins=19, alpha=0.5)

ax.axvline(np.log(y_iran).mean(), color='red', linestyle='--')

ax.set(title= next_country + ': Posterior predictive of the mean', xlabel='mean(x)', ylabel='Frequency');

plt.scatter(x, np.log(y_iran), s=50, color='black')

iran_mu_m = iran_edi_trace['mu'].mean(0)

sigma = iran_edi_trace['sigma'].mean(0)

print(sigma)

plt.plot(x, iran_mu_m, c='k',lw=4, color='blue')

plt.fill_between(x, iran_mu_m+1*sigma, iran_mu_m-1*sigma, alpha=0.6, color='C1')

plt.fill_between(x, iran_mu_m+2*sigma, iran_mu_m-2*sigma, alpha=0.6,  color='C1')

plt.xlabel('Days from >= 10', fontsize=10)

plt.ylabel('log(Confirmed Cases)', fontsize=10)

plt.title(next_country, fontsize=20)
plt.scatter(x, y_iran, s=50, color='black')

iran_yhat = np.exp( iran_edi_trace['mu'].mean(0)) 

plt.plot(x, iran_yhat, color='red', lw=3)

plt.xlabel('Days from >= 10', fontsize=10)

plt.ylabel('Confirmed Cases', fontsize=10)

plt.title(next_country, fontsize=20)

plt.axvline(25,  color='C1', linestyle='--')



plt.scatter(x, np.log(y_iran), s=50, color='black')

mu_m = iran_edi_trace['mu'].mean(0)

sigma= iran_edi_trace['sigma'].mean(0)

print(sigma)

plt.plot(x, mu_m, c='k',lw=4, color='blue')

plt.fill_between(x, mu_m+1*sigma, mu_m-1*sigma, alpha=0.6, color='C1')

plt.fill_between(x, mu_m+2*sigma, mu_m-2*sigma, alpha=0.6,  color='C1')
# _, ax = plt.subplots(1,1, figsize=(6,6))



# plt.scatter(x, y, s=50, color='black')

# mu_m = np.exp(new_case_trace['mu'].mean(0))

# sigma= np.exp(new_case_trace['sigma'].mean(0))

# print(sigma)

# plt.plot(x, mu_m, c='k',lw=4, color='blue')

# plt.fill_between(x, mu_m + np.exp(1)*sigma, mu_m- np.exp(1)*sigma, alpha=0.6,  color='C1')

# plt.fill_between(x, mu_m + np.exp(2)*sigma, mu_m- np.exp(2)*sigma, alpha=0.6,  color='C1')

# ax.axvline(x=22, lw=1, color='red')

# ax.set_xlabel('Days after 100 cases', fontsize=20)

# ax.set_ylabel('Number of new cases', fontsize=20)

# ax.set_title(country, fontsize=30)
country = 'Korea, South'

lower =100

this_country_df = get_per_country_data(train_df, country=country, initial_number=lower)

plot_newcases_and_fatalities(this_country_df, country=a_country)



 
_, ax = plt.subplots(figsize=(6, 6))

y = this_country_df.ConfirmedCases

x = np.arange(1, len(y)+1)

print('number of days', len(y))

sns.scatterplot(x, np.log(y))

ax.set_ylabel('log(ConfirmedCases)')


with pm.Model() as country_model:

    A = pm.Normal('A', mu=5, sd=10)

    B = pm.Normal('B', mu=2, sd=10)

    K = pm.HalfCauchy('K', 1)

    sigma = pm.Normal('sigma', mu=5, sd=10)

    

    mu = pm.Deterministic('mu', A / (1 + B*np.exp(-K*x)))

    y_pred = pm.Normal('y_pred', mu, sigma, observed=np.log(y))

   

    

    country_trace = pm.sample(2000, tune=2000, chains=4)

    country_ppc = pm.sample_posterior_predictive(country_trace, 2000) 
pm.summary(country_trace, var_names=['A', 'B', 'K', 'sigma'])


future_dates = np.arange(30, 60)

[np.rint(predict_value(country_trace, x)) for x in future_dates]



country_ppc['y_pred'].shape

_, ax = plt.subplots(figsize=(6, 6))

ax.hist([y_pred.mean() for y_pred in country_ppc['y_pred']], bins=19, alpha=0.5)

ax.axvline(np.log(y).mean())

ax.set(title= country + ': Posterior predictive of the mean', xlabel='mean(x)', ylabel='Frequency');
plt.scatter(x, np.log(y), s=50, color='black')

country_mu_m = country_trace['mu'].mean(0)

country_sigma = country_trace['sigma'].mean(0)

print(country_sigma)

plt.plot(x, country_mu_m, c='k',lw=4, color='blue')

plt.fill_between(x, country_mu_m+1*country_sigma, country_mu_m-1*country_sigma, alpha=0.6, color='C1')

plt.fill_between(x, country_mu_m+2*country_sigma, country_mu_m-2*country_sigma, alpha=0.6,  color='C1')

plt.xlabel('Days from >= 100', fontsize=10)

plt.ylabel('log(Confirmed Cases)', fontsize=10)

plt.title(country, fontsize=20)
plt.scatter(x, y, s=50, color='black')



yhat = np.exp(country_trace['mu'].mean(0)) + y[0] 

plt.plot(x, yhat, color='red', lw=3)

plt.yscale('log')

plt.xlabel('Days from >= 10', fontsize=10)

plt.ylabel('Confirmed Cases', fontsize=10)

plt.title(country, fontsize=20)

# N= 20

# train_agg_df = train_df[['Country/Region', 'Date', 'ConfirmedCases', 'Fatalities']].groupby(['Country/Region', 'Date']).sum()

# # train_agg_df.head()

# top_countries = train_agg_df.groupby('Country/Region').sum().sort_values(by='ConfirmedCases').tail(N).index

# top_countries


countries_df = []

lower = 100



for country in top_countries:

    df = get_per_country_data(train_df, country, initial_number=100)

    df['country'] = country

    df['days_since_100'] = np.arange(1, df.shape[0]+1)

    print('Country: ', country, df.shape[0], df.index[0])

    countries_df.append(df)

    

count_since_100_df = pd.concat(countries_df, axis=0)

count_since_100_df = count_since_100_df.reset_index()

count_since_100_df.shape

count_since_100_df.head()

NUM_COLORS = len(top_countries)

clrs = sns.color_palette('husl', n_colors=NUM_COLORS)



_, ax = plt.subplots(1,1, figsize=(15, 10))

ax.set_yscale('log')

ax.set_xlim((0, 60))

for i, country in  enumerate(top_countries):

    col = clrs[i]

    if country == 'Iran':

        col = 'black'

    df = count_since_100_df[count_since_100_df.country==country]

    sns.lineplot(x='days_since_100', y='ConfirmedCases',data=df, ax=ax, label=country, lw=6, color=col)

plt.figure(figsize=(15,5))



top_n_countires_df = train_agg_df.loc[top_countries].reset_index('Country/Region')

ax = sns.lineplot(data=top_n_countires_df.reset_index('Date'), y='ConfirmedCases', hue='Country/Region', x='Date', lw=5)

a = plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='small'  

)
models_and_traces = {}

lower = 10

for country in top_countries:

    if (country =='China') or (country =='Iran') or (country =='Italy'):

        continue

    print('modeling:', country)

    

    df = get_per_country_data(train_df, country=country, initial_number=lower)

    y = df.ConfirmedCases.values

    y = y - y[0]+1

    x = np.arange(1, len(y)+1)

    

    with pm.Model() as model:

        A = pm.Normal('A', mu=5, sd=10)

        B = pm.Normal('B', mu=2, sd=10)

        K = pm.HalfCauchy('K', 1)

        sigma = pm.Normal('sigma', mu=5, sd=10)

    

        mu = pm.Deterministic('mu', A / (1 + B*np.exp(-K*x)))

        y_pred = pm.Normal('irany_pred', mu=mu, sd=sigma, observed=np.log(y )) 

    

        trace = pm.sample(2000, tune=2000, chains=4)

        ppc = pm.sample_posterior_predictive(trace, 2000)

        models_and_traces[country] = (model, trace, ppc)

    

        

        

        
spain_model, spain_trace, spain_ppc = models_and_traces['Spain']

pm.summary(spain_trace, var_names=['A', 'B', 'K', 'sigma'])





spain_df = get_per_country_data(train_df, country="Spain", initial_number=lower)

spain_y = spain_df.ConfirmedCases.values

spain_y = spain_y - spain_y[0]+1

spain_x = np.arange(1, len(spain_y)+1)









plt.scatter(spain_x, np.log(spain_y), s=50, color='black')

spain_mu_m = spain_trace['mu'].mean(0)

spain_mu_m.shape

spain_sigma= spain_trace['sigma'].mean(0)

print(spain_sigma)

plt.plot(spain_x, spain_mu_m, c='k',lw=4, color='blue')

plt.fill_between(spain_x, spain_mu_m+1*spain_sigma, spain_mu_m-1*spain_sigma, alpha=0.6, color='C1')

plt.fill_between(spain_x, spain_mu_m+2*spain_sigma, spain_mu_m-2*spain_sigma, alpha=0.6,  color='C1')
### Negative Binomial with Exponential regression

country = 'Iran'

lower =10



df = get_per_country_data(train_df, country=country, initial_number=lower)

y = df.ConfirmedCases.values



x = np.arange(1, len(y)+1)



plt.plot(x, y)
with pm.Model() as nb_ex_model:

    ### Priors

    A = pm.Normal('A', 5, 10)

    B = pm.Normal('B', 1, 10)

    sigma = pm.HalfNormal('sigma', 1)

    

    tetha = pm.Deterministic('tetha', A + B* x)

    y_pred = pm.NegativeBinomial('y_pred', tetha, sigma, observed=np.log(y))

    nb_ex_trace = pm.sample(2000, tune=2000, chains=4)

    nb_ex_ppc = pm.sample_posterior_predictive(nb_ex_trace, samples=100)

pm.summary(nb_ex_trace, var_names=['A', 'B',  'sigma'])
a_m = np.exp(nb_ex_trace['A'].mean())

b_m = np.exp(nb_ex_trace['B'].mean())

yhat = a_m * b_m ** x





plt.scatter(x, y)

plt.plot(x, yhat)
#  hierarchical_models  = {}

# tmp_df = top_n_countires_df[top_n_countires_df['Country/Region']!= 'China']

# tmp_df = tmp_df[tmp_df['Country/Region'] != 'Cruise Ship']



# countries = np.unique(tmp_df['Country/Region'].values)

# n_countries = len(countries)

# countries





# # idx = pd.Categorical(tmp_df['Country/Region']).codes

# # groups = len(np.unique(idx))

# # groups





# # Hyperpriors for group nodes

#     mu_a = pm.Normal('mu_a', mu=0., sigma=100)

#     sigma_a = pm.HalfNormal('sigma_a', 5.)

#     mu_b = pm.Normal('mu_b', mu=0., sigma=100)

#     sigma_b = pm.HalfNormal('sigma_b', 5.)



#     # Intercept for each county, distributed around group mean mu_a

#     # Above we just set mu and sd to a fixed value while here we

#     # plug in a common group distribution for all a and b (which are

#     # vectors of length n_counties).

#     a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_counties)

#     # Intercept for each county, distributed around group mean mu_a

#     b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_counties)



#     # Model error

#     eps = pm.HalfCauchy('eps', 5.)



#     radon_est = a[county_idx] + b[county_idx]*data.floor.values



#     # Data likelihood

#     radon_like = pm.Normal('radon_like', mu=radon_est,

#                            sigma=eps, observed=data.log_radon)
#  hierarchical_models  = {}

# tmp_df = top_n_countires_df[top_n_countires_df['Country/Region']!= 'China']

# tmp_df = tmp_df[tmp_df['Country/Region'] != 'Cruise Ship']



# y = tmp_df['ConfirmedCases'].values

# x = np.arange(len(y))







# countries = np.unique(tmp_df['Country/Region'].values)

# n_countries = len(countries)

# # countries





# idx = pd.Categorical(tmp_df['Country/Region']).codes

# groups = len(np.unique(idx))



   

  
# with model:

#     # Sample posterior

#     trace = pm.sample(tune=1500, chains=1, cores=1, target_accept=.9)