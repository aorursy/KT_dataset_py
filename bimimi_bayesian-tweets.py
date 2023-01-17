# Distributions 

    # T

    # Poisson

    # Negative Binomial 



# Model 

    # Hierarchical regression 
# Wrangling 



    # Retrieve inquiry and corresponding response from company for all rows 

    

    # Convert dates col to datetime type

    

    # Calc response time in mins 

    

    # Select airlines 

    

    # Filter inqueries exceeding 60 mins 

    

    # Generate time attributes 

    

        # Generate response word count 
import pandas as pd 

import matplotlib.pyplot as plt 

import numpy as np 

import seaborn as sns 

%matplotlib inline 



from IPython.display import Image

!pip install arviz

import arviz as az

import pymc3 as pm 

import scipy 

import scipy.stats as stats 

import scipy.optimize as opt

import statsmodels.api as sm 

from sklearn import preprocessing 

import theano.tensor as tt



%matplotlib inline 

plt.style.use('ggplot')

colors = ['#373854', '#493267', '#9e379f', '#e86af0', '#7bb3ff',

         '#3bd6c6', '#40e0d0', '#43e8d8', '#89ecda', '#b3ecec']
df = pd.read_csv('../input/twcs/twcs.csv')
df.head()
# Corresponding responses to tweet id



first_inbound = df[pd.isnull(df.in_response_to_tweet_id) & df.inbound]



# Merge



df = pd.merge(first_inbound, df, left_on = 'tweet_id', right_on = 'in_response_to_tweet_id')

df = df[df.inbound_y ^ True]



# New df



df = df[['author_id_x', # tweeter id 

       'created_at_x', # inbound tweet date

       'text_x',

        'author_id_y', # airline id

        'created_at_y', # response date

        'text_y']]



# Datetime



df['created_at_x'] = pd.to_datetime(df['created_at_x'], 

                                    format = '%a %b %d %H:%M:%S +0000 %Y')



df['created_at_y'] = pd.to_datetime(df['created_at_y'], 

                                    format = '%a %b %d %H:%M:%S +0000 %Y')



# Calc time between outbound response and inbound msg

    # i.e. response speed 

    

df['res_time'] = df['created_at_y'] - df['created_at_x']



# Convert time to mins



df['res_time'] = df['res_time'].astype('timedelta64[s]') / 60



# Exclude > 60 mins 



df = df.loc[df['res_time'] <= 60]
df.res_time.describe()
# Select airlines



df = df[(df.author_id_y == 'Delta') 

        | (df.author_id_y == 'AmericanAir') 

        | (df.author_id_y == 'SouthwestAir')

       | (df.author_id_y == 'British_Airways')

       | (df.author_id_y == 'AmericanAir')

       | (df.author_id_y == 'AirAsiaSupport')

       | (df.author_id_y == 'VirginAtlantic')

       | (df.author_id_y == 'AlaskaAir')

       | (df.author_id_y == 'VirginAmerica')

       | (df.author_id_y == 'JetBlue')

]
# Time attributes 



df['created_at_y_dayofweek'] = df['created_at_y'].apply(lambda x:x.dayofweek)

df['created_at_y_day_of_week'] = df['created_at_y'].dt.weekday_name

df['created_at_y_day'] = df['created_at_y'].dt.day



df['created_at_y_is_weekend'] = df['created_at_y_dayofweek'].isin([5,6]).apply(lambda x: 1 

                                                                               if x == True 

                                                                               else 0)
# Response word count



df['word_count'] = df.text_y.apply(lambda x: len(str(x).split()))
df.dtypes
df.head()
# Response time distribution 



plt.figure(figsize = (10, 5))

sns.distplot(df['res_time'], kde = False)

plt.title('Respose frequency by response time')

plt.xlabel('Response time (mins)')

plt.ylabel('Responses');
# Does not appear Gaussian 

    # Gaussian does not properly describe data 
# T-dist 

    # for managing outliers 

    # replace Gaussian likelihood with t-dist 



# T dist = 3 params 

    # mu 

    # scale (analogous to theta)

    # deg. of freedom
# e = exponential dist with mean of 1 

    # if e = 1, then distribution has heavy tails 

        # more probable to find avlues away from mean 

            # compared to Gaussian dist



with pm.Model() as model_t:

    mu = pm.Uniform('mu', lower = 0, upper = 60) # align with filtered response time  

    sd = pm.HalfNormal('sd', sd = 10)

    e = pm.Exponential('e', 1/1) 

    t = pm.StudentT('t', mu = mu, sd = sd, nu = e, observed = df['res_time'])

    trace = pm.sample(2000, tune = 2000)
# MCMC (Markov Chain Monte Carlo)

    # see plausible mu from posterior 

    

az.plot_trace(trace[:1000], var_names = ['mu']);
# Credible mu values between 7.3 - 7.5

    # samples drawn from distributions = significantly different from target distribution 
# How close are inferred means to actual sample mean 



# Use ppc to check for systematic discrepancy between real and simulated data 

ppc = pm.sample_posterior_predictive(trace, samples = 1000, model = model_t)



_, ax = plt.subplots(figsize = (10, 5))



ax.hist([n.mean() for n in ppc['t']], bins = 19, alpha = 0.5)

ax.axvline(df['res_time'].mean())

ax.set(title = 'Posterior predictive of the mean', 

      xlabel = 'mean(x)',

      ylabel = 'freq')
az.summary(trace)
# Highest posterior density 

az.plot_posterior(trace);
df.res_time.mean()
# 94% probability of mu / sd / e / between values



# Inferred mean's uncertainty between 7.3 an 7.5

    # far away from actual sample mean of 13.43

    

        # T dist = not a proper choice 
# Poisson dist 

    # describes P(given num of events occuring within a fixed time / space interval)

        # e.g. phone calls within an hour 

        

    # Events are independent 

    

    # Discrete dist parametrized using mu only

        # mu corresponds to mean and var of the dist 
with pm.Model() as model_p:

    mu = pm.Uniform('mu', lower = 0, upper = 60)

    p = pm.Poisson('p', mu = mu, observed = df['res_time'].values)

    trace = pm.sample(2000, tune = 2000)
az.plot_trace(trace);
# credible mu between 12.9 - 12.98
az.plot_posterior(trace);
ppc = pm.sample_posterior_predictive(trace, samples = 1000, model = model_p)



_, ax = plt.subplots(figsize = (10, 5))



ax.hist([n.mean() for n in ppc['p']], bins = 19, alpha = 0.5)

ax.axvline(df['res_time'].mean())

ax.set(title = 'Posterior predictive of the mean', 

      xlabel = 'mean(x)',

      ylabel = 'freq')
# P dist much closer 

    # Credible mu values between 12.9 - 13.0



az.summary(trace)
# Autocorrelation 

    # degree of similarity between values of the same variabe over successive time interval 

    

# Ideal = no autocor 

# or samples quickly drop to low autocorr values



_ = pm.autocorrplot(trace, var_names = ['mu'])



# Poisson Autocorrelation decreasing over x-ax
# Poisson ppc 



ppc = pm.sample_posterior_predictive(trace, 100, model_p, random_seed = 150) # 100 posterior predictive samples 



pred = az.from_pymc3(trace = trace, posterior_predictive = ppc)



az.plot_ppc(pred, figsize = (10, 5), mean = False)



plt.xlim(0, 60); 

# Observed p = KDE (kernel density estimate of data)



# Posterior predictive p = KDEs computed from each of 100 posterior predictive samples 

    # reflect uncertainty regarding inferred distribution of predicted data 

    

# Above shows scale of P dist may not be a reasonable proxy for SD of the data even after filtering outliers 
ppc = pm.sample_posterior_predictive(trace, samples = 1000, model = model_p)



_, ax = plt.subplots(figsize = (10, 5))



ax.hist([n.mean() for n in ppc['p']], 

        bins = 19, alpha = 0.5)



ax.axvline(df['res_time'].mean())



ax.set(title = 'Posterior predictive of the mean',

      xlabel = 'mean(x)', 

      ylabel = 'freq');
# Small gap between inferred and actual mean 



# Problem: Poisson dist uses same param to describe mean and var 



    # Poisson defined by lambda (rate param)

        # lambda = P(events in time interval)
# Use negative-binomial dist

    # vary var independently of dist mean 

    

# 2 params (mu and alpha)
with pm.Model() as model_n:

    mu = pm.Uniform('mu', lower = 0, upper = 60)

    alpha = pm.Uniform('alpha', lower = 0, upper = 100)

    

    pred = pm.NegativeBinomial('pred', mu = mu, alpha = alpha)

    est = pm.NegativeBinomial('est', mu = mu, alpha = alpha, 

                             observed = df['res_time'].values)

    

    trace = pm.sample(2000, tune = 2000)
# MCMC

az.plot_trace(trace, var_names = ['mu', 'alpha']);
az.plot_posterior(trace);
# credible mu between 12.8 - 13.1
# PPC 



ppc = pm.sample_posterior_predictive(trace, 100, model_n, random_seed = 150) # 100 posterior predictive samples 



pred = az.from_pymc3(trace = trace, posterior_predictive = ppc)



az.plot_ppc(pred, figsize = (10, 5), mean = False)



plt.xlim(0, 60); 
# Uncertainty around inferred distribution reflects kde nicely
ppc = pm.sample_posterior_predictive(trace, samples = 1000, model = model_n)



_, ax = plt.subplots(figsize = (10, 5))



ax.hist([n.mean() for n in ppc['est']], 

        bins = 19, alpha = 0.5)



ax.axvline(df['res_time'].mean())



ax.set(title = 'Posterior predictive of the mean',

      xlabel = 'mean(x)', 

      ylabel = 'freq');
# More closer than Poisson



# Credible mu between 12.8 - 13.1
# Posterior predictive dist 



x_lim = 60



pred = trace.get_values('pred')

mu = trace.get_values('mu').mean()



fig = plt.figure(figsize = (10, 6))

fig.add_subplot(211)



_ = plt.hist(pred, range = [0, x_lim], bins = x_lim, color = colors[1])

_ = plt.xlim(1, x_lim)

_ = plt.ylabel('freq')

_ = plt.title('posterior predictive distribution')



fig.add_subplot(212)



_ = plt.hist(df['res_time'].values, range =[0, x_lim], bins = x_lim)

_ = plt.xlabel('response time (mins)')

_ = plt.ylabel('freq')

_ = plt.title('observed data distribution')
# Posterior for negative binomial somewhat resembles observed 

    # compare with others 
# Bayesian methods for hierarchical modelling / multilevel model



    # Study each airline as a separated entity

    

    # Estimate res_time of each airline AND entire data 

    

# Assuming different airlines has different res_times



    # model each airline independently 

    

    # estimate params mu and alpha for each

        # negative-binomial distribution 
# Airline inquiry & response volume 

plt.figure(figsize = (15, 8))



sns.countplot(x = 'author_id_y',

             data = df, 

             order = df['author_id_y'].value_counts().index)



plt.xlabel('airline')

plt.ylabel('response num')

plt.title('respse / airline')

plt.xticks(rotation = 50);
# Results for airlines with fewer customer inquiries may be more extreme



    # Expect higher uncertainty for these airlines than those with larger volumes
# Model airlines using Negative binomial dist 



air_traces = {}



# CAT to INT 



pre = preprocessing.LabelEncoder()

air_idx = pre.fit_transform(df['author_id_y'])

air = pre.classes_

n_air = len(air) 
air
for p in air:

    with pm.Model() as model_a:

        alpha = pm.Uniform('alpha', lower = 0, upper = 100)

        mu = pm.Uniform('mu', lower = 0, upper = 60)

        

        data = df[df['author_id_y'] == p]['res_time'].values

        

        est = pm.NegativeBinomial('est', mu = mu, alpha = alpha, observed = data)

        pred = pm.NegativeBinomial('pred', mu = mu, alpha = alpha)

        

        trace = pm.sample(2000, tune = 2000)

        

        air_traces[p] = trace
# Posterior predictive dist for airlines 



# Plot

fig, axs = plt.subplots(3, 2, 

                       figsize = (15, 8))



axs = axs.ravel()

y_left_max = 1000

y_right_max = 1000

x_lim = 60

ix = [0, 3, 7]



for i, j, p in zip([0, 1, 2], [0, 2, 4], air[ix]):

    axs[j].set_title('observed: %s' % p)

    axs[j].hist(df[df['author_id_y'] == p]['res_time'].values,

               range = [0, x_lim], 

               bins = x_lim,

               histtype = 'bar')

    axs[j].set_ylim([0, y_left_max])

    

for i, j, p in zip([0, 1, 2], [1, 3, 5], air[ix]):

    axs[j].set_title('posterior predictive dist: %s' % p)

    axs[j].hist(air_traces[p].get_values('pred'),

                     range = [0, x_lim],

                     bins = x_lim,

                     histtype = 'bar',

                     color = colors[1])

    axs[j].set_ylim([0, y_right_max])

    

axs[4].set_xlabel('res time (mins)')

axs[5].set_xlabel('res time (mins)')



plt.tight_layout()
# AirAsia different 

    # Indicate AA takes longer to respond than others 

        # or incomplete small sample extreme results 

            # relative to VA / BA
fig, axs = plt.subplots(3, 2, 

                       figsize = (15, 8))



axs = axs.ravel()

y_left_max = 1000

y_right_max = 1000

x_lim = 60

ix = [4, 6, 2]



for i, j, p in zip([0, 1, 2], [0, 2, 4], air[ix]):

    axs[j].set_title('observed: %s' % p)

    axs[j].hist(df[df['author_id_y'] == p]['res_time'].values,

               range = [0, x_lim], 

               bins = x_lim,

               histtype = 'bar')

    axs[j].set_ylim([0, y_left_max])

    

for i, j, p in zip([0, 1, 2], [1, 3, 5], air[ix]):

    axs[j].set_title('posterior predictive dist: %s' % p)

    axs[j].hist(air_traces[p].get_values('pred'),

                     range = [0, x_lim],

                     bins = x_lim,

                     histtype = 'bar',

                     color = colors[1])

    axs[j].set_ylim([0, y_right_max])

    

axs[4].set_xlabel('res time (mins)')

axs[5].set_xlabel('res time (mins)')



plt.tight_layout()
# Larger samples less extreme results 



# Posterior predictive dist does not very significantly 
# Multilevel / hierarchical regression 



df = df[['res_time', 'author_id_y', 'created_at_y_is_weekend', 'word_count']]

formula = 'response_time ~ ' + ' + '.join(['%s' % variable for variable in df.columns[1:]])



formula
# CAT to INT 

    

# Est baseline param for each airline's res_time



# Est other params across airline data
pre = preprocessing.LabelEncoder()

air_idx = pre.fit_transform(df['author_id_y'])

air = pre.classes_

n_air = len(air) 





with pm.Model() as model_h:

    intercept = pm.Normal('intercept', mu = 0, sd = 100, 

                         shape = n_air)

    

    slope_weekend = pm.Normal('slope_weekend', mu = 0, sd = 100)

    

    slope_word = pm.Normal('slope_word', mu = 0, sd = 100)



    mu = tt.exp(intercept[air_idx] 

                + slope_weekend*df.created_at_y_is_weekend 

                + slope_word*df.word_count)



    est = pm.Poisson('est', mu = mu, observed = df['res_time'].values)

    

    start = pm.find_MAP()

    step = pm.Metropolis()



    trace = pm.sample(3500, step, start = start, progressbar = True)
# MCMC



az.plot_trace(trace);
# Most likely param value for each airline.

    # param = Beta 

        # Beta informs if and how weekend and word count increases res_time 



# Weekend inquiries increases res_time marginally



# Greater word count increases res_time marginally 



# Baseline res_time varies for each airline 



# Hierarchical model estimates Beta for every airline 
# Plot 94% credible interval for each airline's param 

    # Most likely param value (intercept)



_, ax = pm.forestplot(trace, var_names = ['intercept'])

ax[0].set_yticklabels(air.tolist());
# Model has ittle uncertainty for each airline 
# R2 



ppc = pm.sample_posterior_predictive(trace, samples = 2000,

                                    model = model_h)



az.r2_score(df.res_time.values, ppc['est'])