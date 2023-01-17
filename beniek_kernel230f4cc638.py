# This Python 3 notebook was adapted (with minimal changes) from Kevin Systrom's work:  

# https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra



import os



from matplotlib import pyplot as plt

from matplotlib.dates import date2num, num2date

from matplotlib import dates as mdates

from matplotlib import ticker

from matplotlib.colors import ListedColormap

from matplotlib.patches import Patch



from scipy import stats as sps

from scipy.interpolate import interp1d



from IPython.display import clear_output



# We must (at least temporarily) exclude data from the teritories below

# for calculating Rt for individual states.

# The totals (AUS) still include the filteresd states' data

FILTERED_STATE_CODES = ['ACT']



%config InlineBackend.figure_format = 'retina'
def highest_density_interval(pmf, p=.9, debug=False):

    # If we pass a DataFrame, just call this recursively on the columns

    if(isinstance(pmf, pd.DataFrame)):

        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],

                            index=pmf.columns)

    

    cumsum = np.cumsum(pmf.values)

    

    # N x N matrix of total probability mass for each low, high

    total_p = cumsum - cumsum[:, None]

    

    # Return all indices with total_p > p

    lows, highs = (total_p > p).nonzero()

    

    # Find the smallest range (highest density)

    best = (highs - lows).argmin()

    

    low = pmf.index[lows[best]]

    high = pmf.index[highs[best]]

    

    return pd.Series([low, high], index=[f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])



#hdi = highest_density_interval(posteriors, debug=True)

#hdi.tail()
data_file = '../input/australiacovid19/COVID-19_New_Aus.csv'

states = pd.read_csv(data_file,

                    usecols=['date', 'state', 'new_cases'],

                    dayfirst=True,

                    parse_dates=['date'],

                    index_col=['state', 'date'],

                    squeeze=True).sort_index()



states.tail()
# 'AUS' means the total for all states and teritories

# Change it to 'NSW, 'VIC', etc. to get plots for individual states

state_name = 'AUS'



def prepare_cases(cases, cutoff=25):

    #new_cases = cases



    smoothed = cases.rolling(7,

        win_type='gaussian',

        min_periods=1,

        center=True).mean(std=2).round()

    

    idx_start = np.searchsorted(smoothed, cutoff)

    

    smoothed = smoothed.iloc[idx_start:]

    original = cases.loc[smoothed.index]

    

    return original, smoothed



cases = states[state_name]



original, smoothed = prepare_cases(cases,1)



plt.figure();

original.plot(title=f"{state_name}: New Cases per Day",

            c='k',

            linestyle=':',

            alpha=.5,

            label='Actual',

            legend=True,

            figsize=(500/72, 300/72))



ax = smoothed.plot(label='Smoothed', legend=True)



ax.get_figure().set_facecolor('w')
# We create an array for every possible value of Rt

R_T_MAX = 4

r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)



# Gamma is 1/serial interval

# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article

# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316

GAMMA = 1/7



def get_posteriors(sr, sigma=0.15):



    # (1) Calculate Lambda

    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))



    

    # (2) Calculate each day's likelihood

    likelihoods = pd.DataFrame(

        data = sps.poisson.pmf(sr[1:].values, lam),

        index = r_t_range,

        columns = sr.index[1:])

    

    # (3) Create the Gaussian Matrix

    process_matrix = sps.norm(loc=r_t_range,

                              scale=sigma

                             ).pdf(r_t_range[:, None]) 



    # (3a) Normalize all rows to sum to 1

    process_matrix /= process_matrix.sum(axis=0)

    

    # (4) Calculate the initial prior

    #prior0 = sps.gamma(a=4).pdf(r_t_range)

    prior0 = np.ones_like(r_t_range)/len(r_t_range)

    prior0 /= prior0.sum()



    # Create a DataFrame that will hold our posteriors for each day

    # Insert our prior as the first posterior.

    posteriors = pd.DataFrame(

        index=r_t_range,

        columns=sr.index,

        data={sr.index[0]: prior0}

    )

    

    # We said we'd keep track of the sum of the log of the probability

    # of the data for maximum likelihood calculation.

    log_likelihood = 0.0



    # (5) Iteratively apply Bayes' rule

    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):



        #(5a) Calculate the new prior

        current_prior = process_matrix @ posteriors[previous_day]

        

        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)

        numerator = likelihoods[current_day] * current_prior

        

        #(5c) Calcluate the denominator of Bayes' Rule P(k)

        denominator = np.sum(numerator)

        

        # Execute full Bayes' Rule

        posteriors[current_day] = numerator/denominator

        

        # Add to the running sum of log likelihoods

        log_likelihood += np.log(denominator)

    

    return posteriors, log_likelihood



# Note that we're fixing sigma to a value just for the example

posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)



# Note that this takes a while to execute - it's not the most efficient algorithm

hdis = highest_density_interval(posteriors, p=.9)



most_likely = posteriors.idxmax().rename('ML')



# Look into why you shift -1

result = pd.concat([most_likely, hdis], axis=1)



result.tail()
def plot_rt(result, ax, state_name):

    

    ax.set_title(f"{state_name}")

    

    # Colors

    ABOVE = [1,0,0]

    MIDDLE = [1,1,1]

    BELOW = [0,0,0]

    cmap = ListedColormap(np.r_[

        np.linspace(BELOW,MIDDLE,25),

        np.linspace(MIDDLE,ABOVE,25)

    ])

    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5

    

    index = result['ML'].index.get_level_values('date')

    values = result['ML'].values

    

    # Plot dots and line

    ax.plot(index, values, c='k', zorder=1, alpha=.25)

    ax.scatter(index,

               values,

               s=40,

               lw=.5,

               c=cmap(color_mapped(values)),

               edgecolors='k', zorder=2)

    

    # Aesthetically, extrapolate credible interval by 1 day either side

    lowfn = interp1d(date2num(index),

                     result['Low_90'].values,

                     bounds_error=False,

                     fill_value='extrapolate')

    

    highfn = interp1d(date2num(index),

                      result['High_90'].values,

                      bounds_error=False,

                      fill_value='extrapolate')

    

    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),

                             end=index[-1]+pd.Timedelta(days=1))

    

    ax.fill_between(extended,

                    lowfn(date2num(extended)),

                    highfn(date2num(extended)),

                    color='k',

                    alpha=.1,

                    lw=0,

                    zorder=3)



    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);

    

    # Formatting

    ax.xaxis.set_major_locator(mdates.MonthLocator())

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.xaxis.set_minor_locator(mdates.DayLocator())

    

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax.yaxis.tick_right()

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.margins(0)

    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)

    ax.margins(0)

    ax.set_ylim(0.0, 4.0)

    ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))

    fig.set_facecolor('w')



    

fig, ax = plt.subplots(figsize=(600/72,400/72))



# Since we now use a uniform prior, the first datapoint is pretty bogus, so just truncating it here

plot_rt(result[1:], ax, state_name)

ax.set_title(f'COVID-19 Real-time Effective Reproduction Rate $R_t$ for {state_name}')

ax.xaxis.set_major_locator(mdates.WeekdayLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
sigmas = np.linspace(1/20, 1, 20)



targets = ~states.index.get_level_values('state').isin(FILTERED_STATE_CODES)

states_to_process = states.loc[targets]



results = {}



for state_name, cases in states_to_process.groupby(level='state'):

    

    print(state_name)

    new, smoothed = prepare_cases(cases, cutoff=10)

    

    if len(smoothed) == 0:

        new, smoothed = prepare_cases(cases, cutoff=1)

    

    result = {}

    

    # Holds all posteriors with every given value of sigma

    result['posteriors'] = []

    

    # Holds the log likelihood across all k for each value of sigma

    result['log_likelihoods'] = []

    

    for sigma in sigmas:

        posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)

        result['posteriors'].append(posteriors)

        result['log_likelihoods'].append(log_likelihood)

    

    # Store all results keyed off of state name

    results[state_name] = result

    clear_output(wait=True)



print('Done.')
# Each index of this array holds the total of the log likelihoods for

# the corresponding index of the sigmas array.

total_log_likelihoods = np.zeros_like(sigmas)



# Loop through each state's results and add the log likelihoods to the running total.

for state_name, result in results.items():

    total_log_likelihoods += result['log_likelihoods']



# Select the index with the largest log likelihood total

max_likelihood_index = total_log_likelihoods.argmax()



# Select the value that has the highest log likelihood

sigma = sigmas[max_likelihood_index]

print(f'Optimal sigma = {sigma:.2f}')

# Plot it

#fig, ax = plt.subplots()

#ax.set_title(f"Maximum Likelihood value for $\sigma$ = {sigma:.2f}");

#ax.plot(sigmas, total_log_likelihoods)

#ax.axvline(sigma, color='k', linestyle=":")
final_results = None



for state_name, result in results.items():

    print(state_name)

    posteriors = result['posteriors'][max_likelihood_index]

    hdis_90 = highest_density_interval(posteriors, p=.9)

    hdis_50 = highest_density_interval(posteriors, p=.5)

    most_likely = posteriors.idxmax().rename('ML')

    result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)

    if final_results is None:

        final_results = result

    else:

        final_results = pd.concat([final_results, result])

    clear_output(wait=True)



# Since we now use a uniform prior, the first datapoint is pretty bogus, so just truncating it here

final_results = final_results.groupby('state').apply(lambda x: x.iloc[1:].droplevel(0))



print('Done.')
ncols = 2

nrows = int(np.ceil(len(results) / ncols))



fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))



for i, (state_name, result) in enumerate(final_results.groupby('state')):

    plot_rt(result.iloc[1:], axes.flat[i], state_name)



fig.tight_layout()

fig.set_facecolor('w')