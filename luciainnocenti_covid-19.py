# Importing packages
import pandas as pd
import numpy as np

from scipy import stats as sps
from scipy.interpolate import interp1d

from pathlib import Path 

from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker

from datetime import date, timedelta

from IPython.display import clear_output
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Importing data
sdate = date(2020, 2, 24) + timedelta(days=1)  # start date
edate = date.today() - timedelta(days=1)  # last date

delta = edate - sdate + timedelta(days=1)  # number of reports available until today

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-" \
      + str(edate.year) + '{:02d}'.format(edate.month) + '{:02d}'.format(edate.day) + ".csv"

dateparser = lambda x: pd.to_datetime(x).date  # .date cut off time
regions_new = pd.read_csv(url,
                     usecols=['data', 'denominazione_regione', 'nuovi_positivi'],
                     parse_dates=['data'],
                     date_parser = dateparser,
                     index_col=['denominazione_regione', 'data'],
                     squeeze=True).sort_index()

for i in range(delta.days):
    day = edate - timedelta(days=i + 1)
    url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-" \
          + str(day.year) + '{:02d}'.format(day.month) + '{:02d}'.format(day.day) + ".csv"
    regions = pd.read_csv(url,
                     usecols=['data', 'denominazione_regione', 'nuovi_positivi'],
                     parse_dates=['data'],
                     date_parser = dateparser,
                     index_col=['denominazione_regione', 'data'],
                     squeeze=True).sort_index()
    regions_new = pd.concat([regions, regions_new], axis=0, ignore_index=False)
regions = regions_new.copy()

regions.head()
# Gaussian smooth function
def prepare_cases(cases, cutoff = 5): 
  # cutoff: minimum number of new cases to consider each day
    smoothed = cases.rolling(7, # mobile window of a week
        win_type = 'gaussian',
        min_periods = 1, # windows step
        center=True).mean(std=2).round()
    idx_start = np.searchsorted(smoothed, cutoff) # indexes of days with more than cutoff new cases
    smoothed = smoothed.iloc[idx_start:] # smoothed values
    original = cases.loc[smoothed.index] # original values 
    return original, smoothed
region_name = 'Lombardia' # as example

cases = regions.xs(region_name).rename(f"Casi in {region_name}")

original, smoothed = prepare_cases(cases, cutoff = 25)

fig, ax = plt.subplots()

original.plot(title = f"Nuovi casi per giorno del {region_name}",
               c = 'k',
               linestyle = ':',
               alpha = .5,
               label = 'Reale',
               legend = True,
               figsize = (10, 7))

ax = smoothed.plot(label='Smoothed',
                   legend=True)

ax.get_figure().set_facecolor('w')
GAMMA = 1/7
# We create an array for every possible value of R_t
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

def get_posteriors(sr, sigma=0.15): 
    
    # (1) Calculate Lambda
    lam = sr[:-1].values*np.exp(GAMMA*(r_t_range[:, None]-1)) 
    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix 
    process_matrix = sps.norm(loc = r_t_range,
                              scale = sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    prior0 = sps.gamma(a = 4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index = r_t_range,
        columns = sr.index,
        data = {sr.index[0]: prior0}
    )
    
    # We keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)*P(R_t)
        numerator = likelihoods[current_day]*current_prior
        
        #(5c) Calculuate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood

# Note that we're fixing sigma to a value just for the example
posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)
ax = posteriors.plot(title=f'{region_name} - Posteriori giornaliere per $R_t$',
           legend = False, 
           lw = 1,
           alpha = 1,
           colormap = 'Blues',
           xlim = (0,4))

ax.set_xlabel('$R_t$');
def highest_density_interval(pmf, p=.9):
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()

    # Find the smallest range (highest density)
    best = (highs-lows).argmin()
    
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])
# This takes a while to execute - it's not the most efficient algorithm
hdis = highest_density_interval(posteriors, p=.9)
most_likely = posteriors.idxmax().rename('ML')

# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

result.tail()
def plot_rt(result, ax, region_name):
    
    ax.set_title(f"{region_name}")
    
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['ML'].index.get_level_values('data')

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
    
    extended = pd.date_range(start=pd.Timestamp('2020-02-24'),
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
    ax.set_ylim(-1.0, 5.0)
    ax.set_xlim(pd.Timestamp('2020-02-20'), pd.Timestamp(date.today()-timedelta(days=1)))
    fig.set_facecolor('w')
fig, ax = plt.subplots(figsize=(10,7))

plot_rt(result, ax, region_name)
ax.set_title(f'$R_t$ giornaliero in {region_name}')
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
sigmas = np.linspace(1/20, 1, 20)

results = {}

for region_name, cases in regions.groupby(level='denominazione_regione'):
    
    print(region_name)
    new, smoothed = prepare_cases(cases)
    
    result = {}
    
    # Holds all posteriors with every given value of sigma
    result['posteriors'] = []
    
    # Holds the log likelihood across all k for each value of sigma
    result['log_likelihoods'] = []
    
    for sigma in sigmas:
        posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
        result['posteriors'].append(posteriors)
        result['log_likelihoods'].append(log_likelihood)
    
    # Store all results keyed off of regions name
    results[region_name] = result
    clear_output(wait=True)

print('Fatto.')
# Each index of this array holds the total of the log likelihoods for
# the corresponding index of the sigmas array.
total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each regions's results and add the log likelihoods to the running total.
for region_name, result in results.items():
    total_log_likelihoods += result['log_likelihoods']

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()

# Select the value that has the highest log likelihood
sigma = sigmas[max_likelihood_index]

# Plot it
fig, ax = plt.subplots()
ax.set_title(f"Valore massimo della log-verosomiglianza per $\sigma$ = {sigma:.2f}");
ax.plot(sigmas, total_log_likelihoods)
ax.axvline(sigma, color='k', linestyle=":")
final_results = None

for region_name, result in results.items():
    print(region_name)
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

print('Fatto.')
ncols = 3
nrows = int(np.ceil(len(results)/ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))
print(axes.flat[1])
for i, (region_name, result) in enumerate(final_results.groupby('denominazione_regione')):
    plot_rt(result, axes.flat[i], region_name)

fig.tight_layout()
fig.set_facecolor('w')
FULL_COLOR = [.7,.7,.7]
NONE_COLOR = [179/255,35/255,14/255]
PARTIAL_COLOR = [.5,.5,.5]
ERROR_BAR_COLOR = [.3,.3,.3]

mr = final_results.groupby(level=0)[['ML', 'High_90', 'Low_90']].last()

def plot_standings(mr, figsize=None, title='Più recenti $R_t$ per regione'):
    if not figsize:
        figsize = ((15.9/50)*len(mr)+.1,2.5)
        
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(title)
    err = mr[['Low_90', 'High_90']].sub(mr['ML'], axis=0).abs()
    bars = ax.bar(mr.index,
                  mr['ML'],
                  width=.825,
                  color=FULL_COLOR,
                  ecolor=ERROR_BAR_COLOR,
                  capsize=2,
                  error_kw={'alpha':.5, 'lw':1},
                  yerr=err.values.T)

    labels = mr.index.to_series()
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.margins(0)
    ax.set_ylim(0,2.)
    ax.axhline(1.0, linestyle=':', color='k', lw=1)

    leg = ax.legend(handles=[
                        Patch(label='Full', color=FULL_COLOR),
                    ],
                    title='Lockdown',
                    ncol=3,
                    loc='upper left',
                    columnspacing=.75,
                    handletextpad=.5,
                    handlelength=1)

    leg._legend_box.align = "left"
    fig.set_facecolor('w')
    return fig, ax


mr.sort_values('ML', inplace=True)
plot_standings(mr)