import pandas as pd

import numpy as np



from matplotlib import pyplot as plt

from matplotlib.dates import date2num, num2date

from matplotlib import dates as mdates

from matplotlib import ticker

from matplotlib.colors import ListedColormap

from matplotlib.patches import Patch



from scipy import stats as sps

from scipy.interpolate import interp1d



from IPython.display import clear_output



%config InlineBackend.figure_format = 'retina'
# Load the patient CSV

patients = pd.read_csv(

    'https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv',

    parse_dates=[

        'Accurate_Episode_Date',

        'Case_Reported_Date',

        'Test_Reported_Date',

        'Specimen_Date',

    ],

)
prov_cases = patients.groupby('Accurate_Episode_Date')['Row_ID'].count()

prov_cases.name = 'New_Cases'

prov_cases.sort_index(inplace=True)

#prov_cases = prov_cases.reindex(pd.date_range(prov_cases.index[0], prov_cases.index[-1]), fill_value=0)

prov_cases
region_cases = patients.groupby(['Reporting_PHU', 'Accurate_Episode_Date'])['Row_ID'].count()

region_cases.name = 'New_Cases'

region_cases.sort_index(inplace=True)

region_cases
all_data_patients = patients[['Accurate_Episode_Date', 'Case_Reported_Date']].dropna().sort_values(by='Case_Reported_Date')



all_data_patients_int = all_data_patients.copy()

for c in ['Accurate_Episode_Date', 'Case_Reported_Date']:

    all_data_patients_int[c] = (all_data_patients_int[c] - pd.to_datetime('2020-01-01')).dt.days



ax = all_data_patients_int.plot.scatter(

    title='Onset vs. Confirmed Dates - COVID19',

    x='Case_Reported_Date',

    y='Accurate_Episode_Date',

    alpha=.1,

    lw=0,

    s=10,

    figsize=(6,6))



#formatter = mdates.DateFormatter('%m/%d')

#locator = mdates.WeekdayLocator(interval=2)



#for axis in [ax.xaxis, ax.yaxis]:

#    axis.set_major_formatter(formatter)

#    axis.set_major_locator(locator)
# Calculate the delta in days between onset and confirmation

delay = (all_data_patients['Case_Reported_Date'] - all_data_patients['Accurate_Episode_Date']).dt.days



# Convert samples to an empirical distribution

p_delay = delay.value_counts().sort_index()

new_range = np.arange(0, p_delay.index.max()+1)

p_delay = p_delay.reindex(new_range, fill_value=0)

p_delay /= p_delay.sum()



# Show our work

fig, axes = plt.subplots(ncols=2, figsize=(12,6))

p_delay.plot(title='P(Delay)', ax=axes[0])

p_delay.cumsum().plot(title='P(Delay <= x)', ax=axes[1])

for ax in axes:

    ax.set_xlabel('days')
def adjust_onset_for_right_censorship(onset, p_delay):

    cumulative_p_delay = p_delay.cumsum()

    

    max_date = onset.index[-1]

    adjusted = onset.copy()

    for d, ons in onset.iteritems():

        delay = (max_date-d).days

        if delay <= cumulative_p_delay.index[-1]:

            adjusted.loc[d] = ons / cumulative_p_delay[delay]

        else:

            adjusted.loc[d] = ons  # 0% observed delays



    return adjusted, cumulative_p_delay

def smooth_cases(cases):

    new_cases = cases



    smoothed = new_cases.rolling(7,

        win_type='gaussian',

        min_periods=1,

        center=True).mean(std=2).round()

        #center=False).mean(std=2).round()

    

    zeros = smoothed.index[smoothed.eq(0)]

    if len(zeros) == 0:

        idx_start = 0

    else:

        last_zero = zeros.max()

        idx_start = smoothed.index.get_loc(last_zero) + 1

    smoothed = smoothed.iloc[idx_start:]

    original = new_cases.loc[smoothed.index]

    

    return smoothed



cases = prov_cases.rename("Ontario cases")

original = cases



adjusted, cum_p_delay = adjust_onset_for_right_censorship(cases, p_delay)

smoothed = smooth_cases(adjusted)



original.plot(title="Ontario New Cases per Day",

               c='k',

               linestyle=':',

               alpha=.5,

               label='Actual',

               legend=True,

             figsize=(600/72, 400/72))



ax = adjusted.plot(label='Adjusted for Right-Censorship',

                   legend=True)

ax = smoothed.plot(label='Smoothed and Adjusted for Right-Censorship',

                   legend=True)

ax.get_figure().set_facecolor('w')
# We create an array for every possible value of Rt

R_T_MAX = 12

r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)



# Gamma is 1/serial interval

# https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article

GAMMA = 1/4



def get_posteriors(sr, window=7, min_periods=1):

    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))



    # Note: if you want to have a Uniform prior you can use the following line instead.

    # I chose the gamma distribution because of our prior knowledge of the likely value

    # of R_t.

    

    # prior0 = np.full(len(r_t_range), np.log(1/len(r_t_range)))

    prior0 = np.log(sps.gamma(a=3).pdf(r_t_range) + 1e-14)



    likelihoods = pd.DataFrame(

        # Short-hand way of concatenating the prior and likelihoods

        data = np.c_[prior0, sps.poisson.logpmf(sr[1:].values, lam)],

        index = r_t_range,

        columns = sr.index)



    # Perform a rolling sum of log likelihoods. This is the equivalent

    # of multiplying the original distributions. Exponentiate to move

    # out of log.

    posteriors = likelihoods.rolling(window,

                                     axis=1,

                                     min_periods=min_periods).sum()

    posteriors = np.exp(posteriors)



    # Normalize to 1.0

    posteriors = posteriors.div(posteriors.sum(axis=0), axis=1)

    

    return posteriors



posteriors = get_posteriors(smoothed)
ax = posteriors.plot(title='Ontario - Daily Posterior for $R_t$',

           legend=False, 

           lw=1,

           c='k',

           alpha=.3,

           xlim=(0.4,4))



ax.set_xlabel('$R_t$');
def highest_density_interval(pmf, p=.95):

    

    # If we pass a DataFrame, just call this recursively on the columns

    if(isinstance(pmf, pd.DataFrame)):

        return pd.DataFrame([highest_density_interval(pmf[col]) for col in pmf],

                            index=pmf.columns)

    

    cumsum = np.cumsum(pmf.values)

    best = None

    for i, value in enumerate(cumsum):

        for j, high_value in enumerate(cumsum[i+1:]):

            if (high_value-value > p) and (not best or j<best[1]-best[0]):

                best = (i, i+j+1)

                break

            

    low = pmf.index[best[0]]

    high = pmf.index[best[1]]

    return pd.Series([low, high], index=['Low', 'High'])
# Note that this takes a while to execute - it's not the most efficient algorithm

hdis = highest_density_interval(posteriors)



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

    

    index = result['ML'].index.get_level_values('Accurate_Episode_Date')

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

                     result['Low'].values,

                     bounds_error=False,

                     fill_value='extrapolate')

    

    highfn = interp1d(date2num(index),

                      result['High'].values,

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

    ax.set_ylim(0.0,3.5)

    ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('Accurate_Episode_Date')[-1]+pd.Timedelta(days=1))

    fig.set_facecolor('w')



    

fig, ax = plt.subplots(figsize=(600/72,400/72))



plot_rt(result, ax, 'Ontario')

ax.set_title(f'Real-time $R_t$ for Ontario')

ax.set_ylim(.5,3.5)

ax.xaxis.set_major_locator(mdates.WeekdayLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
target_regions = []

for reg, cases in region_cases.groupby(level=0):

    if cases.max() >= 30:

        target_regions.append(reg)

target_regions
results = {}



provinces_to_process = region_cases.loc[target_regions]



for prov_name, grp in provinces_to_process.groupby(level=0):

    clear_output(wait=True)   

    print(f'Processing {prov_name}')

    cases = grp.droplevel(0)

    adjusted, cum_p_delay = adjust_onset_for_right_censorship(cases, p_delay)

    smoothed = smooth_cases(adjusted)

    print('\tGetting Posteriors')

    try:

        posteriors = get_posteriors(smoothed)

    except:

        display(cases)

    print('\tGetting HDIs')

    hdis = highest_density_interval(posteriors)

    print('\tGetting most likely values')

    most_likely = posteriors.idxmax().rename('ML')

    result = pd.concat([most_likely, hdis], axis=1)

    results[prov_name] = result #.droplevel(0)

    

clear_output(wait=True)

print('Done.')
ncols = 4

nrows = int(np.ceil(len(results) / ncols))



# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))



for i, (prov_name, result) in enumerate(results.items()):

    plot_rt(result, axes.flat[i], prov_name)



fig.tight_layout()

fig.set_facecolor('w')
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
npi_df = pd.read_csv('/kaggle/input/covid19-challenges/npi_canada.csv', parse_dates=['start_date', 'end_date'])

npi_df
npi_df.columns
npi_df['intervention_category'].unique()
npi_df[(npi_df['intervention_category']=='Declaration of emergency (or similar)')\

      &((npi_df['subregion']=='All')|(npi_df['subregion'].isnull()))]