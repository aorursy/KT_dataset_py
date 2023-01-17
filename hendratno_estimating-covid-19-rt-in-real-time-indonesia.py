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



FILTERED_REGIONS = [

    'Diamond Princess'

]



%config InlineBackend.figure_format = 'retina'
# Column vector of k

k = np.arange(0, 70)[:, None]



# Different values of Lambda

lambdas = [10, 20, 30, 40]



# Evaluated the Probability Mass Function (remember: poisson is discrete)

y = sps.poisson.pmf(k, lambdas)



# Show the resulting shape

print(y.shape)
fig, ax = plt.subplots()



ax.set(title='Poisson Distribution of Cases\n $p(k|\lambda)$')



plt.plot(k, y,

         marker='o',

         markersize=3,

         lw=0)



plt.legend(title="$\lambda$", labels=lambdas);
k = 20



lam = np.linspace(1, 45, 90)



likelihood = pd.Series(data=sps.poisson.pmf(k, lam),

                       index=pd.Index(lam, name='$\lambda$'),

                       name='lambda')



likelihood.plot(title=r'Likelihood $P\left(k_t=20|\lambda\right)$');
k = np.array([20, 40, 55, 90])



# We create an array for every possible value of Rt

R_T_MAX = 12

r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)



# Gamma is 1/serial interval

# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article

# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316

GAMMA = 1/7



# Map Rt into lambda so we can substitute it into the equation below

# Note that we have N-1 lambdas because on the first day of an outbreak

# you do not know what to expect.

lam = k[:-1] * np.exp(GAMMA * (r_t_range[:, None] - 1))



# Evaluate the likelihood on each day and normalize sum of each day to 1.0

likelihood_r_t = sps.poisson.pmf(k[1:], lam)

likelihood_r_t /= np.sum(likelihood_r_t, axis=0)



# Plot it

ax = pd.DataFrame(

    data = likelihood_r_t,

    index = r_t_range

).plot(

    title='Likelihood of $R_t$ given $k$',

    xlim=(0,10)

)



ax.legend(labels=k[1:], title='New Cases')

ax.set_xlabel('$R_t$');
posteriors = likelihood_r_t.cumprod(axis=1)

posteriors = posteriors / np.sum(posteriors, axis=0)



columns = pd.Index(range(1, posteriors.shape[1]+1), name='Day')

posteriors = pd.DataFrame(

    data = posteriors,

    index = r_t_range,

    columns = columns)



ax = posteriors.plot(

    title='Posterior $P(R_t|k)$',

    xlim=(0,10)

)

ax.legend(title='Day')

ax.set_xlabel('$R_t$');
most_likely_values = posteriors.idxmax(axis=0)

most_likely_values
def highest_density_interval(pmf, p=.9):

    # If we pass a DataFrame, just call this recursively on the columns

    if(isinstance(pmf, pd.DataFrame)):

        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],

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

    return pd.Series([low, high], index=[f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])



hdi = highest_density_interval(posteriors)

hdi.tail()
ax = most_likely_values.plot(marker='o',

                             label='Most Likely',

                             title=f'$R_t$ by day',

                             c='k',

                             markersize=4)



ax.fill_between(hdi.index,

                hdi['Low_90'],

                hdi['High_90'],

                color='k',

                alpha=.1,

                lw=0,

                label='HDI')



ax.legend();
#original data

# original_states = pd.read_csv("/kaggle/input/covid19-cases-in-us/covid-19_us-states.csv",

#                      usecols=[0,1,3],

#                      index_col=['state', 'date'],

#                      parse_dates=['date'],

#                      squeeze=True).sort_index()



#Indonesia data

states = pd.read_csv("/kaggle/input/covid19-indonesia/covid_19_indonesia_time_series_all.csv",

                     usecols=[0,2,7],

                     index_col=['Location', 'Date'],

                     parse_dates=['Date'],

                     squeeze=True

                    ).sort_index()

states.rename_axis(index={'Location': 'location', 'Date': 'date'}, inplace=True)

states.rename('cases', inplace=True)





#Ignore date for Diamond Princess passenger

states.drop(labels = 'Diamond Princess', level=0, inplace=True)

#Ignore states without data

all_states = states.index.get_level_values(0).unique()

for state in all_states:

    if (states.xs(state).max() <= 1):

        print(f"Dropping {location}. It has no data.")

        states.drop(labels = state, level=0, inplace=True)

all_states = states.index.get_level_values(0).unique()



#Impute missing dates with zeros

start_date = states.index.get_level_values(1).min()

end_date = states.index.get_level_values(1).max()

all_dates_index = pd.MultiIndex.from_product(

    [all_states, pd.date_range(start_date, end_date)],

    names=['location', 'date'])

# set indes with all known dates

states = states.reindex(all_dates_index, fill_value=np.NaN)

# set first value to 0 for each state

for state in all_states:

    states.loc[(state, start_date)] = 0

# fill missing values

states.fillna(method='ffill', inplace=True)



print(states.index.get_level_values(0).unique())
# original_state_name = 'New York'

state_name = 'Indonesia'



def prepare_cases(cases):

    new_cases = cases.diff()



    smoothed = new_cases.rolling(7,

        win_type='gaussian',

        min_periods=1,

        center=True).mean(std=2).round()

    

    zeros = smoothed.index[smoothed.eq(0)]

    if len(zeros) == 0:

        idx_start = 0

    else:

        last_zero = zeros.max()

        idx_start = smoothed.index.get_loc(last_zero) + 1

    smoothed = smoothed.iloc[idx_start:]

    original = new_cases.loc[smoothed.index]

    

    return original, smoothed



cases = states.xs(state_name).rename(f"{state_name} cases")

# cases = original_states.xs(original_state_name).rename(f"{original_state_name} cases")



original, smoothed = prepare_cases(cases)



original.plot(title=f"{state_name} New Cases per Day",

               c='k',

               linestyle=':',

               alpha=.5,

               label='Actual',

               legend=True,

             figsize=(500/72, 400/72))



ax = smoothed.plot(label='Smoothed',

                   legend=True)



ax.get_figure().set_facecolor('w')
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

    prior0 = sps.gamma(a=4).pdf(r_t_range)

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

posteriors, log_likelihood = get_posteriors(smoothed, sigma=.15)
ax = posteriors.plot(title=f'{state_name} - Daily Posterior for $R_t$',

           legend=False, 

           lw=1,

           c='k',

           alpha=.3,

           xlim=(0.4,4))



ax.set_xlabel('$R_t$');
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

    

    extended = pd.date_range(start=pd.Timestamp('2020-03-21'),

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

    ax.set_ylim(0.0, 5.0)

    ax.set_xlim(pd.Timestamp('2020-03-1'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))

    fig.set_facecolor('w')



    

fig, ax = plt.subplots(figsize=(600/72,400/72))



plot_rt(result, ax, state_name)

ax.set_title(f'Real-time $R_t$ for {state_name}')

ax.xaxis.set_major_locator(mdates.WeekdayLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
sigmas = np.linspace(1/20, 1, 20)



targets = ~states.index.get_level_values('location').isin(FILTERED_REGIONS)

states_to_process = states.loc[targets]



results = {}



for state_name, cases in states_to_process.groupby(level='location'):

    

    print(state_name)

    new, smoothed = prepare_cases(cases)

    if len(smoothed) < 2:

        print(f"Not enough data to plot: {state_name}")

        clear_output(wait=True)

        continue



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



# Plot it

fig, ax = plt.subplots()

ax.set_title(f"Maximum Likelihood value for $\sigma$ = {sigma:.2f}");

ax.plot(sigmas, total_log_likelihoods)

ax.axvline(sigma, color='k', linestyle=":")
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



print('Done.')
ncols = 4

nrows = int(np.ceil(len(results) / ncols))



fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))



for i, (state_name, result) in enumerate(final_results.groupby('location')):

    if len(result) < 2:

        print(f"Not enough data to plot: {state_name}")

        continue

    plot_rt(result, axes.flat[i], state_name)



fig.tight_layout()

fig.set_facecolor('w')
# Uncomment the following line if you'd like to export the data

final_results.to_csv('rt.csv')
# As of 1 August 2020

# lssr = large-scale social restrictions (Pembatasan Sosial Berskala Besar / PSBB)

# tq = territorial quarantine (Karantina Wilayah)

no_lssr_no_tq = [

    'Aceh',

    'Bali',

    'Bengkulu',

    'Daerah Istimewa Yogyakarta',

    'Gorontalo',

    'Jambi',

    'Jawa Tengah',

    'Jawa Timur',

    'Kalimantan Barat',

    'Kalimantan Selatan',

    'Kalimantan Tengah',

    'Kalimantan Timur',

    'Kalimantan Utara',

    'Kepulauan Bangka Belitung',

    'Kepulauan Riau',

    'Lampung',

    'Maluku',

    'Maluku Utara',

    'Nusa Tenggara Barat',

    'Nusa Tenggara Timur',

    'Papua',

    'Papua Barat',

    'Riau',

    'Sulawesi Barat',

    'Sulawesi Selatan',

    'Sulawesi Tengah',

    'Sulawesi Tenggara',

    'Sulawesi Utara',

    'Sumatera Barat',

    'Sumatera Selatan',

    'Sumatera Utara'

]



partial_lssr = [

    'Banten',

    'Indonesia',

    'Jawa Barat'

]



lssr = [

    'DKI Jakarta'

]



partial_tq = []



tq = []



#FULL_COLOR = [.7,.7,.7]

#NONE_COLOR = [179/255,35/255,14/255]

#PARTIAL_COLOR = [.5,.5,.5]

#ERROR_BAR_COLOR = [.3,.3,.3]



# Color select: https://htmlcolorcodes.com/



NONE_COLOR = [180/255,0/255,0/255]

PARTIAL_LSSR_COLOR = [180/255,90/255,90/255]

LSSR_COLOR = [180/255,135/255,135/255]

PARTIAL_TQ_COLOR = [180/255,156/255,156/255]

TQ_COLOR = [180/255,177/255,177/255]

ERROR_BAR_COLOR = [.3,.3,.3]
final_results
filtered = final_results.index.get_level_values(0).isin(FILTERED_REGIONS)

mr = final_results.loc[~filtered].groupby(level=0)[['ML', 'High_90', 'Low_90']].last()



def plot_standings(mr, figsize=None, title='Most Recent $R_t$ by Location'):

    if not figsize:

        figsize = ((15.9/50)*len(mr)+.1,2.5)

        

    fig, ax = plt.subplots(figsize=figsize)



    ax.set_title(title)

    err = mr[['Low_90', 'High_90']].sub(mr['ML'], axis=0).abs()

    bars = ax.bar(mr.index,

                  mr['ML'],

                  width=.825,

                  color=TQ_COLOR,

                  ecolor=ERROR_BAR_COLOR,

                  capsize=2,

                  error_kw={'alpha':.5, 'lw':1},

                  yerr=err.values.T)



    for bar, state_name in zip(bars, mr.index):

        if state_name in no_lssr_no_tq:

            bar.set_color(NONE_COLOR)

        if state_name in partial_lssr:

            bar.set_color(PARTIAL_LSSR_COLOR)

        if state_name in lssr:

            bar.set_color(LSSR_COLOR)

        if state_name in partial_tq:

            bar.set_color(PARTIAL_TQ_COLOR)

        if state_name in tq:

            bar.set_color(TQ_COLOR)



    labels = mr.index.to_series().replace({'Daerah Istimewa Yogyakarta':'DI Yogyakarta','Kepulauan Bangka Belitung':'Kep Bangka Belitung'})

    ax.set_xticklabels(labels, rotation=90, fontsize=11)

    ax.margins(0)

    ax.set_ylim(0,6.)

    ax.axhline(1.0, linestyle=':', color='k', lw=1)



    leg = ax.legend(handles=[

                        Patch(label='None', color=NONE_COLOR),

                        Patch(label='Partial Large-Scale Social Restriction', color=PARTIAL_LSSR_COLOR),

                        Patch(label='Large-Scale Social Restriction', color=LSSR_COLOR),

                        Patch(label='Partial Territorial Quarantine', color=PARTIAL_TQ_COLOR),

                        Patch(label='Territorial Quarantine', color=TQ_COLOR)

                    ],

                    title='Restriction Status',

                    ncol=3,

                    loc='upper left',

                    columnspacing=.75,

                    handletextpad=.5,

                    handlelength=1)



    leg._legend_box.align = "left"

    fig.set_facecolor('w')

    return fig, ax



mr.sort_values('ML', inplace=True)

plot_standings(mr);
mr.sort_values('High_90', inplace=True)

plot_standings(mr);
show = mr[mr.High_90.le(1)].sort_values('ML')

fig, ax = plot_standings(show, title='Likely Under Control');
show = mr[mr.Low_90.ge(1.0)].sort_values('Low_90')

fig, ax = plot_standings(show, title='Likely Not Under Control');

ax.get_legend().remove()