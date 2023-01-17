import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import warnings

warnings.filterwarnings('ignore')



os.chdir('/kaggle/input/covid19')



conf_raw = pd.read_csv('confirmed_cases.csv') # 'total-confirmed-cases-of-covid-19-per-million-people.csv' from Our World in Data

tests_raw = pd.read_csv('tests.csv') # 'full-list-cumulative-total-tests-per-thousand.csv' from Our World in Data

deaths_raw = pd.read_csv('deaths.csv') # 'total-daily-covid-deaths-per-million.csv' from Our World in Data

daily_conf_raw = pd.read_csv('daily_confirmed_cases.csv') # 'total-and-daily-cases-covid-19.csv' from Our World in Data



tests_raw['Entity'] = tests_raw['Entity'].apply(lambda s: s.split('-')[0][:-1])
correl = deaths_raw.merge(daily_conf_raw, on=['Entity','Date']).rename(

    columns={'Daily confirmed deaths per million (deaths per million)':'Daily deaths/million',

             'Daily new confirmed cases (cases)':'Daily cases'}

)[['Entity','Date','Daily deaths/million','Daily cases']]

correl['Date'] = pd.to_datetime(correl['Date'])

correl = correl[correl['Daily cases']>50] # Reducing noisy data

countries_correl = correl['Entity'].value_counts()

countries_correl = countries_correl[countries_correl > 10].index.to_list()

correl = correl[correl['Entity'].apply(lambda i: i in countries_correl)]
def plot_correl(country):    

    df = correl[correl['Entity']==country]

    df = df.sort_values('Date', ascending=True)

    plt.figure(figsize=(20,7))

    plt.plot(df['Date'], df['Daily deaths/million'].rolling(window=6).mean()/df['Daily deaths/million'].max(), label='Daily deaths (scaled)')

    plt.plot(df['Date'], df['Daily cases'].rolling(window=6).mean()/df['Daily cases'].max(), label='Daily cases (scaled)')

    plt.xticks(ticks=df['Date'],rotation=45)

    plt.title(country)

    plt.legend()

    plt.show()

plot_correl('France')

plot_correl('Italy')

plot_correl('Spain')

plot_correl('South Korea')

plot_correl('United Kingdom')
from pandas.tseries.offsets import DateOffset

N = 4



offset_deaths = deaths_raw[['Entity','Date','Total confirmed deaths per million (deaths per million)']]

offset_deaths['Date'] = pd.to_datetime(offset_deaths['Date'])

offset_deaths['Date'] = offset_deaths['Date'].apply(lambda t : t-DateOffset(days=N))

conf_raw['Entity-1'] = conf_raw['Entity'].apply(lambda s: s[:-1])

tests_raw['Entity-1'] = tests_raw['Entity']

full = tests_raw.merge(conf_raw, on=['Entity-1','Date']).rename(columns={'Entity_y':'Entity'})

full['Date'] = pd.to_datetime(full['Date'])

full = full.merge(offset_deaths, on=['Entity','Date'])



data = full[['Entity', 

             'Date', 

             'Total tests per thousand',

             'Total confirmed cases of COVID-19 per million people (cases per million)',

             'Total confirmed deaths per million (deaths per million)']]

data = data.rename(columns={'Total tests per thousand':'Tests/thousand',

                            'Total confirmed cases of COVID-19 per million people (cases per million)':'Cases/million',

                            'Total confirmed deaths per million (deaths per million)':f'Deaths/million d+{N}',

                            })

data['Date'] = pd.to_datetime(data['Date'])

eps = 0.0

data = data[data[f'Deaths/million d+{N}'] > eps]
# Will only consider data from after this date, in order to have a more homogenous dataset

start_date = pd.to_datetime('2020-03-15')

data = data[data['Date']>start_date]



# Some countries are excluded from the dataset because of reliability issues

countries_to_exclude = ['Malaysia', 'Philippines', 'Australia', 'Bahrain', 'Indonesia', 'India',

                        'Pakistan', 'Costa Rica', 'Ecuador', 'Uruguay', 'Thailand', 'Lithuania', 

                        'Tunisia', 'Senegal', 'Turkey', 'Serbia', 'Panama', 'Peru', 'Paraguay',

                        'Mexico', 'Bangladesh', 'Bolivia', 'Chile', 'Ethiopia', 'Argentina',

                        'Ghana', 'Colombia', 'El Salvador', 'Hungary']

data = data[data['Entity'].apply(lambda i: i not in countries_to_exclude)]



# Computing the relevant ratios

data['Confirmed/test'] = data['Cases/million']/(1000*data['Tests/thousand']) # Converted to Tests/million

data[f'Death d+{N}/test'] = data[f'Deaths/million d+{N}']/(1000*data['Tests/thousand'])



# Final list of countries to be considered

countries = data['Entity'].value_counts().sort_values(ascending=False).index.to_list()

print('List of countries to be considered (sorted by descending number of data points):\n', countries)

print(f'\nNumber of countries: {len(countries)}')

print(f'Number of data points: {len(data)}')
# Some corrections need to be made to further homogenize our data

correction = dict(zip(countries, [1.0]*len(countries))) # The value in this dict will be applied 

                                                        # as a multiplier of the value of

                                                        # Deaths d+N / tests for each country



# Correcting for differences in mortality rates due to age distribution and possibly genetic and cutural factors

# Source: https://twitter.com/TrevorSutcliffe/status/1246944321107976192

correction['Italy'] = 0.5

correction['Austria'] = 1.5

correction['Germany'] = 1.5



# Converting number of people tested into number of samples, assuming 2 samples/person on average

# (Some countries report the number of people tested, others report the number of samples)

correction['South Korea'] = 0.5

correction['United Kingdom'] = 0.5

correction['Norway'] = 0.5

correction['Netherlands'] = 0.5

correction['Sweden'] = 0.5



# Correcting for potential relative undercount in deaths (these are assumptions)

correction['Belgium'] *= 0.5

correction['France'] *= 0.8

correction['Italy'] *= 1.2





for i in data.index:

    line = data.loc[i]

    data.loc[i,f'Death d+{N}/test'] = line[f'Death d+{N}/test'] * correction[line['Entity']]



# Removing outliers from dataset    

from scipy import stats

data = data[(np.abs(stats.zscore(data[['Confirmed/test',f'Death d+{N}/test']])) < 3).all(axis=1)]
import matplotlib.pyplot as plt

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)

def scatter_countries(countries,m=None,alpha=1,scale='linear', n_countries=19):

    """

    This function plots a scatter plot using Matplotlib.

        * countries: list of Strings corresponding to values of 'Entity' in data

        * m: first parameter of the regression (optional)

        * alpha: second paramter of the regression (optional)

        * scale: String (optional) is the scale used for x values

    """

    plt.figure(figsize=(10,10))

    for country in countries[:n_countries]:

        df = data[data['Entity']==country]

        plt.scatter(df[f'Death d+{N}/test'], df['Confirmed/test'],marker='+', label=country)

    df = data[data['Entity'].apply(lambda i: i in countries[n_countries:])]

    plt.scatter(df[f'Death d+{N}/test'], df['Confirmed/test'],marker='+', label='Other')

    if m is not None:

        eps = 0.00001

        x = np.linspace(eps, data[f'Death d+{N}/test'].max(), 1000)

        plt.plot(x,m*np.exp(alpha*np.log(x)))

    plt.xscale(scale)

    plt.xlabel(f'Deaths $d+{N}$ per test')

    plt.ylabel('Positives per test')

    plt.legend()

    plt.show()



countries_plot = countries[:19]

scatter_countries(countries, n_countries=19) # Limiting number of countries for readability
import statsmodels.formula.api as sfa

from scipy.stats import pearsonr



# Linear regression without intercept between y and sqrt(x)

x = data[f'Death d+{N}/test'].values

y = data['Confirmed/test'].values

df = pd.DataFrame({'x': np.sqrt(x), 'y': y})

r = sfa.ols('y ~ x + 0', data=df).fit()



fig, ax = plt.subplots(figsize=(10, 10))

plt.xlabel("$\sqrt{x}$")

plt.ylabel("$y$")

ax.scatter(x=np.sqrt(x), y=y)

ax.plot(np.sqrt(x), r.fittedvalues)



corr, _ = pearsonr(np.sqrt(x), y)

print(f'Pearson Correlation - R^2: {corr:.4f}')
from scipy.optimize import curve_fit



def f(x,m,alpha):

    return(m*np.power(x,alpha))



popt, pcov = curve_fit(f, x, y,[1.5,0.5], bounds=(0.2,2))

m, alpha = popt

print(f"Optimal parameters found:\n\tm = {m:.4f}\n\talpha = {alpha:.4f}")

corr, _ = pearsonr(np.power(x,alpha), y)

print(f'Pearson Correlation - R^2: {corr:.4f}')



scatter_countries(countries,*popt, n_countries=19)
def pred(d,pop,popt,test_sensitivity=0.50):

    """

    This function computes a prediction of the proportion of infected people in country with:

        * d: number of deaths in the country

        * pop: population of the country

        * popt: parameters to be used in the model

        * test_precision: assumption on the probability of being tested positive, if infected.

    """

    frac = d/pop

    return(f(frac,*popt)/test_sensitivity)



# April 24 data --> predictions about April 20

off_deaths_Italy = 25549 * 1.2

pop_Italy = 60480000

print(f"In Italy: {100*pred(off_deaths_Italy,pop_Italy,popt):.2f}% infected")

off_deaths_France = 21889 * 0.8

pop_France = 66990000

print(f"In France: {100*pred(off_deaths_France,pop_France,popt):.2f}% infected")

off_deaths_NYC = 16388

pop_NYC = 8623000

print(f"In NYC: {100*pred(off_deaths_NYC,pop_NYC,popt):.2f}% infected")



print(f"\nEstimated current IFR in France: {100*(off_deaths_France/(pred(off_deaths_France,pop_France,popt)*pop_France)):.3f}%")

print(f"Estimated current IFR in NYC: {100*(off_deaths_NYC/(pred(off_deaths_NYC,pop_NYC,popt)*pop_NYC)):.3f}%")

print(f"Estimated current IFR in Italy: {100*(off_deaths_Italy/(pred(off_deaths_Italy,pop_Italy,popt)*pop_Italy)):.3f}%")
off_deaths_world = 191962

pop_world = 7794799000

print(f"World: \n\n{100*pred(off_deaths_world,pop_world,popt):.2f}% infected")

print(f"{100*(off_deaths_world/(pred(off_deaths_world,pop_world,popt)*pop_world)):.3f}% estimated current IFR")
print('At the end of March:')



off_deaths_Netherlands = 540 # At the end of March

pop_Netherlands = 17280000

print(f"\nIn the Netherlands: {100*pred(off_deaths_Netherlands,pop_Netherlands,popt):.2f}% infected")



off_deaths_SantaClara = 25 # At the end of March

pop_SantaClara = 1928000

print(f"In Santa Clara county: {100*pred(off_deaths_SantaClara,pop_SantaClara,popt):.2f}% infected")



off_deaths_LA = 617 * 7994/13816 # early April data

pop_LA = 10040000

print(f"In Los Angeles county: {100*pred(off_deaths_LA,pop_LA,popt):.2f}% infected")



print("\nAs of April 14:")

off_deaths_Stockholm = 1400 * 944/1580

pop_Stockholm = 2377081

print(f"\nIn Stockholm county: {100*pred(off_deaths_Stockholm,pop_Stockholm,popt):.2f}% infected")



print("\nIn Geneva county:")

off_deaths_Geneva_10 = 858 * 4438/27856

off_deaths_Geneva_17 = 1141 * 4438/27856

pop_geneva = 499480

print(f"\nBy April 10: {100*pred(off_deaths_Geneva_10,pop_geneva,popt):.2f}% infected")

print(f"By April 17: {100*pred(off_deaths_Geneva_17,pop_geneva,popt):.2f}% infected")



print("\nAs of April 20:")

off_deaths_NYS = 20982 # April 24 data

off_deaths_NYC = 16388

pop_NYS = 19450000

pop_NYC = 8623000

print(f"\nIn New York State: {100*pred(off_deaths_NYS,pop_NYS,popt):.2f}% infected")

print(f"In New York City: {100*pred(off_deaths_NYC,pop_NYC,popt):.2f}% infected")
belgium = data[data['Entity']=='Belgium']

x = belgium[f'Death d+{N}/test'].values

y = belgium['Confirmed/test'].values



def f(x,m,alpha):

    return(m*np.power(x,alpha))



popt_belgium, pcov_belgium = curve_fit(f, x, y,[1.5,0.5], bounds=(0.2,2))

scatter_countries(["Belgium"],*popt_belgium)

m, alpha = popt_belgium

corr, _ = pearsonr(np.power(x,alpha), y)

print(f'Pearson Correlation - R^2: {corr:.4f}')



off_deaths_Belgium = 5453 * 0.5

pop_Belgium = 11400000

print(f"\nIn Belgium: {100*pred(off_deaths_Belgium,pop_Belgium,popt_belgium):.2f}% infected")

print(f"Estimated death rate in Belgium: {100*(off_deaths_Belgium/(pred(off_deaths_Belgium,pop_Belgium,popt_belgium)*pop_Belgium)):.3f}%")

countries_to_include = ['Netherlands']

cp_data = data[data['Entity'].apply(lambda i: i in countries_to_include)]

x = cp_data[f'Death d+{N}/test'].values

y = cp_data['Confirmed/test'].values



def f(x,m,alpha):

    return(m*np.power(x,alpha))



popt_cp_data, pcov_cp_data = curve_fit(f, x, y,[1.5,0.5], bounds=(0.2,2))

scatter_countries(countries_to_include,*popt_cp_data)

m, alpha = popt_cp_data

corr, _ = pearsonr(np.power(x,alpha), y)

print(f'Pearson Correlation - R^2: {corr:.4f}')

off_deaths_Netherlands = 3601

pop_Netherlands = 17280000

print(f"\nIn the Netherlands, April 14: {100*pred(off_deaths_Netherlands,pop_Netherlands,popt_cp_data):.2f}% infected")

print(f"Estimated death rate in the Netherlands: {100*(off_deaths_Netherlands/(pred(off_deaths_Netherlands,pop_Netherlands,popt_cp_data)*pop_Netherlands)):.3f}%")

off_deaths_Netherlands = 540 # At the end of March

pop_Netherlands = 17280000

print(f"\nIn the Netherlands, end of March: {100*pred(off_deaths_Netherlands,pop_Netherlands,popt_cp_data):.2f}% infected")

print(f"Estimated death rate in the Netherlands: {100*(off_deaths_Netherlands/(pred(off_deaths_Netherlands,pop_Netherlands,popt_cp_data)*pop_Netherlands)):.3f}%")