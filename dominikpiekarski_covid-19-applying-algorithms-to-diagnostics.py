import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime

%matplotlib inline
df_confirmed = pd.read_csv('../input/covid-19-cssegisanddata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv('../input/covid-19-cssegisanddata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_recovered = pd.read_csv('../input/covid-19-cssegisanddata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
df_active = pd.DataFrame([]).append(df_confirmed)
df_active['4/1/20'] = df_active['4/1/20'].sub(df_deaths['4/1/20'])
df_active['4/1/20'] = df_active['4/1/20'].sub(df_recovered['4/1/20'])
df_population = pd.read_csv('../input/world-population-united-nations/world_population_data.csv')
df_population['pop2020'] = (df_population['pop2020'] * 1000).astype(int)
number_of_cases = 2000
population_size = 200000
infected_fraction = number_of_cases / population_size
np.random.seed(0)
data_set = np.array(np.random.binomial(1, infected_fraction, population_size))
data_set
# This may add some error for smaller datasets as it fillls last row with 0s
def split_padded(a,n):
    padding = (-len(a))%n
    return np.array(np.split(np.concatenate((a,np.zeros(padding))),n))

def format_percent(n):
    return "%.2f" % (n * 100) + "%"

def calc_test_data(batches, data_set_length):
    batches, tests_made, iterations = batches
    print(f'Results for test fraction of {format_percent(test_fraction)}')
    print('')
    print('Input data:')
    print('')
    print(f'Population: {data_set_length}')    
    print(f'Test fraction: {format_percent(test_fraction)} - on first batch we\'re testing {round(1/test_fraction)} people x ~{round(data_set_length/(1/test_fraction))} batches')
    print(f'Infected people: {number_of_cases} - this gives us {format_percent(infected_fraction)} infected fraction')
    print('')
    print('Outputs:')
    print('')
    print(f'Iterations: {iterations}')
    print(f'Tests made: {tests_made}')
    print('')
    print(f'Fraction of how many tests have we used, compared to whole population: {format_percent(tests_made / data_set_length)}')
def testing(test_fraction, data, iterations=1):
    population = data
    tests_made = 0
    population_number = len(population)
    tests_amount = math.ceil(population_number * test_fraction)
    if(tests_amount < 2):
        tests_amount = 2
    population_splitted = split_padded(population, tests_amount)
    tests_made += tests_amount
    population_splitted = np.array(population_splitted)
    condition = np.any(population_splitted, axis=1)
    infected_batches = population_splitted[condition]
    batches_to_return = []
    if (len(infected_batches) and len(infected_batches[0]) > 1):
        for batch in infected_batches:
            inner_batches_to_return, inner_tests_made, inner_iterations = testing(
                test_fraction, batch, iterations+1
            )
            batches_to_return = np.append(batches_to_return, inner_batches_to_return)
            tests_made += inner_tests_made
        return batches_to_return, tests_made, inner_iterations
    return infected_batches, math.ceil(tests_made), iterations
test_fraction = 0.01
res = testing(test_fraction, data_set)
calc_test_data(res, len(data_set))
def testing_alt(test_fraction, infected_fraction, population_size):
    np.random.seed(0)
    data_set = np.random.binomial(1, infected_fraction, population_size)
    return testing(test_fraction, data_set)

def generate_df_from_all(infected_fraction, population_size, start=0, end=0.1, step=0.002, custom_breakpoints=[]):
    breakpoints = np.arange(start, end, step) + 0 if start > 0 else step
    if(len(custom_breakpoints) > 0):
        breakpoints = custom_breakpoints
    df = pd.DataFrame([])
    for test_fraction in breakpoints:
        batches, tests_amount, iterations = testing_alt(test_fraction, infected_fraction, population_size)
        df = df.append(
            pd.DataFrame(
                [[test_fraction, tests_amount, iterations]], 
                columns=['Test fraction', 'Test Amount', 'Iterations'],
            ),
            ignore_index=True
        )
    return df

def create_plot(df, country, population, test_amount_divisor=1):
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(16, 8))
    fig.suptitle(country, fontsize=16, fontweight=700)
    ax.plot(df['Test fraction'], df['Test Amount'] / test_amount_divisor)
    ax.plot(df['Test fraction'], df['Iterations'])
    
    empty_str = ''
    test_amount_divisor_str = f'(in {test_amount_divisor}s)'
    ax.set(
        title="Logarithmic",
        yscale="log",
        ylabel=f'Iterations / Test Amount{test_amount_divisor_str if test_amount_divisor > 1 else empty_str}',
        xlabel="Fraction of population at which we\'re testing",
    )
    
    ax1.plot(df['Test fraction'], df['Test Amount'] / test_amount_divisor)
    ax1.plot(df['Test fraction'], df['Iterations'])
    
    ax1.set(
        title="Linear",
        ylabel=f'Iterations / Test Amount{test_amount_divisor_str if test_amount_divisor > 1 else empty_str}',
        xlabel="Fraction of population at which we\'re testing",
    )
    
    first = df.iloc[0]
    last = df.iloc[df.shape[0] - 1]
    tf_first = first['Test fraction']
    ta_first = first['Test Amount']
    it_first = first['Iterations']
    tf_last = last['Test fraction']
    ta_last = last['Test Amount']
    it_last = last['Iterations']
    
    print('Summary:')
    print('')
    print('First DataFrame row')
    print(f'Test fraction:{tf_first * 100}%, Test Amount: {int(ta_first)}, Iterations: {int(it_first)}')
    print(f'Fraction of how many people have we tested, compared to whole population: {format_percent(ta_first/ population)}')
    print('')
    print('Last DataFrame row')
    print(f'Test fraction: {tf_last * 100}%, Test Amount: {int(ta_last)}, Iterations: {int(it_last)}')    
    print(f'Fraction of how many tests have we used, compared to whole population: {format_percent(ta_last/ population)}')
    return fig
test_fraction = 0.05
calc_test_data(testing_alt(test_fraction, infected_fraction, population_size), population_size)
test_fraction = 0.1
calc_test_data(testing_alt(test_fraction, infected_fraction, population_size), population_size)
test_fraction = 0.25
calc_test_data(testing_alt(test_fraction, infected_fraction, population_size), population_size)
old_infected = infected_fraction
infected_fraction = 0.5
test_fraction = 0.02
calc_test_data(testing_alt(test_fraction, infected_fraction, population_size), population_size)
infected_fraction = old_infected
def get_number_of_cases(country):
    data = df_confirmed[df_confirmed['Country/Region'] == country]
    return data['4/1/20'].to_list()[0]

def get_population(cca2):
    return df_population[df_population['cca2'] == cca2]['pop2020'].to_list()[0]

start = 0.00125
stop = 0.08
step = 0.005

df = generate_df_from_all(
    infected_fraction, 
    population_size,
    start,
    stop,
    step,
)

df
create_plot(df, "Example country", population_size, 1000);
def prep_data(country_name, country_code):
    number_of_cases = get_number_of_cases(country_name)
    population_size = get_population(country_code)
    infected_fraction = (number_of_cases / population_size) * 4
    start = 0.00125
    stop = 0.08
    step = 0.005
    
    df = generate_df_from_all(
        infected_fraction, 
        population_size,
        start,
        stop,
        step,
    )

    return df, population_size
df, population = prep_data('Poland', 'PL')
df
create_plot(df, 'Poland', population, 1000);
df, population = prep_data('Italy', 'IT')
df
create_plot(df, 'Italy', population, 1000);
df, population = prep_data('Spain', 'ES')
df
create_plot(df, 'Spain', population, 1000);
df, population = prep_data('France', 'FR')
df
create_plot(df, 'France', population, 1000);
df, population = prep_data('Germany', 'DE')
df
create_plot(df, 'Germany', population, 1000);
df, population = prep_data('US', 'US')
df
create_plot(df, 'United States', population, 1000);