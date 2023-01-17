import pandas as pd
input_file_dir = '../input/giprojekatoutput1/'
df = pd.read_csv(input_file_dir + 'output.csv', index_col=None)
import re

from datetime import timedelta
def parseTimedelta(s):

    if s is None:

        return None

    

    d = re.match(

            r'(?P<days>\d+) days (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)\.(?P<microseconds>\d+)',

            str(s)

        ).groupdict(0)

    

    return timedelta(**dict(( (key, float(value)) for key, value in d.items() )))
df['time_us'] = df['time'].apply(lambda x: parseTimedelta(x)).apply(lambda x: x.seconds * 1000000 + x.microseconds)
df = df[['t_len', 'p_len', 'sa_factor', 'tally_factor', 'time_us', 'mem']]



df
df['scaled_time_s'] = df.apply(lambda row: row['time_us'] * row['p_len'] / 1000000, axis=1)

df['scaled_mem'] = df.apply(lambda row: row['mem'] / row['t_len'], axis=1)



processed_df = df[['sa_factor', 'tally_factor', 'scaled_time_s', 'scaled_mem']].groupby(by=['sa_factor', 'tally_factor']).mean().reset_index()

df.drop(columns=['scaled_time_s', 'scaled_mem'])



processed_df
original_scaled_time_s = processed_df['scaled_time_s'].loc[0]

original_scaled_mem = processed_df['scaled_mem'].loc[0]



processed_df['time_increase'] = processed_df['scaled_time_s'].apply(lambda x: x / original_scaled_time_s)

processed_df['mem_improvement'] = processed_df['scaled_mem'].apply(lambda x: original_scaled_mem / x)



processed_df
min_time_increase = processed_df['time_increase'].min()

max_time_increase = processed_df['time_increase'].max()



min_mem_improvement = processed_df['mem_improvement'].min()

max_mem_improvement = processed_df['mem_improvement'].max()



processed_df['norm_time_increase'] = processed_df['time_increase'].apply(lambda x: (x - min_time_increase) / (max_time_increase - min_time_increase))

processed_df['norm_mem_improvement'] = processed_df['mem_improvement'].apply(lambda x: (x - min_mem_improvement) / (max_mem_improvement - min_mem_improvement))



processed_df['trade_off'] = processed_df.apply(lambda row: 2 * (1 - row['norm_time_increase']) * row['norm_mem_improvement'] / ((1 - row['norm_time_increase']) + row['norm_mem_improvement']), axis=1)



processed_df
from IPython.display import FileLink
processed_df.to_csv('processed_output.csv', index=None)

FileLink('processed_output.csv')
import matplotlib.pyplot as plt

import seaborn as sns
g = sns.FacetGrid(processed_df.rename(columns={'time_increase': 'Time increase ratio'}), col='sa_factor', height=6, aspect=.5)

g.map(sns.barplot, 'tally_factor', 'Time increase ratio');

plt.show()
g = sns.FacetGrid(processed_df.rename(columns={'mem_improvement': 'Memory improvement ratio'}), col='sa_factor', height=6, aspect=.5)

g.map(sns.barplot, 'tally_factor', 'Memory improvement ratio');

plt.show()
g = sns.FacetGrid(processed_df.rename(columns={'trade_off': 'Trade-off'}), col='sa_factor', height=6, aspect=.5)

g.map(sns.barplot, 'tally_factor', 'Trade-off');

plt.show()