import math

import numpy as np

import pandas as pd

from scipy import stats

import os



matches    = pd.read_csv('../input/matches.csv')

deliveries = pd.read_csv('../input/deliveries.csv')
print(f'Number of rows    = {len(matches)}')

print(f'Number of columns = {len(matches.columns)}')

matches.head()
win_by_runs_data = matches[matches['win_by_runs'] > 0].win_by_runs

print(f'Number of rows = {len(win_by_runs_data)}')

win_by_runs_data.head()
win_by_runs_rows = len(win_by_runs_data) # No. of values in the set (n)

win_by_runs_sum = sum(win_by_runs_data) # Sum of all numbers



print(f'Sum of all numbers = {win_by_runs_sum}, No. of values in the set = {win_by_runs_rows}')



win_by_runs_arithmetic_mean = win_by_runs_sum / win_by_runs_rows # Calculating arithmetic mean

print(f'Arithmetic mean = {win_by_runs_arithmetic_mean}')
win_by_runs_arithmetic_mean_verify = win_by_runs_data.mean()

print(f'Arithmetic mean (verify) = {win_by_runs_arithmetic_mean_verify}')
win_by_runs_geo_mean = stats.mstats.gmean(win_by_runs_data)

print(f'Geometric mean = {win_by_runs_geo_mean}')
win_by_runs_10 = list(win_by_runs_data[:10])

print(win_by_runs_10)

print(sorted(win_by_runs_10))
win_by_runs_10_median = win_by_runs_data[:10].median()

print(f'Median (first 10) = {win_by_runs_10_median}')



win_by_runs_median = win_by_runs_data.median()

print(f'Median = {win_by_runs_median}')
# Retrieve frequency (sorted, descending order)

win_by_runs_data.value_counts(sort=True, ascending=False).head()
win_by_runs_data_mode = win_by_runs_data.mode()

print(f'Mode = {list(win_by_runs_data_mode)}')
win_by_runs_max = win_by_runs_data.max()

win_by_runs_min = win_by_runs_data.min()

win_by_runs_range = win_by_runs_max - win_by_runs_min



print(f'Largest = {win_by_runs_max}, Smallest = {win_by_runs_min}, Range = {win_by_runs_range}')
win_by_runs_25_perc = stats.scoreatpercentile(win_by_runs_data, 25)

win_by_runs_75_perc = stats.scoreatpercentile(win_by_runs_data, 75)



win_by_runs_iqr = stats.iqr(win_by_runs_data)

print(f'Q1 (25th percentile) = {win_by_runs_25_perc}')

print(f'Q3 (75th percentile) = {win_by_runs_75_perc}')

print(f'IQR = Q3 - Q1 = {win_by_runs_75_perc} - {win_by_runs_25_perc} = {win_by_runs_iqr}')
win_by_runs_95_perc = stats.scoreatpercentile(win_by_runs_data, 95)

print(f'95th percentile = {win_by_runs_95_perc}')
win_by_wickets_data = matches[matches.win_by_wickets > 0].win_by_wickets

print(f'Number of rows = {len(win_by_wickets_data)}')

win_by_wickets_data.head()
# Step 1: calculate mean(μ)

win_by_wickets_mean = win_by_wickets_data.mean()

print(f'Mean = {win_by_wickets_mean}')



# Step 2: calculate numerator part - sum of (x - mean)

win_by_wickets_var_numerator = sum([(x - win_by_wickets_mean) ** 2 for x in win_by_wickets_data])



# Step 3: calculate variane

win_by_wickets_variance = win_by_wickets_var_numerator / len(win_by_wickets_data)

print(f'Variance = {win_by_wickets_variance}')



# Step 4: calculate standard deviation

win_by_wickets_standard_deviation = math.sqrt(win_by_wickets_variance)

print(f'Standard deviation = {win_by_wickets_standard_deviation}')
win_by_wickets_standard_deviation_verify = win_by_wickets_data.std(ddof = 0)

print(f'Standard deviation = {win_by_wickets_standard_deviation_verify}')
win_by_runs_std = win_by_runs_data.std(ddof = 0)

print(f'| Mean               = {win_by_runs_arithmetic_mean} | Median  = {win_by_runs_median} |')

print(f'| Standard deviation = {win_by_runs_std} | IQR     = {win_by_runs_iqr} |')
win_by_wickets_dist = win_by_wickets_data.value_counts(sort=False)

plt = win_by_wickets_dist.plot.bar(color='lightblue')

plt.axvline(x = win_by_wickets_mean - 1, color='blue', linewidth=2.0)

plt.axvline(x = win_by_wickets_mean - win_by_wickets_standard_deviation - 1, color='red', linewidth=2.0, linestyle='dashed')

plt.axvline(x = win_by_wickets_mean + win_by_wickets_standard_deviation - 1, color='red', linewidth=2.0, linestyle='dashed')
win_by_runs_mad = win_by_runs_data.mad()

print(f'Mean absolute deviation = {win_by_runs_mad}')
plt = win_by_runs_data.to_frame().boxplot(whis='range', vert=False)

plt.set_xlim([0, 200])

plt.set_xlabel('Win by runs')
win_by_runs_data.to_frame().boxplot(vert=False)