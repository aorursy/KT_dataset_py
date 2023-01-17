import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import seaborn as sns
import scipy
import shutil
import time
import warnings

from datetime import date, datetime
from git import Repo
from plotly.subplots import make_subplots
from scipy import stats
from tqdm import tqdm
from typing import Dict, List, Tuple

Date = datetime.date
# Progress bar features.
tqdm.pandas()

# Make random numbers and Python hashes consistent.
np.random.seed(0)
%env PYTHONHASHSEED=0

warnings.filterwarnings("ignore")
def get_date(seconds_since_epoch):
    return time.strftime("%a %d %b %Y", time.gmtime(seconds_since_epoch))
latest_data = pd.read_csv("https://raw.githubusercontent.com/nychealth/coronavirus-data/master/data-by-modzcta.csv")
latest_data
latest_data = latest_data.set_index('MODIFIED_ZCTA')
latest_data
REPO_NAME = 'repo'
repo = Repo.clone_from('https://github.com/nychealth/coronavirus-data.git', REPO_NAME, branch='master')
repo
# Master branch commit
head_commit = repo.commit('HEAD')
head_commit
get_date(head_commit.committed_date)
head_commit.committed_date
# The first date where the data is available is May 18:
# https://github.com/nychealth/coronavirus-data/commit/50e60ee5c8f36198abc9b697761128dd29f10fc9.
FIRST_DATE = date(2020, 5, 18)
LATEST_DATE = date.today()

MAX_NUM_COMMITS_BACK = (LATEST_DATE - FIRST_DATE).days + 1
print("Number of commits to process / x-points: ", MAX_NUM_COMMITS_BACK)

# Note: The date is brittly fetched as the commit date is not necessarily the date the CSV 
# info is for.
# It would be better if they had METADATA to read in the file.
dates = pd.date_range(end=get_date(head_commit.committed_date), periods=MAX_NUM_COMMITS_BACK)

# Verify that early dates appear first, and later dates appear later.
for i in range(len(dates) - 1):
    assert(dates[i] < dates[i + 1])

dates
FILE = 'https://raw.githubusercontent.com/nychealth/coronavirus-data/%s/data-by-modzcta.csv'
data_by_date = []
for i in range(MAX_NUM_COMMITS_BACK):
    # Read data from recent commits, starting with the oldest.
    past_commit = repo.commit('HEAD~' + str(MAX_NUM_COMMITS_BACK - i - 1))
    data_from_one_date = pd.read_csv(FILE % past_commit.hexsha)
    reindexed_data_by_zip = data_from_one_date.set_index('MODIFIED_ZCTA')
    data_by_date.append(reindexed_data_by_zip)
    
full_series = pd.Series(data_by_date, index=dates)
len(full_series.index)
ZIPS = [
    # UWS - 59th-76th St.
    10023,
    # UWS - 76th to 91st St.
    # 10024,
    # 48th-59th St, West side to 5th Ave.
    10019,
    # Times Sq
    10036,
    # LIC near Court Sq.
    # 11101,
    # Washington Sq Park.
    # 10012,
    # Chelsea.
    # 10011
    # East Village
    10009
]
# Inspect testing rates.
def ComputeCovidTestsPerDay(total_covid_tests_by_date: List[Tuple[Date, int]]) -> List[int]:
    test_counts = [date_and_count[1] for date_and_count in total_covid_tests_by_date]
    tests_per_day = []
    for i in range(len(test_counts) - 1):
        tests_per_day.append(test_counts[i + 1] - test_counts[i])
    return tests_per_day

# Currently not needed. May use in the future.
def LinearlyExtrapolate():
    # Column was only added in mid-June, so we linearly extrapolate backwards.
    pass
    
    # TODO: linear model.
    # Ensure we have this many entries.
    NUM_ENTRIES_EXPECTED = (LATEST_DATE - FIRST_DATE).days
    xi = range(len(pd.date_range(FIRST_DATE, LATEST_DATE)))
    print('xi:', xi)
    yi = [date_and_count[1] for date_and_count in total_covid_tests_by_date]
    print('yi:', yi)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=xi, y=yi)
    line = slope*xi + intercept
    print(line)

# Just for debugging.
def TestCovidTestPerDayFunctions():
    zcta = 10019
    total_covid_tests_by_date: List[Tuple[Date, int]] = []
    for date in dates:
        if 'TOTAL_COVID_TESTS' in full_series[date].columns:
            total_covid_tests_by_date.append((date, full_series[date].at[zcta, 'TOTAL_COVID_TESTS']))
    tests_per_day: List[int] = ComputeCovidTestsPerDay(total_covid_tests_by_date)

    xi = range(len(tests_per_day))
    print('xi:', xi)
    yi = tests_per_day
    print('yi:', yi)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=xi, y=yi)
    line = slope*xi + intercept
    print('line y-values:', line)
    print('slope:', slope)

    fig = make_subplots(rows=1, cols=2)

    y_cumulative = [date_and_count[1] for date_and_count in total_covid_tests_by_date]
    fig.append_trace(go.Scatter(
        name='Cumulative Tests',
        x=list(range(len(y_cumulative))),
        y=y_cumulative,
        mode='markers',
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        name='Covid Tests Per Day',
        x=list(range(len(tests_per_day))),
        y=tests_per_day,
        mode='markers',
    ), row=1, col=2)

    fig.update_layout(height=400, width=700, title_text=str(zcta) + " Test Count By Day")
    fig.show()
    return px.scatter(x=list(range(len(tests_per_day))), y=tests_per_day, trendline='ols')

TestCovidTestPerDayFunctions()
def ComputeCaseCountDeltasForZip(case_counts: List[int]) -> List[int]:
    deltas = []
    for i in range(len(case_counts) - 1):  # One fewer because these are deltas.
        # Get deltas between case counts. Use max() to clamp any negative numbers to 0 
        # (negative numbers indicate data corrections).
        deltas.append(max(0, case_counts[i + 1] - case_counts[i]))
    return deltas
def BuildZipToStatsDict(zctas) -> Tuple[Dict[str, int], Dict[str, int],
                                        Dict[str, int], Dict[str, int]]:
    zip_to_case_counts = {}
    zip_to_case_rates = {}
    zip_to_case_count_deltas = {}
    zip_to_covid_tests_per_day = {}
    
    for zcta in zctas:
        case_counts_for_zip = []
        case_rates_for_zip = []
        total_covid_tests_by_date: List[Tuple[Date, int]] = []
        for date in dates:
            case_counts_for_zip.append(full_series[date].at[zcta, 'COVID_CASE_COUNT'])
            case_rates_for_zip.append(full_series[date].at[zcta, 'COVID_CASE_RATE'])
            
            if 'TOTAL_COVID_TESTS' in full_series[date].columns:
                total_covid_tests_by_date.append((date, full_series[date].at[zcta, 'TOTAL_COVID_TESTS']))
        
        zip_to_case_counts[zcta] = case_counts_for_zip
        zip_to_case_rates[zcta] = case_rates_for_zip
        
        zip_to_case_count_deltas[zcta] = ComputeCaseCountDeltasForZip(case_counts_for_zip)
        zip_to_covid_tests_per_day[zcta] = ComputeCovidTestsPerDay(total_covid_tests_by_date)
        
    return (zip_to_case_counts, zip_to_case_rates, zip_to_case_count_deltas, zip_to_covid_tests_per_day)

# The Central Indexed Data.
zip_to_case_counts, zip_to_case_rates, zip_to_case_count_deltas, zip_to_covid_tests_per_day = BuildZipToStatsDict(ZIPS)
def BasicGraphZip(zcta):
    date_and_case_count_series = pd.Series(zip_to_case_counts[zcta], index=dates)
    date_and_case_rate_series = pd.Series(zip_to_case_rates[zcta], index=dates)

    combined_df = pd.concat([date_and_case_count_series, date_and_case_rate_series], axis=1)
    combined_df.plot(subplots=True, grid=True, 
                     title=['Cumulative COVID Case Count for Zip ' + str(zcta), 
                            'Cumulative COVID Case Rate per 100,000 people '],
                     legend=False)

for zcta in ZIPS:
    BasicGraphZip(zcta)
def GraphDeltas(case_count_deltas: List[int], tests_per_day: List[int], ax1, ax2, ax3):
    ## ax1: Bar Plot
    
    # First create a series indexed by date.
    assert(len(case_count_deltas) >= 20)
    start_index = len(case_count_deltas) - 20  # Inspect last n days to reduce bar plot size.
    
    # Skip 1 value because delta means we're counting the "spaces in between" counts.
    deltas_to_use = case_count_deltas[start_index:]
    delta_series = pd.Series(deltas_to_use, index=dates[start_index + 1:])
    
    delta_series.plot(kind='bar', ax=ax1)
    
    ax1.set(yticks=range(max(deltas_to_use) + 2))
    
    ## ax2: Linear Plot
    delta_df = pd.DataFrame(case_count_deltas, columns=['count'])
    # Add an artificial x column describing the index.
    delta_df['x'] = range(len(delta_df.index))
    
    # Labels.
    (r, p) = stats.pearsonr(delta_df['x'], delta_df['count'])
    r2 = np.round(r**2, 2)
    r = np.round(r, 2)
    p = np.round(p, 2)
    text = 'r=%s, r^2=%s, p=%s' % (r, r2, p)
    ax2.text(.2, .8, # x, y relative position
             text, 
             transform=ax2.transAxes,
             fontsize=14)
    
    sns.regplot(data=delta_df, x='x', y='count', fit_reg=True, ci=90, ax=ax2)
    ax2.set(yticks=range(max(case_count_deltas) + 2))
    
    ## ax3: Linear Plot for Tests per Day.
    tests_per_day_df = pd.DataFrame(tests_per_day, columns=['test_count'])
    tests_per_day_df['x'] = range(len(tests_per_day_df))
    
    # Labels.
    (r, p) = stats.pearsonr(tests_per_day_df['x'], tests_per_day_df['test_count'])
    r2 = np.round(r**2, 2)
    r = np.round(r, 2)
    p = np.round(p, 2)
    text = 'r=%s, r^2=%s, p=%s' % (r, r2, p)
    ax3.text(.2, .8, # x, y relative position
             text, 
             transform=ax3.transAxes,
             fontsize=14)
    
    sns.regplot(data=tests_per_day_df, x='x', y='test_count', fit_reg=True, ci=90, ax=ax3)
    

def GatherCombinedZipData(zip_to_metric: Dict[int, List]):
    arbitrary_entry = ZIPS[0]
    result = [0] * len(zip_to_metric[arbitrary_entry])
    print('Start: ', result)
    for zcta in ZIPS:
        print('Other: ', zip_to_metric[zcta])
        for i in range(len(zip_to_metric[zcta])):
            result[i] += zip_to_metric[zcta][i]
    return result

PREV_ZIPS_COMBINED_KEY = 'Previous Zips Combined'

print('Combining case count data.')
combined_zip_case_count_data = GatherCombinedZipData(zip_to_case_count_deltas)
print('Combined:', combined_zip_case_count_data)
zip_to_case_count_deltas[PREV_ZIPS_COMBINED_KEY] = combined_zip_case_count_data

print()

print('Combining test count data.')
combined_test_count_data = GatherCombinedZipData(zip_to_covid_tests_per_day)
print('Combined:', combined_test_count_data)
zip_to_covid_tests_per_day[PREV_ZIPS_COMBINED_KEY] = combined_test_count_data

for zcta in ZIPS + [PREV_ZIPS_COMBINED_KEY]:
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14,5))
    fig.suptitle(str(zcta) + ' Stats')

    # TODO: parameterize "num-days-back".
    GraphDeltas(zip_to_case_count_deltas[zcta], zip_to_covid_tests_per_day[zcta], ax1, ax2, ax3)
    
    ax1.set_title('Case Count Change')

    ax2.set_title('Linear plot of Case Count Change')
    ax2.set(xlabel='days')

    ax3.set_title('Linear plot of Covid Tests Administered')
    ax3.set(xlabel='days')
# Ensure the number of entries for each zip is identical.
num_entries = len(zip_to_case_count_deltas[next(iter(zip_to_case_count_deltas))])
for zip_code in zip_to_case_count_deltas:
    assert(len(zip_to_case_count_deltas[zip_code]) == num_entries)
fig = make_subplots(rows=1, cols=1)

fig.append_trace(go.Bar(
    name='10023',
    x=pd.date_range(FIRST_DATE, LATEST_DATE),
    y=zip_to_case_count_deltas[10023],
#     mode='markers',
), row=1, col=1)

# xi = range(len(pd.date_range(FIRST_DATE, LATEST_DATE)) - 1)
# slope, intercept, r_value, p_value, std_err = stats.linregress(x=xi, y=zip_to_case_count_deltas[10023])
# line = slope*xi + intercept
# print(line)

# fig.append_trace(go.Scatter(
#     name='trend line',
#     x=list(xi),
#     y=line
# ), row=1, col=1)

fig.append_trace(go.Bar(
    name='10019',
    x=pd.date_range(FIRST_DATE, LATEST_DATE),
    y=zip_to_case_count_deltas[10019],
#     mode='markers',
), row=1, col=1)


fig.update_layout(height=600, width=600, title_text="Bar Graphs of Case Counts", yaxis = {'dtick': 1})

# Gather case count data for one zip code.
TARGET_ZIP = 10023
full_series[dates[0]].at[TARGET_ZIP, 'COVID_CASE_COUNT']
case_counts_for_zip = []
for date in dates:
    case_counts_for_zip.append(full_series[date].at[TARGET_ZIP, 'COVID_CASE_COUNT'])
case_counts_for_zip
case_rates_for_zip = []
for i, date in enumerate(dates):
    case_rates_for_zip.append(full_series[date].at[TARGET_ZIP, 'COVID_CASE_RATE'])
date_and_case_count_series = pd.Series(case_counts_for_zip, index=dates)
date_and_case_rate_series = pd.Series(case_rates_for_zip, index=dates)

combined_df = pd.concat([date_and_case_count_series, date_and_case_rate_series], axis=1)
combined_df
combined_df.plot(subplots=True, grid=True, title=['Cumulative COVID Case Count for Zip ' + str(TARGET_ZIP),
                                                  'Cumulative COVID Case Rate per 100,000 people '])
date_and_case_rate_series.plot(grid=True, title='COVID Case Rate per 100,000 people', subplots=True)
date_and_case_count_series.plot(grid=True, title='COVID Case Counts for ' + str(TARGET_ZIP)) #, yticks=range(530,580,5))
case_counts_for_zip
# Get deltas between case counts. Use max() to clamp any negative numbers to 0
# (negative numbers indicate data corrections).
deltas = []
for i in range(len(case_counts_for_zip) - 1):
    deltas.append(max(0, case_counts_for_zip[i + 1] - case_counts_for_zip[i]))
deltas
# First we havea Series indexed by date. There are no column names in Series.
start = 10
# Skip 1 value because delta implies -1 data point.
delta_series = pd.Series(deltas[start:], index=dates[start + 1:])
delta_series.plot(kind='bar')
# Instead of being indexed by date, reset the index to the default whole numbers.
delta_df = delta_series.to_frame()
delta_df = delta_df.reset_index()
delta_df
delta_df = delta_df.rename(columns={'index': 'date', 0: 'count'})
delta_df.index
print(len(delta_df.index))
delta_df['x'] = range(len(delta_df.index))
delta_df
# Linear regression
g = sns.lmplot(data=delta_df, x='x', y='count')
g = g.set(yticks=range(4))
type(g)
# Plot with Pearson's correlation coefficient.
def r2(x, y):
    r, p_value = stats.pearsonr(x, y)
    return r**2

def r(x, y):
    return stats.pearsonr(x, y)[0]

def r(x, y):
    r, p_value = stats.pearsonr(x, y)
    return (r, p_value)

sns.jointplot(data=delta_df, x='x', y='count', kind="reg", stat_func=r2, xlim=(0,30), ylim=(0,5))
!ls -la
# Clean-up
dir_to_delete = REPO_NAME
try:
    shutil.rmtree(dir_to_delete)
    print('Deleted', dir_to_delete)
except OSError as e:
    print('Error: %s : %s' % (dir_to_delete, e.strerror))