import numpy as np

import pandas as pd



from pathlib import Path

data_path_benchmark = Path('/kaggle/input/covid19-benchmarks/')

data_path_competition = Path('/kaggle/input/covid19-global-forecasting-week-4/')

data_path_actuals = Path('/kaggle/input/covid19-models-raw-data')



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt
train = pd.read_csv(data_path_actuals / 'actuals.csv', index_col='Id')

wk4_test = pd.read_csv(data_path_competition / 'test.csv', index_col='ForecastId')



wk4_preds_fatalities = pd.read_csv(data_path_benchmark / 'wk4_preds_fatalities_selected_deduped.csv', index_col='ForecastId')



ihme_cols = ['location_name', 'date', 'deaths_mean','deaths_lower', 'deaths_upper']

ihme_1 = pd.read_csv(data_path_benchmark / 'ihme_2020_04_13.csv')[ihme_cols]

ihme_2 = pd.read_csv(data_path_benchmark / 'ihme_2020_04_16.csv')[ihme_cols]



lanl_cols = ['dates', 'q.50', 'state']

lanl_1 = pd.read_csv(data_path_benchmark / 'lanl_2020_04_12.csv')[lanl_cols]

lanl_2 = pd.read_csv(data_path_benchmark / 'lanl_2020_04_15.csv')[lanl_cols]
states = [

  'Alabama','Alaska','Arizona','Arkansas','California','Colorado',

  'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho','Illinois',

  'Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland',

  'Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana',

  'Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York',

  'North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania',

  'Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah',

  'Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming']
def plot_fatalities(state):



    fig = plt.figure(figsize=(12,8))

    ax = plt.axes()

    

    # to help with ylim

    max_preds = []



    # Actual

    select = (train['Country_Region']=='US') & (train['Province_State']==state)

    ids = train.loc[select].index.tolist()

    dates = train.loc[select, 'Date'].values

    data = train.loc[ids, 'Fatalities'].values

    plt.plot(dates, data, c='k', label='Actual', linewidth=4)

    max_preds.append(data[-1])



    # LANL 1

    lanl_ids = lanl_1['state'] == state

    dates = lanl_1.loc[lanl_ids, 'dates'].values

    data = lanl_1.loc[lanl_ids, 'q.50'].values

    plt.plot(dates, data, label=f'LANL (Apr 12)', c='g', alpha=0.5, linestyle='--')

    max_preds.append(data[-1])

    

    # IHME 1

    ihme_ids = ihme_1['location_name'] == state

    dates = ihme_1.loc[ihme_ids, 'date'].values.tolist()

    data = ihme_1.loc[ihme_ids, 'deaths_mean'].cumsum().values.tolist()

    start = dates.index('2020-03-15')

    dates = dates[start:]

    data = data[start:]

    plt.plot(dates, data, label=f'IHME (Apr 13)', alpha=0.5, c='b', linestyle='--')

    max_preds.append(data[-1])



#     # Kaggle Median

    select = (wk4_test['Country_Region']=='US') & (wk4_test['Province_State']==state)

    ids = wk4_test.loc[select].index.tolist()

    dates = wk4_test.loc[select, 'Date'].tolist()

    data = wk4_preds_fatalities.loc[ids].quantile(0.5, axis=1).values

#     plt.plot(dates[13:], data[13:], c='r', linestyle='--', label=f'Kaggle Median (Apr 14)') # predictions start on 13th row

    max_preds.append(data[-1])



    # Top four Week 3 teams

    subs = ['15210308.csv', '15210199.csv', '15208266.csv', '15210154.csv']

    data = wk4_preds_fatalities.loc[ids, subs].quantile(0.5, axis=1).values

    plt.plot(dates[13:], data[13:], c='r', label=f'Kaggle Top 4 Teams (Apr 14)') # predictions start on 13th row

    max_preds.append(data[-1])

    

    # LANL 2

    lanl_ids = lanl_2['state'] == state

    dates = lanl_2.loc[lanl_ids, 'dates'].values

    data = lanl_2.loc[lanl_ids, 'q.50'].values

    plt.plot(dates, data, label=f'LANL (Apr 15)', c='g')

    max_preds.append(data[-1])

    

    # IHME 2

    ihme_ids = ihme_1['location_name'] == state

    dates = ihme_2.loc[ihme_ids, 'date'].values.tolist()

    data = ihme_2.loc[ihme_ids, 'deaths_mean'].cumsum().values.tolist()

    start = dates.index('2020-03-15')

    dates = dates[start:]

    data = data[start:]

    plt.plot(dates, data, label=f'IHME (Apr 16)', c='b')

    max_preds.append(data[-1])

    



    

    fig.autofmt_xdate()

    ax.set_xlim(('2020-04-01', '2020-05-31'))

    ax.grid(False)

    plt.xticks(rotation=90)

    plt.title(f'{state} Fatalities (Cumulative)\n', fontsize=20)



    ylim = int(np.ceil(max(max_preds) / 100.0)) * 100 # round up to nearest 100

    ax.set_ylim(0, ylim)

    plt.legend(fontsize=14, loc=2)

    plt.show()
plot_fatalities('Wyoming')
plot_fatalities('Texas')
plot_fatalities('Ohio')
for state in states:

    plot_fatalities(state)