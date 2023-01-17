# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import random

from scipy.stats import ttest_ind



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
calendar = pd.read_csv(os.path.join(dirname, "calendar.csv"))

calendar.tail(5)
calendar[~pd.isna(calendar['event_name_1']) & ~pd.isna(calendar['event_name_2'])]
calendar['event_name_1'].unique()
sales = pd.read_csv(os.path.join(dirname, "sales_train_validation.csv"))

print(sales.shape)

print(sales['item_id'].unique().shape)

print(sales['store_id'].unique().shape)
sales.head()
def get_event_days(calendar):

    events = {i:[] for i in np.concatenate((calendar['event_name_1'].unique(), calendar['event_name_2'].unique())) if not pd.isna(i)}

    for event in events.keys():

        event_days = calendar.iloc[np.where((calendar['event_name_1'] == event) | (calendar['event_name_2'] == event))]['d'].tolist()

        events[event] = event_days

    return events
events = get_event_days(calendar)

print(events['SuperBowl'])
def get_event_range(events, offset = -14, possible_days = None):

    events_range = {}

    for event, days in events.items():

        events_range[event] = []

        days = [int(float(day[2:])) for day in days]

        if offset < 0:

            for day in days:

                e_range = ["d_" + str(i) for i in list(range(max(1, day + offset), day + 1))]

                events_range[event].extend(e_range)

        else:

            for day in days:

                e_range = ["d_" + str(i) for i in list(range(day, day + offset + 1))]

                events_range[event].extend(e_range)

        if possible_days is not None:

            events_range[event] = [i for i in events_range[event] if i in possible_days]

    return events_range   
seven_day_after = get_event_range(events, 7, sales.columns)

seven_day_before = get_event_range(events, -7, sales.columns)

one_month_after = get_event_range(events, 30, sales.columns)

one_month_before = get_event_range(events, -30, sales.columns)

one_month_twoway = {event: sorted(list(set(one_month_before[event] + one_month_after[event]))) for event in events.keys()}
def get_nonevent_days(events, non_avail_days, all_days, count = None):

    #events: is a dictionary of event names to list of observance of each event

    #non_avail_days: list of days for each event that we don't want to consider as non-event (dictionary formatted again)

    #all_days: list of all possible days in the dataset

    #count: how many non-event days per event should be sampled. if count is none, len of days in "events" is used.

    all_days = set(all_days)

    non_events = {}

    for event in events.keys():

        all_nonevent = all_days - set(non_avail_days[event])

        if count is None:

            count = len(events[event])

        non_events[event] = list(random.sample(all_nonevent, count))

    return non_events

        
nonevents = get_nonevent_days(seven_day_before, one_month_twoway, list(sales.columns)[6:])
print(len(nonevents['SuperBowl']))

print(len(events['SuperBowl']))

print(len(one_month_before['SuperBowl']))

print(len(one_month_twoway['SuperBowl']))
event_range = seven_day_before

nonevents = get_nonevent_days(event_range, one_month_twoway, list(sales.columns)[6:])

event_item_significance = pd.DataFrame({'item_id':sales['item_id'].unique()})

comparison_dfs = {}

for event in events.keys():

    sales_event = sales[['item_id'] + event_range[event]]

    sales_nonevent = sales[['item_id'] + nonevents[event]]

    sum_event= sales_event.groupby('item_id').sum()

    sum_event['event_sum'] = sum_event.sum(axis = 1)

    sum_nonevent = sales_nonevent.groupby('item_id').sum()#

    sum_nonevent['nonevent_sum'] = sum_nonevent.sum(axis = 1)

    sum_event.reset_index(inplace = True)

    sum_nonevent.reset_index(inplace = True)

    sales_comp = sum_event.merge(sum_nonevent)

    #sum_event.head()

    test_stat, pvals = ttest_ind(sales_comp[event_range[event]], sales_comp[nonevents[event]], axis = 1, equal_var = False)

    sales_comp['test_stat'] = test_stat

    sales_comp['pval_' + event] = pvals

    comparison_dfs[event] = sales_comp

    event_item_significance = event_item_significance.merge(sales_comp[['item_id', 'pval_' + event]], on = "item_id")
event_item_significance.head()
def plot_item(event, item_index):

    item_id = comparison_dfs[event].iloc[item_index]['item_id']

    item = pd.DataFrame({'event_sale': comparison_dfs[event].iloc[item_index][event_range[event]].values,

                        'nonevent_sale': comparison_dfs[event].iloc[item_index][nonevents[event]].values})

    plt.boxplot(item.T, labels = ['event', 'nonevent'])

    plt.title("sale of " + item_id + " wrt " + event + " pval:" + str(event_item_significance[event_item_significance['item_id'] == item_id]['pval_' + event].values))

    #item.head()
event = 'SuperBowl'

item_index = 5

plot_item(event, item_index)
event = 'SuperBowl'

item_index = 1000

plot_item(event, item_index)
event = 'SuperBowl'

item_index = 3000

plot_item(event, item_index)
event = 'Easter'

item_index = 1000

plot_item(event, item_index)