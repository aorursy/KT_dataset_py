from collections import Counter
from functools import reduce
import operator
from bisect import bisect_left

import numpy as np
import pandas as pd

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.options.display.max_rows = 500
pd.set_option('display.max_colwidth', None)
orders = pd.read_csv('order_brush_order.csv')
orders.head()
orders.info()
for col in orders:
    print(f'{col} nunique: ',orders[col].nunique())
    
for col in orders:
    print(f'{col} duplicated: ',sum(orders[col].duplicated()))
orders.head()
orders[orders.groupby('shopid').event_time.apply(pd.Series.duplicated,keep=False)].sort_values('event_time')
orders['event_time'] = pd.to_datetime(orders['event_time']) # to ensure proper sorting, not necessary but to be safe
orders.dtypes
# sort for easy debugging when comparing against kaggle examples 
orders_sorted = orders.sort_values(['shopid','event_time'])
orders_sorted.head(100)
# good for preventing repeated time spent on groupby, but cannot slice groupby object to estimate time during full run
shop_gb = orders_sorted.groupby(['shopid'])  
test_cases = {1:8996761,
              2:27121667,
              3:145777302,
              4:181009364
             }

order_shop = shop_gb.get_group(test_cases[3])

# imagine order_shop df is passed in to this apply func
event_times = order_shop['event_time'].values
array_length = len(event_times)
counter_list = []


for start_idx, start_time in enumerate(event_times[:-2]):
    counts_for_start_time = {}
    
    max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
    max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time
    
    if max_end_idx < start_idx + 2:
        continue # no need to continue if cannot form at least 3 rows
    
    
    if start_idx:  
        left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
        min_end_time = max_end_time - left_timeshift_possible  
        
        # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
        min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time)-1) 

        print('start: {}  min_end: {}  max_end: {}'.format(start_idx,min_end_idx,max_end_idx))
    else:
        min_end_idx = start_idx + 2
    
    for window_tail_idx in range(min_end_idx, max_end_idx + 1):
        current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
        concentration_ratio = len(current_window)/current_window['userid'].nunique()
        
        if concentration_ratio >= 3: 

            counts_for_start_time.update(dict(current_window['userid'].value_counts()))
            
            event_times[start_idx-1]
            event_times[window_tail_idx+1]
            current_window
            counts_for_start_time
        
    # prevent appending empty counts_for_start_time (for clean debugging prints, doesn't affect Counter summation later)
    if counts_for_start_time:
        counter_list.append(Counter(counts_for_start_time))  # prepare Counter type for accumulation later
        counter_list
            
                
if counter_list:  # if not empty [{}] for shops with no brushing:
    reduced_counter_list = reduce(operator.add,counter_list)
    reduced_counter_list
    max_value = max(reduced_counter_list.values())
    users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)
    print('FINAL ANSWER BELOW')
    '&'.join(map(str,users))
else:
    '0'
test_cases = {1:8996761,
              2:27121667,
              3:145777302,
              4:181009364
             }

order_shop = shop_gb.get_group(test_cases[3])

# imagine order_shop df is passed in to this apply func
event_times = order_shop['event_time'].values
array_length = len(event_times)
counter_list = []


for start_idx, start_time in enumerate(event_times[:-2]):
    
    max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
    max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time
    
    if max_end_idx < start_idx + 2:
        continue # no need to continue if cannot form at least 3 rows
    
    
    if start_idx:  
        left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
        min_end_time = max_end_time - left_timeshift_possible  
        
        # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
        min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time)-1) 

        print('start: {}  min_end: {}  max_end: {}'.format(start_idx,min_end_idx,max_end_idx))
    else:
        min_end_idx = start_idx + 2
    
    for window_tail_idx in range(min_end_idx, max_end_idx + 1):
        current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
        concentration_ratio = len(current_window)/current_window['userid'].nunique()
        
        if concentration_ratio >= 3: 
            counter_list.append(Counter(current_window['userid']))
            
            event_times[start_idx-1]
            event_times[window_tail_idx+1]
            current_window
        
                
if counter_list:  # if not empty [{}] for shops with no brushing:
    reduced_counter_list = reduce(operator.add,counter_list)
    reduced_counter_list
    max_value = max(reduced_counter_list.values())
    users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)
    print('FINAL ANSWER BELOW')
    '&'.join(map(str,users))
else:
    '0'
test_cases = {1:8996761,
              2:27121667,
              3:145777302,
              4:181009364
             }

order_shop = shop_gb.get_group(test_cases[1])

# imagine order_shop df is passed in to this apply func
event_times = order_shop['event_time'].values
array_length = len(event_times)
user_set = set()


for start_idx, start_time in enumerate(event_times[:-2]):
    
    max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
    max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time
    
    start_idx, max_end_idx
    
    if max_end_idx < start_idx + 2:
        print('skip')
        continue # no need to continue if cannot form at least 3 rows
    
    start_idx,max_end_idx
    
    if start_idx:  
        left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
        min_end_time = max_end_time - left_timeshift_possible  
        
        # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
        min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time) - 1) 

        print('min_end: {} max_end: {}'.format(min_end_idx,max_end_idx))
    else:
        min_end_idx = start_idx + 2
    
    for window_tail_idx in range(min_end_idx, max_end_idx + 1):
        current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
        concentration_ratio = len(current_window)/current_window['userid'].nunique()
        
        if concentration_ratio >= 3: 
            current_window_counts = Counter(current_window['userid'])

            max_value = max(current_window_counts.values())
            user_set.update(user for user, count in current_window_counts.items() if count ==  max_value)
            
            current_window
            current_window_counts
                
if user_set:  # if not empty [{}] for shops with no brushing:
    users = sorted(user_set)
    print(users)
    print('FINAL ANSWER BELOW')
    '&'.join(map(str,users))
else:
    '0'
    
    # ADD RETURN STATEMENT WHEN PASTING INTO FUNCTION
test_cases = {1:8996761,
              2:27121667,
              3:145777302,
              4:181009364,
              5:155143347,
              6:156883302
             }

order_shop = shop_gb.get_group(test_cases[6])

# imagine order_shop df is passed in to this apply func
event_times = order_shop['event_time'].values
array_length = len(event_times)
counter_list = []


for start_idx, start_time in enumerate(event_times[:-2]):
    counts_for_start_time = {}
    
    max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
    max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time
    
    if max_end_idx < start_idx + 2:
        continue # no need to continue if cannot form at least 3 rows
    
    
    if start_idx:  
        left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
        min_end_time = max_end_time - left_timeshift_possible  

        bisected_idx = bisect_left(event_times, min_end_time)
        # short-circuit prevents IndexError when event_times[bisected_idx] after or 
        if bisected_idx == array_length or event_times[bisected_idx] >  min_end_time:
            bisected_idx -= 1
            while bisected_idx > start_idx+2 and event_times[bisected_idx-1] == event_times[bisected_idx]: 
                bisected_idx -= 1
        # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
        min_end_idx = max(start_idx + 2,bisected_idx) 
    #   print('start: {}  min_end: {}  max_end: {}'.format(start_idx,min_end_idx,max_end_idx))
    else:
        min_end_idx = start_idx + 2
    
    for window_tail_idx in range(min_end_idx, max_end_idx + 1):
        current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
        concentration_ratio = len(current_window)/current_window['userid'].nunique()
        
        if concentration_ratio >= 3: 

            counts_for_start_time.update(dict(current_window['userid'].value_counts()))
            
            event_times[start_idx-1]
            event_times[window_tail_idx+1]
            current_window
            counts_for_start_time
        
    # prevent appending empty counts_for_start_time (for clean debugging prints, doesn't affect Counter summation later)
    if counts_for_start_time:
        counter_list.append(Counter(counts_for_start_time))  # prepare Counter type for accumulation later
        counter_list
            
                
if counter_list:  # if not empty [{}] for shops with no brushing:
    reduced_counter_list = reduce(operator.add,counter_list)
    reduced_counter_list
    max_value = max(reduced_counter_list.values())
    users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)
    print('FINAL ANSWER BELOW')
    '&'.join(map(str,users))
else:
    '0'
test_cases = {1:8996761,
              2:27121667,
              3:145777302,
              4:181009364,
              5:155143347,
              6:156883302
             }

order_shop = shop_gb.get_group(test_cases[5])

# imagine order_shop df is passed in to this apply func
event_times = order_shop['event_time'].values
array_length = len(event_times)

order_user = {}

for start_idx, start_time in enumerate(event_times[:-2]):
    
    max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
    max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time
    
    if max_end_idx < start_idx + 2:
        continue # no need to continue if cannot form at least 3 rows
    
    
    if start_idx:  
        left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
        min_end_time = max_end_time - left_timeshift_possible  

        bisected_idx = bisect_left(event_times, min_end_time)
        # short-circuit prevents IndexError when event_times[bisected_idx] after or 
        if bisected_idx == array_length or event_times[bisected_idx] >  min_end_time:
            bisected_idx -= 1
            while bisected_idx > start_idx+2 and event_times[bisected_idx-1] == event_times[bisected_idx]: 
                bisected_idx -= 1
        # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
        min_end_idx = max(start_idx + 2,bisected_idx) 
    #   print('start: {}  min_end: {}  max_end: {}'.format(start_idx,min_end_idx,max_end_idx))
    else:
        min_end_idx = start_idx + 2
    
    for window_tail_idx in range(min_end_idx, max_end_idx + 1):
        current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
        concentration_ratio = len(current_window)/current_window['userid'].nunique()
        
        if concentration_ratio >= 3: 
            
            order_user.update(dict(zip(current_window['orderid'],current_window['userid'])))            
            
            event_times[start_idx-1]
            event_times[window_tail_idx+1]
            current_window
        

if order_user:
    user_counts = {userid:list(order_user.values()).count(userid) for userid in set(order_user.values())}
    max_value = max(user_counts.values())
    users = sorted(user for user,count in user_counts.items() if count == max_value)
    print('FINAL ANSWER BELOW')
    '&'.join(map(str,users))
else:
    '0'
def find_brush_enum_window_aggregate(order_shop):
    
    event_times = order_shop['event_time'].values
    array_length = len(event_times)
    counter_list = []


    for start_idx, start_time in enumerate(event_times[:-2]):
        counts_for_start_time = {}

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows

        if start_idx:  
            left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
            min_end_time = max_end_time - left_timeshift_possible  

            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time)-1) 
        else:
            # no row before start_idx == 0, so no restriction from bisected_idx
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                counts_for_start_time.update(dict(current_window['userid'].value_counts()))
        

        # prevent appending empty counts_for_start_time (for clean debugging prints, doesn't affect Counter summation later)
        if counts_for_start_time:
            counter_list.append(Counter(counts_for_start_time))  # prepare Counter type for accumulation later

    if counter_list:  # if not empty [{}] for shops with no brushing:
        reduced_counter_list = reduce(operator.add,counter_list)
        max_value = max(reduced_counter_list.values())
        users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)
        return '&'.join(map(str,users))
    else:
        return '0'
#result_enum_window_aggregate = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window_aggregate)
#result_enum_window_aggregate = result_enum_window_aggregate.reset_index(name='userid')
#result_enum_window_aggregate.to_csv('enum_window_aggregate.csv',index=False)
def find_brush_enum_window_no_update(order_shop):
    event_times = order_shop['event_time'].values
    array_length = len(event_times)
    counter_list = []


    for start_idx, start_time in enumerate(event_times[:-2]):

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows


        if start_idx:  
            left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
            min_end_time = max_end_time - left_timeshift_possible  

            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time)-1)
            
        else:
            # no row before start_idx == 0, so no restriction from bisected_idx
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                counter_list.append(Counter(current_window['userid']))


    if counter_list:  # if not empty [{}] for shops with no brushing:
        reduced_counter_list = reduce(operator.add,counter_list)
        max_value = max(reduced_counter_list.values())
        users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)

        return '&'.join(map(str,users))
    else:
        return '0'
#result_enum_window_no_update = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window_no_update)
#result_enum_window_no_update = result_enum_window_no_update.reset_index(name='userid')
#result_enum_window_no_update.to_csv('enum_window_no_update.csv',index=False)
def find_brush_enum_window(order_shop):


    event_times = order_shop['event_time'].values
    array_length = len(event_times)
    user_set = set()


    for start_idx, start_time in enumerate(event_times[:-2]):

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows

        if start_idx:  
            left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
            min_end_time = max_end_time - left_timeshift_possible  

            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2, bisect_left(event_times, min_end_time)-1)

        else:
            # no row before start_idx == 0, so no restriction from bisected_idx
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                current_window_counts = Counter(current_window['userid'])
                current_window_counts
                max_value = max(current_window_counts.values())
                user_set.update(user for user, count in current_window_counts.items() if count ==  max_value)


    if user_set:  # if not empty [{}] for shops with no brushing:
        users = sorted(user_set)
        return '&'.join(map(str,users))
    else:
        return '0'
#result_enum_window = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window)
#result_enum_window = result_enum_window.reset_index(name='userid')
#result_enum_window.to_csv('enum_window.csv',index=False)
def find_brush_enum_window_bisect(order_shop):
    event_times = order_shop['event_time'].values
    array_length = len(event_times)
    counter_list = []


    for start_idx, start_time in enumerate(event_times[:-2]):
        counts_for_start_time = {}

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows


        if start_idx:  
            left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
            min_end_time = max_end_time - left_timeshift_possible  

            bisected_idx = bisect_left(event_times, min_end_time)
            
            # short-circuit prevents IndexError when event_times[bisected_idx] after or 
            if bisected_idx == array_length or event_times[bisected_idx] >  min_end_time:
                bisected_idx -= 1
                while bisected_idx > start_idx+2 and event_times[bisected_idx-1] == event_times[bisected_idx]: 
                    bisected_idx -= 1
            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2,bisected_idx) 

        else:
            # no row before start_idx == 0, so no restriction from bisected_idx 
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                counts_for_start_time.update(dict(current_window['userid'].value_counts()))
                
        # prevent appending empty counts_for_start_time (for clean debugging prints, doesn't affect Counter summation later)
        if counts_for_start_time:
            counter_list.append(Counter(counts_for_start_time))  # prepare Counter type for accumulation later

    if counter_list:  # if not empty [{}] for shops with no brushing:
        reduced_counter_list = reduce(operator.add,counter_list)
        max_value = max(reduced_counter_list.values())
        users = sorted(user for user,count in reduced_counter_list.items() if count == max_value)
        return '&'.join(map(str,users))
    else:
        return '0'
#result_enum_window_bisect = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window_bisect)
#result_enum_window_bisect = result_enum_window_bisect.reset_index(name='userid')
#result_enum_window_bisect.to_csv('enum_window_bisect.csv',index=False)
def find_brush_enum_window_dedup(order_shop):
    
    event_times = order_shop['event_time'].values
    array_length = len(event_times)

    order_user = {}

    for start_idx, start_time in enumerate(event_times[:-2]):

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows


        if start_idx:  
            left_timeshift_possible = start_time - event_times[start_idx-1]  # can be handled by df.diff outside apply
            min_end_time = max_end_time - left_timeshift_possible  

            bisected_idx = bisect_left(event_times, min_end_time)
            # short-circuit prevents IndexError when event_times[bisected_idx] after or 
            if bisected_idx == array_length or event_times[bisected_idx] >  min_end_time:
                bisected_idx -= 1
                while bisected_idx > start_idx+2 and event_times[bisected_idx-1] == event_times[bisected_idx]: 
                    bisected_idx -= 1
            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2,bisected_idx) 

        else:
            # no row before start_idx == 0, so no restriction from bisected_idx
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                order_user.update(dict(zip(current_window['orderid'],current_window['userid'])))            


    if order_user:
        user_counts = {userid:list(order_user.values()).count(userid) for userid in set(order_user.values())}
        max_value = max(user_counts.values())
        users = sorted(user for user,count in user_counts.items() if count == max_value)
        
        return '&'.join(map(str,users))
    else:
        return '0'
result_enum_window_dedup = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window_dedup)
result_enum_window_dedup = result_enum_window_dedup.reset_index(name='userid')
result_enum_window_dedup.to_csv('enum_window_dedup.csv',index=False)
def find_brush_enum_window_dedup_diff(order_shop):
    
    event_times = order_shop['event_time'].values
    array_length = len(event_times)
    
    order_user = {}
    # insert to shift right 1 position for natural indexing using start_idx
    event_times_diff = np.insert(np.diff(event_times),values=0,obj=0)
    
    for start_idx, start_time in enumerate(event_times[:-2]):

        max_end_time = start_time + np.timedelta64(1, 'h')   # prepare to find elements within this right bound of time
        max_end_idx = bisect_left(event_times, max_end_time) - 1    # find largest idx within time bound, this will be largest possible window for current start_time

        if max_end_idx < start_idx + 2:
            continue # no need to continue if cannot form at least 3 rows


        if start_idx:
            min_end_time = max_end_time - event_times_diff[start_idx]  

            bisected_idx = bisect_left(event_times, min_end_time)
            # short-circuit prevents IndexError when event_times[bisected_idx] after or 
            if bisected_idx == array_length or event_times[bisected_idx] >  min_end_time:
                bisected_idx -= 1
                while bisected_idx > start_idx+2 and event_times[bisected_idx-1] == event_times[bisected_idx]: 
                    bisected_idx -= 1
            # smallest window begins at 3 rows minimum, or idx of largest time less than min_end_time 
            min_end_idx = max(start_idx + 2,bisected_idx)

        else:
            # no row before start_idx == 0, so no restriction from bisected_idx
            min_end_idx = start_idx + 2

        for window_tail_idx in range(min_end_idx, max_end_idx + 1):
            current_window = order_shop.iloc[start_idx: window_tail_idx+1] #iloc excludes right edge
            concentration_ratio = len(current_window)/current_window['userid'].nunique()

            if concentration_ratio >= 3: 
                order_user.update(dict(zip(current_window['orderid'],current_window['userid'])))            


    if order_user:
        user_counts = {userid:list(order_user.values()).count(userid) for userid in set(order_user.values())}
        max_value = max(user_counts.values())
        users = sorted(user for user,count in user_counts.items() if count == max_value)
        
        return '&'.join(map(str,users))
    else:
        return '0'
#result_enum_window_dedup_diff = orders_sorted.groupby(['shopid']).apply(find_brush_enum_window_dedup_diff)
from bisect import bisect, bisect_left

time = [10,10,20,20,20,30,30]
# bisect finds index of array to insert new value to keep array sorted.
# bisect and bisect_left differences show up when the value to be inserted matches exactly one of the values in the array
# such a difference is magnified if that matched value is duplicated in the array

bisect(time,20)
bisect_left(time,20)

# No difference between bisect and bisect_left when value inserted does not clash
bisect(time,21)
bisect_left(time,21) 
bisect_left([1,1,2,2,3,3,3],3.1) 
