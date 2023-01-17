%matplotlib inline

import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from tqdm import tqdm

import gc

import seaborn as sbn

import datetime as dt

import pdb

from collections import OrderedDict
# Replace these paths with relevant paths on your local host

data_path = '../../../input/kkboxchurnprediction/data/'

output_path = '../../../output/kkboxchurnprediction/out/'

# these are the columns that will be output to the CSV

cols = ['num_25_std', 'num_25_sum', 'num_25_min', 'num_25_max', 'num_25_mean',

       'num_50_std', 'num_50_sum', 'num_50_min', 'num_50_max', 'num_50_mean',

       'num_75_std', 'num_75_sum', 'num_75_min', 'num_75_max', 'num_75_mean',

       'num_985_std', 'num_985_sum', 'num_985_min', 'num_985_max',

       'num_985_mean', 'num_100_std', 'num_100_sum', 'num_100_min',

       'num_100_max', 'num_100_mean', 'num_unq_std', 'num_unq_sum',

       'num_unq_min', 'num_unq_max', 'num_unq_mean', 'total_secs_std',

       'total_secs_sum', 'total_secs_min', 'total_secs_max', 'total_secs_mean',

       'earliest_date', 'latest_date', 'log_count', 'msno']

num_stat_grps = 7

stat_grp_size = 5



def update_user_batch(old, new):

#     pdb.set_trace()

    

    if old is None:

        old = np.zeros((num_stat_grps*stat_grp_size) + 3)



    # run the stats

    xct = old[-1] + new[-1]

    

    for i in range(0,num_stat_grps):

        # move the index to the next set of stats

        ni = i*stat_grp_size

        oi = i*stat_grp_size 

        xsum = old[oi+1] + new[ni+1]

        xmean = float(xsum)/float(xct)

        xstd = 0.0 # wasn't sure how to do this on an incremental basis... feel free to add this if your math skills are better than mine

        old[oi+0] = xstd

        old[oi+1] = xsum

        old[oi+2] = new[ni+2] if xct == 1 else new[ni+2] if new[ni+2] < old[oi+2] else old[oi+2]

        old[oi+3] = new[ni+3] if xct == 1 else new[ni+3] if new[ni+3] > old[oi+3] else old[oi+3]

        old[oi+4] = xmean

    

    # increment row count

    old[-1] = xct

    

    # set the earliest date if needed

    if not old[-3]:

        old[-3] = new[-3]

    elif new[-3] < old[-3]:

        old[-3] = new[-3]



    # update latest date

    if new[-2] > old[-2]:

        old[-2] = new[-2] # assumes all filed entries are sequential

    

    return old



            

# simple unit test; uncomment to run

# told = [0.0,6,2,2,2.0, 1.0,6,1,3,2.0, 3.0,20,5,10,6.66, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 100000, 100000, 3]

# tnew = [0.0,4,2,2,2.0, 1.0,6,3,3,3.0, 3.0,20,10,10,10.0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 100, 900000, 2]

# told2 = update_user_batch(told, tnew)

# print(told2)

# assert told2[-1] == 5

# assert told2[1] == 10

# assert told2[2] == 2

# assert told2[3] == 2

# assert told2[4] == 2.0

# assert told2[6] == 12

# assert told2[7] == 1

# assert told2[8] == 3

# assert told2[9] == 2.40

# assert told2[11] == 40

# assert told2[12] == 5

# assert told2[13] == 10

# assert told2[14] == 8.0



# assert told2[-3] == 100

# assert told2[-2] == 900000

# process the logs

print('getting iterator for user_logs...')



# dictionary for running MSNO stats

userlogs = {}



# drop the second file if you don't want the latest logs

files = ['user_logs.csv', 'user_logs_v2.csv']



start = dt.datetime.now()

for f in files:

    print('READING {0}'.format(f))

    log_reader = pd.read_csv(data_path + f, chunksize=1000000)

    for idx, df in enumerate(log_reader):

        print('chunk {0}; num users: {1}; total duration (min): {2:0.1f}'.format(idx, len(userlogs), (dt.datetime.now() - start).total_seconds()/60.0))

        dfg = df.groupby('msno').agg(OrderedDict([

            ('num_25', {'sum', 'mean', 'std', 'min', 'max'}),

            ('num_50', {'sum', 'mean', 'std', 'min', 'max'}),

            ('num_75', {'sum', 'mean', 'std', 'min', 'max'}),

            ('num_985', {'sum', 'mean', 'std', 'min', 'max'}),

            ('num_100', {'sum', 'mean', 'std', 'min', 'max'}),

            ('num_unq', {'sum', 'mean', 'std', 'min', 'max'}),

            ('total_secs', {'sum', 'mean', 'std', 'min', 'max'}),

            ('date', {'first', 'last', 'count'}),

        ]))

        dfg.columns = ['_'.join(col) for col in dfg.columns.ravel()]        

        dfg.reset_index(inplace=True)

        for row in dfg.iterrows():

            newvals = row[1]

            msno = newvals['msno']

            if msno in userlogs:

                userlogs[msno] = update_user_batch(userlogs[msno], newvals[1::]) # col 0 is msno

            else:

                userlogs[msno] = update_user_batch(None, newvals[1::]) # col 0 is msno
del df, dfg; gc.collect()
dfo = pd.DataFrame(data=[np.append(userlogs[key], key) for key in userlogs.keys()], columns=cols)
del userlogs; gc.collect();
for i in range(0,(35-4)):

    dfo[dfo.columns[i]] = dfo[dfo.columns[i]].astype('float')



dfo.dtypes
# output results

dfo.to_csv(data_path + 'user_logs_summary.csv', index=False)