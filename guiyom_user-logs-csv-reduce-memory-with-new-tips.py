# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import time # code performance benchmark

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
# variables used in all parts

userfile = '../input/user_logs.csv'

chuncksize = 2*10**6 #a chuncksize of 2M rows as a starting point

chuncknumbers_max = 20 # we will not read all the file, only 20 chuncks, enough for the demonstration

chunck_number = 0

user_df = pd.DataFrame()

t = time()

for df in pd.read_csv(userfile, chunksize=chuncksize, iterator=True, header=0):

    user_df = user_df.append(df, ignore_index=True)

    chunck_number += 1

    if chunck_number == chuncknumbers_max :

        break

INITIAL_TIME = int(time()-t)

print('done in '+ str(INITIAL_TIME)+'s')



print('memory usage (MB) : ')

INITIAL_MEM = int(user_df.memory_usage(deep=True).sum()/1024**2)

print(INITIAL_MEM)

print('dataframe details')

print(user_df.info(memory_usage='deep'))
chunck_number = 0

user_df = None

list_of_df = []

t = time()

for df in pd.read_csv(userfile, chunksize=chuncksize, iterator=True, header=0):

    # this is a list().append function which is called here, not a Dataframe.append function

    list_of_df.append(df)

    chunck_number += 1

    if chunck_number == chuncknumbers_max :

        break

user_df = pd.concat(list_of_df, ignore_index=True)

# we don't need this list anymore so we suppress it (since it has almost the same size as the obtained dataframe )

del list_of_df

current_time = int(time()-t)

print('done in '+ str(current_time)+'s')

print('performance increase : '+ str(int(100*(1-current_time/INITIAL_TIME))) + '%')

print('memory usage (MB) : ')

current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)

print(current_mem)

# specify dtype associated with each columns of the csv, string dtype correspond also to object dtype

dtype_cols = {'msno': object, 'date':np.int64, 'num_25': np.int32, 'num_50': np.int32, 

             'num_75': np.int32, 'num_985': np.int32, 'num_100': np.int32, 

              'num_unq': np.int32, 'total_secs': np.float32}

user_df = None

chunck_number = 0

list_of_df = []

t = time()

for df in pd.read_csv(userfile, chunksize=chuncksize, iterator=True, header=0, dtype=dtype_cols):

    list_of_df.append(df)

    chunck_number += 1

    if chunck_number == chuncknumbers_max :

        break

user_df = pd.concat(list_of_df, ignore_index=True)

print('done in '+ str(int(time()-t))+'s')

print('memory usage (MB) : ')

current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)

print(current_mem)

gain = int(100*(1-current_mem/INITIAL_MEM))

print('gain :' + str(gain) + '%')
print('memory usage (MB) : ')

user_df.memory_usage(deep=True)/1024**2
print('different msno numbers :')

print(len(user_df.msno.unique()))

print('ratio of unique msno :')

print(str(100*len(user_df.msno.unique())/user_df.shape[0])+'%')
user_df['msno'] = user_df['msno'].astype('category')

print(user_df.info(memory_usage='deep'))

current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)

print(current_mem)

gain = int(100*(1-current_mem/INITIAL_MEM))

print('gain :' + str(gain) + '%')
from datetime import datetime as dt

STARTDATE = dt(2015, 1, 1)

def intdate_as_days(intdate):

    return (dt.strptime(str(intdate), '%Y%m%d') - STARTDATE).days
# remark you need to use pandas > 0.19.1 to be able to use category dtype here 

dtype_cols = {'msno': 'category', 'date':np.int64, 'num_25': np.int32, 'num_50': np.int32, 

             'num_75': np.int32, 'num_985': np.int32, 'num_100': np.int32, 

              'num_unq': np.int32, 'total_secs': np.float32}

user_df = None

chunck_number = 0

list_of_df = []

t = time()

for df in pd.read_csv(userfile, chunksize=chuncksize, iterator=True, header=0, dtype=dtype_cols):

    df['date'] = df['date'].map(lambda x:intdate_as_days(x))

    df['date'] = df['date'].astype(np.int16)

    list_of_df.append(df)

    chunck_number += 1

    if chunck_number == chuncknumbers_max :

        break

user_df = pd.concat(list_of_df, ignore_index=True)

# if you use pandas<0.19, uncomment next line

# user_df['msno'] = user_df['msno'].astype('category')

print('done in '+ str(int(time()-t))+'s')

print('memory usage (MB) : ')

current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)

print(current_mem)

gain = int(100*(1-current_mem/INITIAL_MEM))

print('gain :' + str(gain) + '%')
print(user_df.info(memory_usage='deep'))
dtype_cols = {'msno': object, 'date':np.int64, 'num_25': np.int32, 'num_50': np.int32, 

             'num_75': np.int32, 'num_985': np.int32, 'num_100': np.int32, 

              'num_unq': np.int32, 'total_secs': np.float32}

user_df = None



# loading train.csv into another dataframe

train_df = pd.read_csv('../input/train.csv', dtype={'msno': object, 'is_churn': np.int8})



# we compute only unique values of msno, just in case....

cols_msno = train_df['msno'].unique()



chunck_number = 0

list_of_df = []

t = time()

for df in pd.read_csv(userfile, chunksize=chuncksize, iterator=True, header=0, dtype=dtype_cols):

    # addition to previous script, we will look only to dataframe's msno which are present in train_df

    # only save msno which are already in train_df, 

    append_cond = df['msno'].isin(cols_msno)

    df = df[append_cond]

    

    # as previously...

    df['date'] = df['date'].map(lambda x:intdate_as_days(x))

    df['date'] = df['date'].astype(np.int16)    

    list_of_df.append(df)

    chunck_number += 1

    if chunck_number == chuncknumbers_max :

        break

user_df = pd.concat(list_of_df, ignore_index=True)

user_df['msno'] = user_df['msno'].astype('category')

print('done in '+ str(int(time()-t))+'s')

current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)

print('memory usage (MB) : ' + str(current_mem))

train_df = pd.read_csv('../input/train.csv', dtype={'msno': object, 'is_churn': np.int8})
print('Memory associated with train_df (MB): ')

TRAIN_INIT_MEM = int(train_df.memory_usage(deep=True).sum()/1024**2)

print(TRAIN_INIT_MEM)
train_df['msno'] = train_df['msno'].astype('category')

print('Memory associated with train_df (MB): ')

print(int(train_df.memory_usage(deep=True).sum()/1024**2))
print('different msno numbers in train :')

print(len(train_df.msno.unique()))

print('ratio of unique msno in train:')

print(str(100*len(train_df.msno.unique())/train_df.shape[0])+'%')
# generate the hash dict

hashkey = {}

index = 0

msno_list = train_df['msno'].values

for msno_idx in range(0, len(msno_list)):

    msno = msno_list[msno_idx]

    hashkey.update({msno : '{:09x}'.format(msno_idx)})

# this dict can be saved to a csv file to use it after...

csv_key_file = 'hashkey.csv'

with open(csv_key_file, 'w') as f:

    f.write('msno,hexid\n')

    for k,v in hashkey.items():

        f.write('{0},{1}\n'.format(k,v))

        

# if you want to get  back msno from dict, generate the 'inverse' dict this way

hashkey_reverse = {}

for k,v in hashkey.items(): hashkey_reverse.update({v:k})



# apply this hash to train_df

train_df['msno'] = train_df['msno'].map(lambda x:hashkey.get(x,x))

train_df['msno'] = train_df['msno'].astype('str')

print('Memory associated with train_df (MB): ')

current_mem = int(train_df.memory_usage(deep=True).sum()/1024**2)

print(current_mem)

print('Reduction of (%)')

print(100*(1-current_mem/TRAIN_INIT_MEM))
user_df['msno'] = user_df['msno'].map(lambda x:hashkey.get(x,x))

user_df['msno'] = user_df['msno'].astype('category')

#user_df['msno'] = user_df['msno'].astype('category')

print('Memory associated with final version of user_df (MB): ')

current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)

print('Reduction of (%)')

print(100*(1-current_mem/INITIAL_MEM))