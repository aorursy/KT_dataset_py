import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
import gc
import subprocess
jigsaw_seqlen128_df1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train-processed-seqlen128.csv")
jigsaw_seqlen128_df1.head()
del jigsaw_seqlen128_df1
gc.collect()
jigsaw_seqlen128_df = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train-processed-seqlen128.csv')
jigsaw_seqlen128_df.info(memory_usage='deep')
for dtype in ['float','int','object']:
    selected_dtype = jigsaw_seqlen128_df.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
int_types = ["uint8", "int8", "int16"]
for it in int_types:
    print(np.iinfo(it))
# We will be calculating memory usage a lot,so will create a function save some resource.
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
gl_int = jigsaw_seqlen128_df.select_dtypes(include=['int'])
converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')
print(mem_usage(gl_int))
print(mem_usage(converted_int))
compare_ints = pd.concat([gl_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['Before','After']
compare_ints.apply(pd.Series.value_counts)
gl_float = jigsaw_seqlen128_df.select_dtypes(include=['float'])
converted_float = gl_float.apply(pd.to_numeric,downcast='float')
print(mem_usage(gl_float))
print(mem_usage(converted_float))
compare_floats = pd.concat([gl_float.dtypes,converted_float.dtypes],axis=1)
compare_floats.columns = ['Before','After']
compare_floats.apply(pd.Series.value_counts)
optimized_jigsaw_seqlen128_df = jigsaw_seqlen128_df.copy()
optimized_jigsaw_seqlen128_df[converted_int.columns] = converted_int
optimized_jigsaw_seqlen128_df[converted_float.columns] = converted_float
print(mem_usage(jigsaw_seqlen128_df))
print(mem_usage(optimized_jigsaw_seqlen128_df))
from sys import getsizeof
s1 = 'In progress'
s2 = 'memory consumption'
s3 = 'Objects in python is fun!'
s4 = 'Numerics consume less memory!'
for s in [s1, s2, s3, s4]:
    print(getsizeof(s))
obj_ser = pd.Series(['In progress',
    'memory consumption',
    'Objects in python is fun!',
    'Numerics consume less memory!'])
obj_ser.apply(getsizeof)
jigsaw_seqlen128_df_obj = jigsaw_seqlen128_df.select_dtypes(include=['object']).copy()
jigsaw_seqlen128_df_obj.describe()
rating = jigsaw_seqlen128_df_obj.rating
print(rating.head())
rating_cat = rating.astype('category')
print(rating_cat.head())
rating_cat.head().cat.codes
print(mem_usage(rating))
print(mem_usage(rating_cat))
jigsaw_seqlen128_df_converted_obj = pd.DataFrame()
for col in jigsaw_seqlen128_df_obj.columns:
    num_unique_values = len(jigsaw_seqlen128_df_obj[col].unique())
    num_total_values = len(jigsaw_seqlen128_df_obj[col])
    if num_unique_values / num_total_values < 0.5:
        jigsaw_seqlen128_df_converted_obj.loc[:,col] = jigsaw_seqlen128_df_obj[col].astype('category')
    else:
        jigsaw_seqlen128_df_converted_obj.loc[:,col] = jigsaw_seqlen128_df_obj[col]
print(mem_usage(jigsaw_seqlen128_df_obj))
print(mem_usage(jigsaw_seqlen128_df_converted_obj))
compare_obj = pd.concat([jigsaw_seqlen128_df_obj.dtypes,jigsaw_seqlen128_df_converted_obj.dtypes],axis=1)
compare_obj.columns = ['Before','After']
compare_obj.apply(pd.Series.value_counts)
optimized_jigsaw_seqlen128_df[jigsaw_seqlen128_df_converted_obj.columns] = jigsaw_seqlen128_df_converted_obj
mem_usage(optimized_jigsaw_seqlen128_df)
dtypes = optimized_jigsaw_seqlen128_df.dtypes
dtypes_col = dtypes.index
dtypes_type = [i.name for i in dtypes.values]
column_types = dict(zip(dtypes_col, dtypes_type))
# rather than print all 161 items, we'll
# sample 10 key/value pairs from the dict
# and print it nicely using prettyprint
preview = first2pairs = {key:value for key,value in list(column_types.items())[:10]}
print(preview)
read_and_optimized = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train-processed-seqlen128.csv',dtype=column_types)
print(mem_usage(read_and_optimized))
read_and_optimized.head()
train_df = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", nrows=20000)
train_df.head()
def csv_file_length(fname):
    process = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, error = process.communicate()
    if process.returncode != 0:
        raise IOError(error)
    return int(result.strip().split()[0])

random_rows_selection = csv_file_length('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
print('Number of random rows in "jigsaw-toxic-comment-train.csv" is:', random_rows_selection)
skip_rows = np.random.choice(np.arange(1, random_rows_selection), size=random_rows_selection-1-10000, replace=False)
skip_rows=np.sort(skip_rows)
print('Rows to skip:', len(skip_rows))
print('Remaining rows in the random sample:', random_rows_selection-len(skip_rows))

train = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv', skiprows=skip_rows)
train.head()
del skip_rows
gc.collect()
train_df1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv",skiprows=2000 ,header=None,nrows=20000)
train_df1.head()
train_df2 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv",skiprows=range(1,2000) ,nrows=20000)
train_df2.head()
def number_generator():
    n = 0
    while n < 10:
        yield n
        n += 1
numbers = number_generator()
type(numbers)
next_no = number_generator()
next(next_no)

next(next_no)
import memory_profiler
import time
def even_numbers(numbers):
    even = []
    for num in numbers:
        if num % 2 == 0: 
            even.append(num*num)
            
    return even
if __name__ == '__main__':
    m1 = memory_profiler.memory_usage()
    t1 = time.clock()
    cubes = even_numbers(range(500000))
    t2 = time.clock()
    m2 = memory_profiler.memory_usage()
    time_diff = t2 - t1
    mem_diff = m2[0] - m1[0]
    print(f"It took {time_diff} Secs and {mem_diff} Mb memory to execute Normal method")
import memory_profiler
import time
def even_numbers(numbers):
    for num in numbers:
        if num % 2 == 0:
            yield num * num 
    
if __name__ == '__main__':
    m1 = memory_profiler.memory_usage()
    t1 = time.clock()
    cubes = even_numbers(range(500000))
    t2 = time.clock()
    m2 = memory_profiler.memory_usage()
    time_diff = t2 - t1
    mem_diff = m2[0] - m1[0]
    print(f"It took {time_diff} Secs and {mem_diff} Mb memory to execute Generator method")
import operator
from itertools import product, chain
def accumulate(iterable, func=operator.add, *, initial=None):
    'Return running totals'
    # Example 1: accumulate([4,5,6,7,8,9]) --> 4 9 15 22 30 39
    # Example 2: accumulate([6,7,8,9], initial=1000) --> 1000 1006 1013 1021 
    # Example 3: accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = initial
    if initial is None:
        try:
            total = next(it)
        except StopIteration:
            return
    yield total
    for element in it:
        total = func(total, element)
        yield total
data = [3, 4, 6, 2, 1, 9, 0, 7, 5, 8]
list(accumulate(data, operator.mul))     # running multiplication
list(accumulate(data, max))              # running maximum
mymsg = "Micheal"
msg="My name is "+mymsg+". I live in US"
print(msg)
mymsg = "Micheal"
msg="My name is %s . I live in US"% mymsg
print(msg)
import numpy as np
from random import random
def pi_calculation(n=50000000) -> "area":
    """Estimate pi with monte carlo simulation.
    
    Arguments:
        n: Number of Simulations
    """
    return np.sum(np.random.random(n)**2 + np.random.random(n)**2 <= 1) / n * 4
import pandas as pd

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

no_nans = data.dropna()
one_hot_encoded = pd.get_dummies(no_nans)
# some other temp variables
#processed_data = ...

del no_nans
del one_hot_encoded
import pandas as pd

def data_preprocessing(raw_data):
    no_nans = data.dropna()
    one_hot_encoded = pd.get_dummies(no_nans)
    # some other temp variables
    processed_data = 10
    return processed_data

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
processed_data = data_preprocessing(data)
!pip install line_profiler
%load_ext line_profiler
from random import random

def estimate_pi(n=1e7) -> "area":
    """Estimate pi with monte carlo simulation.
    
    Arguments:
        n: number of simulations
    """
    in_circle = 0
    total = n
    
    while n != 0:
        prec_x = random()
        prec_y = random()
        if pow(prec_x, 2) + pow(prec_y, 2) <= 1:
            in_circle += 1 # inside the circle
        n -= 1
        
    return 4 * in_circle / total
%lprun -f estimate_pi estimate_pi()
!pip install memory_profiler
%load_ext memory_profiler
%%writefile memscript.py
def myfunc():
    a = [1] * 5000000
    b = [2] * 7000000
    del b
    return a
from memscript import myfunc
%mprun -T mprof0 -f myfunc myfunc()
print(open('mprof0', 'r').read())
# Reload modules before executing user code
%load_ext autoreload
%autoreload 2

# setup backend for matplotlibs plots
%matplotlib inline