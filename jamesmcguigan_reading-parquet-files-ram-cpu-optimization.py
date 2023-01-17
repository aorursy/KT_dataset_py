!free -h
!cat /proc/cpuinfo
import pandas as pd

import pyarrow

import pyarrow.parquet as pq

from pyarrow.parquet import ParquetFile



try:   import parquet

except Exception as exception: print(exception)

    

try:   import fastparquet

except Exception as exception: print(exception)    
import pandas as pd

import numpy as np

import pyarrow

import glob2

import gc

import time

import sys

import humanize

import math

import time

import psutil

import gc

import simplejson

import skimage

import skimage.measure

from timeit import timeit

from time import sleep

from pyarrow.parquet import ParquetFile

import pyarrow

import pyarrow.parquet as pq

import signal

from contextlib import contextmanager



pd.set_option('display.max_columns',   500)

pd.set_option('display.max_colwidth',   -1)
@contextmanager

def timeout(time):

    # Register a function to raise a TimeoutError on the signal.

    signal.signal(signal.SIGALRM, raise_timeout)

    # Schedule the signal to be sent after ``time``.

    signal.alarm(time)



    try:

        yield

    except TimeoutError:

        pass

    finally:

        # Unregister the signal so it won't be triggered

        # if the timeout is not reached.

        signal.signal(signal.SIGALRM, signal.SIG_IGN)



def raise_timeout(signum, frame):

    raise TimeoutError
from memory_profiler import profile
!python --version  # Python 3.6.6 :: Anaconda, Inc == original + latest docker (2020-03-14)
pd.__version__  # 0.25.3 == original + latest docker (2020-03-14)
filenames = sorted(glob2.glob('../input/bengaliai-cv19/train_image_data_*.parquet')); filenames
def read_parquet_via_pandas(files=4, cast='uint8', resize=1):

    gc.collect(); sleep(5);  # wait for gc to complete

    memory_before = psutil.virtual_memory()[3]

    # NOTE: loading all the files into a list variable, then applying pd.concat() into a second variable, uses double the memory

    df = pd.concat([ 

        pd.read_parquet(filename).set_index('image_id', drop=True).astype('uint8')

        for filename in filenames[:files] 

    ])

    memory_end= psutil.virtual_memory()[3]        



    print( "sys.getsizeof total", humanize.naturalsize(sys.getsizeof(df)) )

    print( "memory total",        humanize.naturalsize(memory_end - memory_before), '+system', humanize.naturalsize(memory_before) )        

    return df





gc.collect(); sleep(2);  # wait for gc to complete

print('single file:')

time_start = time.time()

read_parquet_via_pandas(files=1); gc.collect()

print( "time: ", time.time() - time_start )

print('----------')

print('pd.concat() all files:')

time_start = time.time()

read_parquet_via_pandas(files=4); gc.collect()

print( "time: ", time.time() - time_start )

pass
import pyarrow

import pyarrow.parquet as pq

from pyarrow.parquet import ParquetFile



pyarrow.__version__  # 0.16.0 == original + latest docker (2020-03-14)
# DOCS: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html

def read_parquet_via_pyarrow_file():

    pqfiles = [ ParquetFile(filename) for filename in filenames ]

    print( "sys.getsizeof", humanize.naturalsize(sys.getsizeof(pqfiles)) )

    for pqfile in pqfiles[0:1]: print(pqfile.metadata)

    return pqfiles



gc.collect(); sleep(2);  # wait for gc to complete

time_start = time.time()

read_parquet_via_pyarrow_file(); gc.collect()

print( "time: ", time.time() - time_start )

pass
# DOCS: https://arrow.apache.org/docs/python/parquet.html

# DOCS: https://arrow.apache.org/docs/python/generated/pyarrow.Table.html

# NOTE: Attempting to read all tables into memory, causes an out of memory exception

def read_parquet_via_pyarrow_table():

    shapes  = []

    classes = []

    sizes   = 0

    for filename in filenames:

        table = pq.read_table(filename) 

        shapes.append( table.shape )

        classes.append( table.__class__ )

        size = sys.getsizeof(table); sizes += size

        print("sys.getsizeof:",   humanize.naturalsize(sys.getsizeof(table))  )        

    print("sys.getsizeof total:", humanize.naturalsize(sizes) )

    print("classes:", classes)

    print("shapes:",  shapes)    





gc.collect(); sleep(2);  # wait for gc to complete

time_start = time.time()

read_parquet_via_pyarrow_table(); gc.collect()

print( "time: ", time.time() - time_start )

pass
import time, psutil, gc



gc.collect(); sleep(2)  # wait for gc to complete

mem_before   = psutil.virtual_memory()[3]

memory_usage = []



def read_parquet_via_pyarrow_table_generator(batch_size=128):

    for filename in filenames[0:1]:  # only loop over one file for demonstration purposes

        gc.collect(); sleep(1)

        for batch in pq.read_table(filename).to_batches(batch_size):

            mem_current = psutil.virtual_memory()[3]

            memory_usage.append( mem_current - mem_before )

            yield batch.to_pandas()

  



time_start = time.time()

count = 0

for batch in read_parquet_via_pyarrow_table_generator():

    count += 1



print( "time:  ", time.time() - time_start )

print( "count: ", count )

print( "min memory_usage: ", humanize.naturalsize(min(memory_usage))  )

print( "max memory_usage: ", humanize.naturalsize(max(memory_usage))  )

print( "avg memory_usage: ", humanize.naturalsize(np.mean(memory_usage)) )

pass    
memory_before = psutil.virtual_memory()[3]

memory_usage  = []



def read_parquet_via_pandas_generator(batch_size=128, reads_per_file=5):

    for filename in filenames:

        num_rows    = ParquetFile(filename).metadata.num_rows

        cache_size  = math.ceil( num_rows / batch_size / reads_per_file ) * batch_size

        batch_count = math.ceil( cache_size / batch_size )

        for n_read in range(reads_per_file):

            cache = pd.read_parquet(filename).iloc[ cache_size * n_read : cache_size * (n_read+1) ].copy()

            gc.collect(); sleep(1);  # sleep(1) is required to allow measurement of the garbage collector

            for n_batch in range(batch_count):            

                memory_current = psutil.virtual_memory()[3]

                memory_usage.append( memory_current - memory_before )                

                yield cache[ batch_size * n_batch : batch_size * (n_batch+1) ].copy()



                

for reads_per_file in [1,2,3,5]: 

    gc.collect(); sleep(5);  # wait for gc to complete

    memory_before = psutil.virtual_memory()[3]

    memory_usage  = []

    

    time_start = time.time()

    count = 0

    for batch in read_parquet_via_pandas_generator(batch_size=128, reads_per_file=reads_per_file):

        count += 1

        

    print( "reads_per_file", reads_per_file, '|', 

           'time', int(time.time() - time_start),'s', '|', 

           'count', count,  '|',

           'memory', {

                "min": humanize.naturalsize(min(memory_usage)),

                "max": humanize.naturalsize(max(memory_usage)),

                "avg": humanize.naturalsize(np.mean(memory_usage)),

                "+system": humanize.naturalsize(memory_before),               

            }

    )

pass    
def read_single_parquet_via_pandas_with_cast(dtype='uint8', normalize=False, denoise=False, invert=True, resize=1, resize_fn=None):

    gc.collect(); sleep(2);

    

    memory_before = psutil.virtual_memory()[3]

    time_start = time.time()        

    

    train = (pd.read_parquet(filenames[0])

               .set_index('image_id', drop=True)

               .values.astype(dtype)

               .reshape(-1, 137, 236, 1))

    

    if invert:                                         # Colors | 0 = black      | 255 = white

        train = (255-train)                            # invert | 0 = background | 255 = line

   

    if denoise:                                        # Set small pixel values to background 0

        if invert: train *= (train >= 25)              #   0 = background | 255 = line  | np.mean() == 12

        else:      train += (255-train)*(train >= 230) # 255 = background |   0 = line  | np.mean() == 244     

        

    if isinstance(resize, bool) and resize == True:

        resize = 2    # Reduce image size by 2x

    if resize and resize != 1:                  

        # NOTEBOOK: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing/

        # Out of the different resize functions:

        # - np.mean(dtype=uint8) produces produces fragmented images (needs float16 to work properly - but RAM intensive)

        # - np.median() produces the most accurate downsampling

        # - np.max() produces an enhanced image with thicker lines (maybe slightly easier to read)

        # - np.min() produces a  dehanced image with thiner lines (harder to read)

        resize_fn = resize_fn or (np.max if invert else np.min)

        cval      = 0 if invert else 255

        train = skimage.measure.block_reduce(train, (1, resize,resize, 1), cval=cval, func=resize_fn)  # train.shape = (50210, 137, 236, 1)

        

    if normalize:

        train = train / 255.0          # division casts: int -> float64 





    time_end     = time.time()

    memory_after = psutil.virtual_memory()[3] 

    return ( 

        str(round(time_end - time_start,2)).rjust(5),

        # str(sys.getsizeof(train)),

        str(memory_after - memory_before).rjust(5), 

        str(train.shape).ljust(20),

        str(train.dtype).ljust(7),

    )





for dtype in ['uint8', 'uint16', 'uint32', 'float16', 'float32']:  # 'float64' caused OOM error

    seconds, memory, shape, dtype = read_single_parquet_via_pandas_with_cast(dtype=dtype)

    print(f'dtype {dtype}'.ljust(18) + f'| {dtype} | {shape} | {seconds}s | {humanize.naturalsize(memory).rjust(8)}')



for denoise in [False, True]:

    seconds, memory, shape, dtype = read_single_parquet_via_pandas_with_cast(denoise=denoise)

    print(f'denoise {denoise}'.ljust(18) + f'| {dtype} | {shape} | {seconds}s | {humanize.naturalsize(memory).rjust(8)}')



for normalize in [False, True]:

    seconds, memory, shape, dtype = read_single_parquet_via_pandas_with_cast(normalize=normalize)

    print(f'normalize {normalize}'.ljust(18) + f'| {dtype} | {shape} | {seconds}s | {humanize.naturalsize(memory).rjust(8)}')    
# division casts: int -> float64 

for dtype in ['float16', 'float32']:

    seconds, memory, shape, dtype = read_single_parquet_via_pandas_with_cast(dtype=dtype, normalize=True)

    print(f'normalize {dtype}'.ljust(18) + f'| {dtype} | {shape} | {seconds}s | {humanize.naturalsize(memory).rjust(8)}')    
# skimage.measure.block_reduce() casts: unit8 -> float64    

for resize in [2, 3, 4]:

    for dtype in ['float16', 'float32', 'uint8']:  # 'float32' almosts causes OOM error 

        gc.collect()

        with timeout(10*60):

            seconds, memory, shape, dtype = read_single_parquet_via_pandas_with_cast(dtype=dtype, resize=resize)

            print(f'resize {resize} {dtype}'.ljust(18) + f'| {dtype} | {shape} | {seconds:6.2f}s | {humanize.naturalsize(memory).rjust(8)}')