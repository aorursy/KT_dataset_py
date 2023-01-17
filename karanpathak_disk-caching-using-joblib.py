!pip install joblib
from joblib import Memory
import numpy as np

from time import sleep 
import warnings

warnings.filterwarnings('ignore')
pwd = "/kaggle/working/"

cache_dir = pwd + 'cache_dir'

mem = Memory(cache_dir)
!ls -ld $pwd*/
input1 = np.vander(np.arange(10**4)).astype(np.float)

input2 = np.vander(np.random.uniform(low=0,high=10**5, size=5000))

print("Shape of input1: ",input1.shape)

print("Shape of input2: ",input2.shape)
def func(x):

    print("Example of Computationally intensive function!")

    print("The result is not cached for this particular input")

    sleep(4.0)

    return np.square(x)
func_mem = mem.cache(func, verbose=0)
!du -sh $cache_dir
%%time

input1_result = func_mem(input1)
%%time

input1_cache_result = func_mem(input1)
%%time

input2_result = func_mem(input2)
%%time

input2_cache_result = func_mem(input2)
!du -sh $cache_dir
@mem.cache(verbose=0)

def func_as_decorator(x):

    print("Example of Computationally intensive function!")

    print("The result is not cached for this particular input")

    sleep(4.0)

    return np.square(x)
%%time

input1_decorator_result = func_as_decorator(input1)
%%time

input1_decorator_result = func_as_decorator(input1)
cache_dir2 = pwd + 'cache_dir2'

memory2 = Memory(cache_dir2, mmap_mode='c')
@memory2.cache(verbose=0)

def func_memmap(x):

    print("Example of Computationally intensive function!")

    print("The result is not cached for this particular input")

    sleep(4.0)

    return np.square(x)
%%time

input1_memmap = func_memmap(x=input1)
%%time

input1_memmap = func_memmap(x=input1)
# Disk utilization before clearning function cache

!du -sh $cache_dir
func_mem.clear()

func_as_decorator.clear()
# Disk utilization after clearning function cache

!du -sh $cache_dir
mem.clear()
!du -sh $cache_dir