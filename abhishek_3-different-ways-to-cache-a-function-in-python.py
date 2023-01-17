import functools

import joblib

import time



import numpy as np
!mkdir /kaggle/working/joblib_cache/
memory = joblib.Memory("/kaggle/working/joblib_cache/", verbose=0)
DICTIONARY_CACHE = {}
def long_running_function(number):

    time.sleep(3)

    if number >= 0:

        return np.sqrt(number)

    else:

        return number
def dict_cache_predict(number):

    if number in DICTIONARY_CACHE:

        return DICTIONARY_CACHE[number]

    else:

        result = long_running_function(number)

        DICTIONARY_CACHE[number] = result

        return result
@functools.lru_cache(maxsize=128)

def long_running_function_functools(number):

    time.sleep(3)

    if number >= 0:

        return(np.sqrt(number))

    else:

        return number
@memory.cache

def long_running_function_joblib(number):

    time.sleep(3)

    if number >= 0:

        return(np.sqrt(number))

    else:

        return number
list_of_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

list_of_numbers = list_of_numbers * 10000
# This is the original function. Its going to take atleast 30*10000 seconds

#%time

#for num in list_of_numbers:

#    _ = long_running_function(num)
%time

for num in list_of_numbers:

    _ = dict_cache_predict(num)
%time

for num in list_of_numbers:

    _ = long_running_function_functools(num)
%time

for num in list_of_numbers:

    _ = long_running_function_joblib(num)
DICTIONARY_CACHE