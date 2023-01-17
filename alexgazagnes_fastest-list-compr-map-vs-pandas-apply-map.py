#  import
import os, sys, logging, random, time, math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# visualization
%matplotlib inline
sns.set()
# create a decorator to give us runing time of a function (eq to %timeit magic ipython function)
def timer(funct) : 
    
    def wrapper(*args, **kwargs) : 
        
        t = time.time()
        _ = funct(*args, **kwargs)
        t = round(time.time() - t, 5)
        
        return t
    
    return wrapper


# test function
@timer
def waiter(s) : 
    time.sleep(s)
    
####

ser = range(5)
print([round(waiter(i),2) for i in ser])
# our list comprehension
@timer
def list_compr_1(ser) :
    _ = [i**2 for i in ser]
    
####

ser = pd.Series(np.arange(10**6))
list_compr_1(ser)
# our list comprehension (without lambda)
@timer
def list_compr_2(ser) :
    f = lambda i : i**2
    _ = [f(i) for i in ser]
    
####

ser = pd.Series(np.arange(10**6))
list_compr_2(ser)
# our buit-in map
@timer
def buin_map_1(ser) :
    _ = map(lambda i : i**2, ser)
    
####

ser = pd.Series(np.arange(10**6))
buin_map_1(ser)
# our buit-in map (without lambda)
@timer
def buin_map_2(ser) :
    funct : lambda i : i**2
    _ = map(funct, ser)
    
####

# ser = pd.Series(np.arange(10**6))
# buin_map_2(ser)
# our pandas method apply
@timer
def pd_apply_1(ser) :
    _ = ser.apply(lambda i : i**2)
    
####

ser = pd.Series(np.arange(10**6))
pd_apply_1(ser)
# our pandas method apply (without lambda)
@timer
def pd_apply_2(ser) : 
    f = lambda i : i**2
    _ = ser.apply(f)
    
####

ser = pd.Series(np.arange(10**6))
pd_apply_2(ser)
# our pandas method map
@timer
def pd_map_1(ser) :
    _ = ser.map(lambda i : i**2)
    
####

ser = pd.Series(np.arange(10**6))
pd_map_1(ser)
# our pandas method map (without lambda)
@timer
def pd_map_2(ser) :
    f = lambda i : i**2
    _ = ser.map(f)
    
####

ser = pd.Series(np.arange(10**6))
pd_map_2(ser)
# define our params 
n = 6
funct_list = [list_compr_1, list_compr_2, buin_map_1, pd_apply_1, pd_apply_2, pd_map_1, pd_map_2]
cols_list  = ["list_compr_1", "list_compr_2", "buin_map_1", "pd_apply_1", "pd_apply_2", "pd_map_1", "pd_map_2"]
# compute for each method running time for 1, 10, 100, 10**n iterations 
def fastest_method(n, N=1, funct_list=funct_list, cols_list=cols_list) : 
    
    res = list()
    i_list = np.arange(1,n)
    
    for i in i_list: 
        ser = pd.Series(np.arange(10**i))
        res.append( [pd.Series([funct(ser) for _ in range(N)]).mean() for funct in funct_list])

    res = pd.DataFrame(res, index=i_list, columns=cols_list)
    
    return res

####

fastest_method(n,10)
# try iter from 10**1 to 10**8, just for one experience
res = fastest_method(8,5)
res
# plot it 
def plot_res(res) : 
    
    fig, ax = plt.subplots(1,1, figsize=(20,10))
    res.plot(ax=ax)
    ax.set_xlabel("nb of iter (logspace)")
    ax.set_ylabel("seconds")
    ax.set_title("fastest method")

####

plot_res(res)
# select small nb of iters
_res = res.iloc[:3]
plot_res(_res)