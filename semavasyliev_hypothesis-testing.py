

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("../input/data.csv", encoding='Windows-1251')
df.head()

import numpy as np
print(list(np.array(df['2010'])))
print(list(np.array(df['2017'])))
import math
def build_variation_series(list_series):
    min_ = min(list_series)
    max_ = max(list_series)
    variation_swing = max_ - min_
    n = len(list_series)
    interval_count = int(1 + 1.322*math.log(n))
    dev_ = variation_swing / interval_count
    step = round(dev_)
    overflow = (step - dev_) * interval_count
    start = min_ - overflow/2
    end = max_ + overflow/2
    interval_values_list = []
    interval_freq_list = []
    intervals_dictionary = {}
    iteration = 0
    val_seria = []
    var_seria = []
    start_0 = start
    while start < end:
        interval_values = 0
        interval = []
        interval.append(start)
        interval_end = start + step
        interval.append(interval_end)
        for value in list_series:
            if value >= start and value < start + step:
                interval_values = interval_values + 1
        interval_values_list.append(interval_values)
        interval_freq_list.append(interval_values/n)
        key = '[' + str(round(start,1)) + ';' + str(round(interval_end,1)) + ']'
        value = str(interval_values) + ' ; ' + str(interval_values/n)
        intervals_dictionary[key] = value
        if iteration > 0:
            for i in range(3):
                val_seria.append(round(start,1))
            for i in range(2):
                var_seria.append(interval_values/n)
            var_seria.append(0)
        else:
            for i in range(2):
                val_seria.append(round(start,1))
            var_seria.append(0)
            
            
        start = interval_end
                
        iteration = iteration + 1
    
    del val_seria[-1]    
    var_seria.append(0)    
    val_seria.append(round(start_0,1))
    return val_seria, var_seria
income_2010 = list(np.array(df['2010']))
income_2017 = list(np.array(df['2017']))
income = income_2010 + income_2017
val_seria, var_seria = build_variation_series(income)
import matplotlib.pyplot as plt
plt.plot(val_seria, var_seria)
plt.xlabel('income')
plt.ylabel('variation')
plt.show()
from scipy import stats
def is_normal(x, alpha = 1e-3):
    k2,p = stats.normaltest(x)
    print('--p-- = {}'.format(p))
    return p > alpha
        
is_normal(np.array(df['2010']))
is_normal(np.array(df['2017']))
def fit_and_plot(dist, data, lower_bound, upper_bound):
    params = dist.fit(data) #return (mean, std) tuple
    arg = params[:-2] #The skewness reduces as the value of alpha increases. (for gamma distribution)
    #gamma is class of continue distributions
    loc = params[0]
    scale = params[1]
    x = np.linspace(0, upper_bound, upper_bound)
    _, ax = plt.subplots(figsize=(30, 10))
    plt.hist(data, bins = 20, range=(lower_bound, upper_bound))
    ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis
    plt.plot(x, dist.pdf(x, loc=loc, scale=scale, *arg), '-', color = "r")
    plt.show()
    return dist, loc, scale
fit_and_plot(stats.norm, list(np.array(df['2017'])), 5000, 14000)
fit_and_plot(stats.norm, list(np.array(df['2010'])), 1000, 5000)
import statistics
def scale(x,y):
    x_transformed = None
    if is_normal(x) == False or is_normal(y) == False:
        x = list(x)
        y = list(y)
        x_transformed = [5 * (1 + (elem-statistics.mean(x)))/statistics.stdev(x) for elem in x]
        y_transformed = [1 + (elem-statistics.mean(y))/statistics.stdev(y) for elem in y]
    else:
        x_transformed = list(x)
        y_transformed = list(y)
        
    return x_transformed, y_transformed 

x_transformed, y_transformed = scale(np.array(df['2010']), np.array(df['2017']))        
x_transformed
y_transformed
from scipy.stats import ttest_ind
t2, p2 = ttest_ind(x_transformed,y_transformed)
print("t = " + str(abs(t2))) #table - 0.851
print("p = " + str(p2))