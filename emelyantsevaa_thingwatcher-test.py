import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import time

import requests

import json

import math
server_ip_address_port = '192.168.152.127:9080'

thing_name = 'ThingWatcherTest'

twx_app_key = '...'
def update_property(property_name_, value):

    

    requests.put('http://' + server_ip_address_port + '/Thingworx/Things/' + thing_name + \

                '/Properties/' + property_name_, \

                 headers={'Accept': 'application/json', 'appKey': twx_app_key, \

                          'Content-Type': 'application/json'}, \

                data = json.dumps({property_name_: value}))   
#update_property('prop_index_02', 135)
x1 = np.array([math.sin(2*math.pi/16 * i) for i in range(1000) ])

x2 = np.sign(x1) + np.random.random_sample((len(x1),))/100
plt.plot(x1[:32], color='b', label='sin')

plt.plot(x2[:32], color='pink', label='rect')

plt.legend(loc='upper right')

plt.show()
#Uncomment to send requests



'''

i = 0

while True:

    update_property('prop_01', x1[i % 1000])

    update_property('prop_index_01', i % 1000)

    i +=1

    time.sleep(1.0)

'''
'''

i = 0

while True:

    update_property('prop_01', x2[i % 1000])

    update_property('prop_index_01', i % 1000)

    i +=1

    time.sleep(1.0)

'''
# Статистические тесты

from scipy.stats import shapiro, ks_2samp
mu, sigma = 0, 1.0 # mean and standard deviation

x3 = np.random.normal(mu, sigma, 1000)
x3_sample = np.random.normal(mu, sigma, 30)
print(shapiro(x3_sample))
fig, ax = plt.subplots(1,2)

fig.set_size_inches((15,4))



ax[0].hist(x3_sample)

ax[1] = sns.distplot(x3_sample, color='g')
pval = []

for i in range(1000):

    x3_sample = np.random.normal(mu, sigma, 30)

    

    test_results = shapiro(x3_sample)

    pval.append(test_results[1])
np.mean(np.array(pval) < 0.05)
x4 = np.random.chisquare(3, 1000)
x4_sample = np.random.chisquare(3, 30)
print(shapiro(x4_sample))
fig, ax = plt.subplots(1,2)

fig.set_size_inches((15,4))



sns.distplot(x3_sample, color='b', ax=ax[(0)])

sns.distplot(x4_sample, color='g', ax=ax[(1)])

plt.show()
fig, ax = plt.subplots(1,2)

fig.set_size_inches((15,4))



ax[0].hist(x3)

ax[0].hist(x4, color='red', alpha=0.5)



ax[1] = sns.distplot(x3)

ax[1] = sns.distplot(x4)
pval_4 = []

for i in range(1000):

    x4_sample = np.random.chisquare(3, 30)    

    test_results = shapiro(x4_sample)

    pval_4.append(test_results[1])
np.mean(np.array(pval_4) < 0.05)
np.count_nonzero(np.array(pval_4) > 0.05)
sample_size_ = 30

pval_ks_ = []



for i in range(1000):

    x3_sample_ = np.random.normal(mu, sigma, sample_size_) 

    x4_sample_ = np.random.chisquare(3, sample_size_)    

    test_results_ = ks_2samp(x3_sample_, x4_sample_)

    pval_ks_.append(test_results_[1])
np.mean(np.array(pval_ks_) < 0.05)
np.mean(np.array(pval_ks_))
sample_size_ = 30

pval_ks_ = []



for i in range(1000):

    x3_sample_ = np.random.normal(mu, sigma, sample_size_) 

    x4_sample_ = np.random.normal(mu, sigma, sample_size_)    

    test_results_ = ks_2samp(x3_sample_, x4_sample_)

    pval_ks_.append(test_results_[1])
np.mean(np.array(pval_ks_) < 0.05)
np.mean(np.array(pval_ks_))
'''

while True:

    

    x3 = np.random.normal(mu, sigma, 1000) 

    

    for i in range(len(x3)):

    

        update_property('prop_02', x3[i % 1000])

        update_property('prop_index_02', i % 1000)

        i +=1

        time.sleep(1.0)

'''
'''

while True:

    

    x4 = np.random.chisquare(3, 1000)

    

    for i in range(len(x4)):    

    

        update_property('prop_02', x4[i % 1000])

        update_property('prop_index_02', i % 1000)

        i +=1

        time.sleep(1.0)

'''