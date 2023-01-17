# imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

%matplotlib inline
# Read the Data

train = pd.read_csv('../input/house-prices-synthetic-demo/bnp_synth_123.csv')

train = train.drop(['ID'],axis=1)

test = pd.read_csv('../input/house-prices-synthetic-demo/bnp_test.csv')

test = test.drop(['ID'],axis=1)

target = train.target

featureNames = train.columns.values
# Function to convert to hexavigesimal base

def az_to_int(az,nanVal=None):

    if az==az:  #catch NaN

        hv = 0

        for i in range(len(az)):

            hv += (ord(az[i].lower())-ord('a')+1)*26**(len(az)-1-i)

        return hv

    else:

        if nanVal is not None:

            return nanVal

        else:

            return az
# Prepare the data: combine, process, split

test['target'] = -999

all_data = train.append(test)



# convert v22 to hexavigesimal

all_data.v22 = all_data.v22.apply(az_to_int)



for c in all_data.columns.values:

    if all_data[c].dtype=='object':

        all_data[c], tmpItter = all_data[c].factorize()



# replace all NA's with -1

all_data.fillna(-1, inplace=True)



# split the data

train = all_data[all_data['target']>-999]

test = all_data[all_data['target']==-999]

test = test.drop(['target'],axis=1)
plt.rcParams['figure.max_open_warning']=300

nbins=20

for c in  featureNames:

    if train[c].dtype != 'object' and c != 'target':

        if c=='v22':

            hbins = 100

        else:

            hbins = nbins

        fig=plt.figure(figsize=(14,4))

        ax1 = fig.add_subplot(1,2,1) 

        

        dataset1 = train[c][~np.isnan(train[c])]

        dataset2 = train[c][~np.isnan(train[c]) & train.target]

        

        # left plot

        hd = ax1.hist((dataset1, dataset2), bins=hbins, histtype='bar',normed=True,

                        color=["blue", "red"],label=['all','target=1'])

        ax1.set_xlabel('Feature: '+c)

        ax1.set_xlim((-1,max(train[c])))

        

        binwidth = hd[1][1]-hd[1][0]

        midpts = (hd[1][:-1]+hd[1][1:])/2

        cdf_all= np.cumsum(hd[0][0])*binwidth

        cdf_ones = np.cumsum(hd[0][1])*binwidth



        # right plot

        ax2 = fig.add_subplot(1,2,2) 

        ax2.set_ylim((0,1))

        ax2.set_xlim((0,nbins))

        ax2.plot(midpts,cdf_all,color='b')

        ax2.plot(midpts,cdf_ones,color='r')

        ax2.plot(midpts,0.5+10*(cdf_all-cdf_ones),color='k')

        ax2.grid()

        ax2.set_xlim((-1,max(train[c])))

        ax2.set_xlabel('cdfs plus cdf_diff*10+0.5')

        ax2.axhline(0.5,color='gray',linestyle='--')