# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#train = pd.read_csv('../input/train_1.csv').fillna('0')

train = pd.read_csv("../input/train_1.csv")

train.head()
# split away the page data from the time series data

train_pages = pd.DataFrame({ 'Page': train["Page"]})

train_pages.head()

        
# from https://www.kaggle.com/muonneutrino/wikipedia-traffic-data-exploration

# create properties for the important data points

import re

def getLang(page):

    sear = re.search('[a-z][a-z].wikipedia.org',page)

    if sear:

        return sear[0][0:2]

    return 'null'



train_pages['language'] = train_pages.Page.map(getLang)



def getExtra(page):

    i = page.find(".org_")

    if i > 0:

        return page[i+len(".org_"):]

    return "NA"



train_pages['namemeta'] = train_pages.Page.map(getExtra)



def getAccess(page):

    spl = page.split("_")

    if len(spl) >= 1:

        return spl[0]

    return 'null'

train_pages['access'] = train_pages.namemeta.map(getAccess)



def getClient(page):

    spl = page.split("_")

    if len(spl) >= 2:

        return spl[1]

    return 'null'

train_pages['client'] = train_pages.namemeta.map(getClient)

train_pages.head()
#train["mean"] = train.drop("Page", axis=1).mean(axis=0)



train['mean'] = train.drop("Page", axis=1).astype(float).mean(axis=1, skipna=True)

train.head()
for i in train.iterrows():

    print(i[1][1:])

    print(max(i[1]))

    raise Exception

train.head()
# rename our time series data briefly - they are sequential over 500 days it seems?

r = list(range(0,550))



from datetime import datetime

dateRoot = datetime.strptime("2015-07-01", "%Y-%m-%d")

def getDateDiff(dateString):

    dateDiff = datetime.strptime(dateString, "%Y-%m-%d")

    return (dateDiff - dateRoot).days()



timef_train = train.drop("Page", axis=1).drop("mean", axis=1)

timef_train.columns = r

timef_train.head()
import scipy.optimize

import matplotlib.pyplot as plt

from math import exp



def gaussian(x, amp, cen, wid):

    return amp * np.exp(-(x-cen)**2 / wid)





#xdata = #timef_train.columns.astype(float);

xdata = list(range(0,550))

for row in timef_train.iterrows():

    ydata = row[1]

    popt, pcov = scipy.optimize.curve_fit(gaussian, xdata, ydata)

    fig = plt.figure()

    ax = fig.add_subplot(2,1,1)

    ax.plot(xdata, ydata, 'b-', label='data')

    ax.plot(xdata, gaussian(xdata, *popt), 'g--', label='fit-test')

    ax.set_yscale('log')

    plt.xlabel('x')

    plt.ylabel('y')

    ax.legend()

    plt.show()

    raise Exception
