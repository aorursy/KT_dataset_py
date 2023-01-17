# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#downloaded from https://github.com/CSSEGISandData/COVID-19

#Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE)

df = pd.read_csv('../input/time_series_covid19_confirmed_global.csv')



#single label for region/country

locs = []

for i in df.index:

	prov = df.iloc[i, 0]

	if isinstance(prov, str):

		loc = df.iloc[i, 1] + '_' +prov

	else:

		loc = df.iloc[i, 1]

	locs.append(loc)

df['Location']=locs #adding this column

df = df.set_index('Location') #reindexing by location

df = df.drop(columns = ['Province/State','Country/Region','Lat','Long']) #drop unnecessary columns

df = df.transpose() #get the dataframe in the right orientation (dates are samples)



df = df.drop(columns=df.columns[df.iloc[-1, :]<1000]) #drop countries with too few cases



# The data for Hubei looks suspicious. Mid-February the Chinese government

# changed their definition of confirmed cases to include clinically but not 

# lab-confirmed, adding previously uncounted cases in one day. This is an artifact.





# first we retrieve the daily cases:

w = df['China_Hubei'].to_numpy()

cases = w[1:] - w[:-1]



oldcases = cases.copy()



# For each of the three datapoints on days 12/2, 13/2, 14/2 we look a the past 

# data since the peak 

yy = cases[15:20]

xx = range(len(yy))

# We predict the next points with an exponential

popt, pcov, = curve_fit(lambda t,a,b: a*np.exp(b*t),  xx,  yy, p0=(1, -1))

perr = np.sqrt(np.diag(pcov))

f, ax = plt.subplots()

plt.scatter(xx, yy)

sns.lineplot(range(len(yy)+3), popt[0]*np.exp(popt[1]*range(len(yy)+3)))

new = popt[0]*np.exp(popt[1]*(range(len(yy), len(yy)+3)))

plt.scatter(range(len(yy), len(yy)+3), new, c = 'r')

cases[20:23] = new



# Now we look at the total number of past and present cases that are clinically

# confirmed

excess = np.sum(oldcases - cases)

sofar = np.sum(cases[:22])



# We correct the past number of cases, assuming a constant ratio lab/clinical

rt = (sofar + excess) / sofar

cases[:22] = cases[:22]*rt



# Visualizing the corrected cases

f, ax = plt.subplots(2, 2, figsize=(15,15))

plt.subplot(2,2,1)

plt.bar(range(1,64), oldcases)

plt.title('Original')

plt.ylim(0, 16000)

plt.subplot(2,2,2)

plt.bar(range(1,64), cases)

plt.title('Smoothed')

plt.ylim(0, 16000)

plt.subplot(2,2,3)

plt.bar(range(1,64), np.cumsum(oldcases))

plt.title('Original')

plt.subplot(2,2,4)

plt.bar(range(1,64), np.cumsum(cases))

plt.title('Smoothed')





df['China_Hubei']= np.cumsum(np.concatenate(([w[0]], cases)))