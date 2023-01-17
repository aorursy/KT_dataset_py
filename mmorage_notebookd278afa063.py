

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)





GT=pd.read_csv('../input/GlobalTemperatures.csv',delimiter=',')

#Variables

dt=pd.to_datetime(GT['dt'])

#yr=(dt.apply(lambda x: x[:4]))

#m=(dt.apply(lambda x: x[5:7]))

#d=(dt.apply(lambda x: x[8:10]))

yr=dt.map(lambda x: x.year)

m=dt.map(lambda x: x.month)

d=dt.map(lambda x: x.day)

#everithing in yr 

date=(yr + m /12 +d/365) -1750.096073



LandT=GT['LandAverageTemperature'].fillna(value=9.94,inplace=False) 	

LandT_err=GT['LandAverageTemperatureUncertainty'] 	

LandT_max=GT['LandMaxTemperature'] 	

LandT_max_err=GT['LandMaxTemperatureUncertainty']

LandT_min=GT['LandMinTemperature'] 	

LandT_min_err=GT['LandMinTemperatureUncertainty']

Land_OceanT=GT['LandAndOceanAverageTemperature']

Land_OceanT_err=GT['LandAndOceanAverageTemperatureUncertainty']

date


#plt.scatter(date,Land_OceanT,marker='o')

#plt.axis([1980,2010,-20,20])

#plt.axis([0.0859,0.0861,-20,20])

plt.plot(date,LandT,marker='.',color='red')
#Sinosoidal shape on the data due to the season ???

#fitting a sinusoidal function

from scipy.optimize import curve_fit

import pylab as plt



guess_freq=11

guess_amplitude=5 

guess_phase=10

guess_offset=0#np.mean(yr)



p0=[guess_freq, guess_amplitude,

    guess_phase, guess_offset]



def sinusoidal(x,freq,amplitude,phase,offset):

    #return np.sin(x*freq+phase)*amplitude+ offset

    return np.sin(x*freq+phase)*amplitude+ offset

    

fit=curve_fit(sinusoidal,date,LandT,p0=p0)



data_first_guess = sinusoidal(LandT, *p0)



fit=curve_fit(sinusoidal,date,LandT,p0=p0)



data_fit = sinusoidal(LandT, *fit[0])

#plt.axis([1750,1810,-20,20])

plt.axis([0.0,50,-20,20])

plt.plot(date,LandT,marker='.',color='red')



plt.plot(data_fit)

fit[0]
GT