# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import functools
from IPython.display import display, Markdown
import math
import os
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import ScalarFormatter
%matplotlib inline
import numpy as np
import pandas as pd
import dask.dataframe as dd
pd.plotting.register_matplotlib_converters()
import seaborn as sns
import scipy as sci
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import sympy as sym
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from scipy.integrate import odeint
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

my_data= pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
print(my_data)
#for infected population
infected = pd.read_csv("/kaggle/input/covid19-in-turkey/time_series_covid_19_confirmed_tr.csv")
infected
#For death population
deathnumber = pd.read_csv("/kaggle/input/covid19-in-turkey/time_series_covid_19_deaths_tr.csv")
deathnumber
#For recovered population 
recovered = pd.read_csv("/kaggle/input/covid19-in-turkey/time_series_covid_19_recovered_tr.csv")
recovered

data_tests = pd.read_csv('../input/covid19-in-turkey/test_numbers.csv')
data = pd.read_csv('../input/covid19-in-turkey/covid_19_data_tr.csv')
data.rename(columns = {"Country/Region":"TurkeyCovid-19","Last_Update" : "date", "Confirmed" : "Confirmed_people", "Deaths" : "Death_Count", "Recovered" : "Cured_Count"}, inplace = True)
increasing_confirmed = [0]
increasing_death = [0]
increasing_cured = [0]
for i in range(len(data)-1):
    increasing_confirmed.append( data["Confirmed_people"][i+1] - data["Confirmed_people"][i] )
    increasing_death.append( data["Death_Count"][i+1] - data["Death_Count"][i] )
    increasing_cured.append( data["Cured_Count"][i+1] - data["Cured_Count"][i] )
data["increasing_confirmed_counter"] = increasing_confirmed
data["increasing_death_counter"] = increasing_death
data["increasing_cured_counter"] = increasing_cured
date_x     = data.date
confirmed_label     = data.increasing_confirmed_counter
death_label    = data.increasing_death_counter
cured_label = data.increasing_cured_counter
deathcured_label =  data.increasing_death_counter  +  data.increasing_cured_counter
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, dpi=150, figsize=(20, 10),sharex='col')
fig.suptitle('Daily Covid-19 Information from 11 March in Turkey to figure SIR-F Model of Covid-19 in Turkey')
ax1.plot(date_x,confirmed_label,'-o',linewidth = 2, alpha=1)
ax2.plot(date_x,death_label,'-o',linewidth = 2, alpha=1)
ax3.plot(date_x,cured_label,'-o',linewidth = 2, alpha=1)
ax4.plot(date_x,deathcured_label,'-o',linewidth = 2, alpha=1)

ax1.set_xticklabels(date_x, rotation=90)
ax2.set_xticklabels(date_x, rotation=90)
ax3.set_xticklabels(date_x, rotation=90)
ax4.set_xticklabels(date_x, rotation=90)

ax1.grid(color='black', linestyle="--", linewidth=1,alpha=1 ,dash_joinstyle = "bevel")
ax2.grid(color='black', linestyle="--", linewidth=1,alpha=1 ,dash_joinstyle = "bevel")
ax3.grid(color='black', linestyle="--", linewidth=1,alpha=1 ,dash_joinstyle = "bevel")
ax4.grid(color='black', linestyle="--", linewidth=1,alpha=1 ,dash_joinstyle = "bevel")
	
ax1.set_title('Confirmed in Turkey')
ax2.set_title('Death in Turkey')
ax3.set_title('Cured in Turkey')
ax4.set_title('Cured +Death in Turkey')

plt.show()
#at initial
N= 83154997

iI=1 #at sir model I means infected so confirmed=infected
iR=0 #at sir model R= recovered
iS=N-iR-iI #N=S+I+R "i" means initial

(x, y, z)= (iS/N, iI/N, iR/N)

beta= 178239/N  # effective contact rate= confirmed/population
gama= 151417/178239 #recovery rate=recovered/confirmed
R0= beta/gama
T=96
tau=2
(t, rho, sigma) = (T/tau, beta*tau, gama*tau)
t = np.linspace(0, 96, 96)
eg_r0, eg_rho = (R0, rho)
eg_sigma = eg_rho / eg_r0
eg_initials = (x, y, z)
display(Markdown(rf"$\rho = {eg_rho},\ \sigma = {eg_sigma}$."))
def deriv(y, t, N, beta, gama):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gama * I
    dRdt = gama * I
    return dSdt, dIdt, dRdt

initial = iS, iI, iR
# to combine time and derivative equations
bam = odeint( deriv , initial, t, args=(N, beta, gama))
S, I, R = bam.T

# For ploting
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', label='Susceptible')
ax.plot(t, I/N, 'r', label='Infected')
ax.plot(t, R/N, 'g', label='Recovered with immunity')

ax.set_ylim(0,1.2)

ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
#Since the number of infected people in 83 million (178 thousand) is quite low, 
#it is not seen on the chart. 
