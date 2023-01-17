#Helpful packages that will be used in this projects:



import numpy as np 

import pandas as pd 

import matplotlib.dates as mdates

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

from datetime import date

from io import StringIO

from pandas.tools.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=15,6

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))