## Importe as bilbiotecas##

import itertools

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import matplotlib.ticker as ticker

from sklearn import preprocessing

import os

import seaborn as sns # visualization

from scipy import stats

from scipy.stats import norm 

import warnings 

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore') #ignore warnings



%matplotlib inline

import gc

import statsmodels.api as sm

from statsmodels.formula.api import ols

  
cdl = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

cdo = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

cdd = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

ttc = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

ttd = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

ttr = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
cdl.head()
#Avalaindo a quantidade de casos por pais

cdl_filt = ['reporting date','country', 'gender', 'age', 'symptom_onset']

cdl_1 = cdl.filter(items=cdl_filt)

cdl_1.head()
plt.style.use('ggplot')



fig=plt.figure(figsize=(30,7))

ax=fig.add_subplot(111)

plt.xlabel('counttry')

plt.ylabel('age')

ax.scatter(cdl_1['country'], cdl_1['age'], alpha=0.5)
cdl.info()
cdl.describe()
cdo.head()
cdo.describe()
cdd.head()
cdd.describe()
ttc.describe()