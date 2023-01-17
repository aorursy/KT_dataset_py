import pandas as pd

import os

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
morta=pd.read_excel('/kaggle/input/mortality-rates-in-the-usa-from-1980-2014/MortalityRateStates.xlsm')
morta.head()
morta=morta[['Location', 'FIPS','Category','Mortality Rate, 1980*','Mortality Rate, 2014*','% Change in Mortality Rate, 1980-2014']]
mortaStates=morta[morta['FIPS']<60]
mortaStates.head()
unitedStates=mortaStates[mortaStates['FIPS']==0]
causes = pd.pivot_table(mortaStates, index = 'Category',columns='Location', values = '% Change in Mortality Rate, 1980-2014')
causes.head()
causes.plot(figsize = (20,10))
causes.plot(kind='bar', figsize=(20,15))
states = pd.pivot_table(mortaStates, index = 'Location', values = '% Change in Mortality Rate, 1980-2014')
states.plot(kind='bar', figsize=(20,15))
categories = pd.pivot_table(mortaStates, index = 'Category', values = '% Change in Mortality Rate, 1980-2014')
categories.plot(kind='bar', figsize=(20,15))
unitedStates.plot(kind='bar', figsize=(20,15))
CausesinUS = pd.pivot_table(mortaStates, index = 'Location', values = '% Change in Mortality Rate, 1980-2014')
CausesinUS.plot(kind='bar', figsize=(20,10))
unitedStates2 = pd.pivot_table(unitedStates, index = 'Category', values = '% Change in Mortality Rate, 1980-2014')
unitedStates2.plot(kind='bar', figsize=(20,10))