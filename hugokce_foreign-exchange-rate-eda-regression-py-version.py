# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import datetime

import openpyxl

from pathlib import Path





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print('Foreign Exchange Rate EDA Work is beginning')
import pandas as pd

ferdfc = pd.read_csv("/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv")

ferdfx = pd.read_excel (r'/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.xlsx')

ferdfc.head()
ferdfc.columns
ferdfc.shape
ferdfc.info()
ferdfc.describe()
ferdfc.columns[ferdfc.isnull().any()]
ferdfc['Time Serie'].value_counts()
ferdfc['Time Serie'].value_counts()
ferdfx.head()
ferdfx.info()
ferdfx.info()
ferdfx.describe()
ferdfx.columns[ferdfx.isnull().any()]
ferdfx['Time Serie'].value_counts()
print('You\'ve reached end of analysis. In a short time it will be developed.')