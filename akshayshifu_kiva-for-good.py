import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
from datetime import datetime, timedelta


import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

df_kiva_loans = pd.read_csv("../input/kiva_loans.csv")
df_kiva_loans.head(5)
countries = df_kiva_loans['country'].value_counts()[df_kiva_loans['country'].value_counts()>3400]
list_countries = list(countries.index)
plt.figure(figsize=(13,8))
sns.barplot(y=countries.index, x=countries.values)
plt.title("Number of borrowers per country", fontsize=16)
plt.xlabel("Nb of borrowers", fontsize=16)
plt.ylabel("Countries", fontsize=16)
plt.show()
plt.figure(figsize=(13,8))
sectors = df_kiva_loans['sector'].value_counts()
sns.barplot(y=sectors.index, x=sectors.values)
plt.xlabel('Number of loans', fontsize=16)
plt.ylabel("Sectors", fontsize=16)
plt.title("Number of loans per sector")
plt.show()
temp = df_kiva_loans['loan_amount']

plt.figure(figsize=(12,8))
sns.distplot(temp[~((temp-temp.mean()).abs()>3*temp.std())]);
plt.ylabel("density estimate", fontsize=16)
plt.xlabel('loan amount', fontsize=16)
plt.title("KDE of loan amount (outliers removed)", fontsize=16)
plt.show()
