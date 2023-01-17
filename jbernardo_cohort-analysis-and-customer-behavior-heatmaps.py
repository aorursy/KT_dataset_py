import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pylab import rcParams

import warnings

warnings.filterwarnings("ignore")

from IPython.display import Markdown as md

import sys

import requests

import datetime

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Reading excel data

data = pd.read_excel("/kaggle/input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx")

#Dropping NaN values

data.dropna(inplace = True)

data.info()

# checking dataframe

data.sample(5)
# Cohort Data preparation

# Creating new columns

data['CohortGroup'] = data.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')

data["Inv_Month"] = data['InvoiceDate'].dt.month

data["Inv_Year"] = data['InvoiceDate'].dt.year

data["Coh_Month"] = data.groupby('CustomerID')['InvoiceDate'].transform('min').dt.month

data["Coh_Year"] = data.groupby('CustomerID')['InvoiceDate'].transform('min').dt.year

years_diff =  data["Inv_Year"] - data["Coh_Year"]

months_diff = data["Inv_Month"] - data["Coh_Month"] 



# Get periods

data["t"] = years_diff * 12 + months_diff + 1



#With these three steps I can get my Cohort Matrix

cohort = data.groupby(['CohortGroup','t']).CustomerID.nunique().unstack()

cohort_group = data.groupby('CohortGroup').CustomerID.nunique()

cohort_per = cohort.divide(cohort_group, axis = 0)
# Plotting data

# source: https://towardsdatascience.com/a-step-by-step-introduction-to-cohort-analysis-in-python-a2cbbd8460ea

with sns.axes_style("white"):

    fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': [1, 11]})

    

    # retention matrix

    sns.heatmap(cohort_per, 

                mask=cohort_per.isnull(), 

                annot=True, 

                fmt='.0%', 

                cmap='RdYlGn',

                linewidths=.5,

                ax=ax[1])

    ax[1].set_title('Análise Cohort - Grupo Cohort é dado pelo mês da primeira compra', fontsize=16)

    ax[1].set(xlabel='Número de periodos [meses]',

              ylabel='')



    # cohort size

    sns.heatmap(pd.DataFrame(cohort_group), 

                annot=True, 

                cbar=False, 

                fmt='g', 

                cmap= (['white']), 

                ax=ax[0])

    ax[0].set(xlabel='# clientes',

              ylabel='')



    fig.tight_layout()
# Check the steps of pandas data handling for Cohort Analysis

# Here is how the data is on excel with the columns we have created before

data.tail(10)
# Step1 - Groupby and count number of uniques customers that bought something on the Online Platform

pd.DataFrame(data.groupby(['CohortGroup','t']).CustomerID.nunique()).tail(10)
# Step 2 - Now unstack it

data.groupby(['CohortGroup','t']).CustomerID.nunique().unstack()
# Step 3 - divide by the cohot group to get the %

cohort.divide(cohort_group, axis = 0)
#Checking what time over the week the users buy most - 2011 data only

datay = data[data.InvoiceDate.dt.year==2011]

datay["Time"] =  datay.InvoiceDate.dt.hour

datay['Weekday'] = datay.InvoiceDate.dt.day_name() #Check pandas version







# Retructuring data

restr_data = datay.groupby(['Time', 'Weekday']).CustomerID.nunique().unstack()

restr_data= restr_data.fillna(0)

# Reordering columns

restr_data = restr_data[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']] # there is no invoice on 'Saturday'!



# Plotting a heat map

plt.figure(figsize=(12,8))

plt.title("Counting invoices per user over the week - 2011")

sns.heatmap(restr_data,

            mask=restr_data.isnull(), 

            annot=True, 

            fmt="g",

            cmap='RdYlGn',

            linewidths=.5);
#Checking what days over the month the users buy most - 2011 data only

datay = data[data.InvoiceDate.dt.year==2011]

datay["Day"] =  datay.InvoiceDate.dt.day

datay['Month'] = datay.InvoiceDate.dt.month_name()





# Retructuring data

restr_data = datay.groupby(['Day', 'Month']).Quantity.count().unstack()

restr_data= restr_data.fillna(0)

# Reordering columns

restr_data = restr_data[['January', 

                         'February', 

                         'March', 

                         'April', 

                         'May', 

                         'June', 

                         'July',

                         'August', 

                         'September', 

                         'October', 

                         'November',

                         'December'

                        ]]

# Plotting a heat map

plt.figure(figsize=(12,8))

plt.title("Mapa de calor da somatória dos pedidos feitos")

sns.heatmap(restr_data,

            mask=restr_data.isnull(), 

            annot=True, 

            fmt="g",

            cmap='RdYlGn',

            linewidths=.5);