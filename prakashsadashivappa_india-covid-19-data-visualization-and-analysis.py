#!/usr/bin/env python

# coding: utf-8



# ### Import necessary libraries



# In[1]:





import numpy as np, pandas as pd

from pandas import ExcelWriter

from pandas import ExcelFile

import matplotlib.pyplot as plt

import warnings

import itertools

from matplotlib.pyplot import figure





# ### Import the dataset



# In[2]:





df=pd.read_csv("../input/covid19-in-india/covid_19_india.csv")





# In[3]:





df.columns





# In[4]:





df.info()





# In[5]:





df['Date']=pd.to_datetime(df.Date, format='%d/%m/%y')





# In[6]:





df.index=df['Date']





# In[7]:





df.tail()





# In[9]:





df=df.drop(['Sno','Date','Time'], axis=1)







df.rename(columns={'State/UnionTerritory':'State'},inplace=True)









#### Clean the names of state 'Nagaland#' to 'Nagaland'









df['State'] = df['State'].replace(to_replace={'.*Nagaland#.*': 'Nagaland'}, regex=True)

df['State'] = df['State'].replace(to_replace={'.*Jharkhand#.*': 'Jharkhand'}, regex=True)







df.groupby(['Date'])['Confirmed', 'Cured', 'Deaths'].sum().plot(kind='line', figsize=(15,7), linestyle='solid', 

                                                                linewidth=3, grid=False,color=('blue','darkgreen','r'))

plt.title('Total number of cases in India')

plt.show()





options=df.index[-1]





# In[26]:





options





# In[27]:





India=df[df.index==options]





# In[28]:





India.info()





# In[30]:







import warnings 

warnings.filterwarnings("ignore")









axes=df.groupby('State').plot(figsize=(15,7), linestyle='solid', 

                              color=('darkgreen','r','blue'), linewidth=3, grid=False, sharey=False, sharex=False)

axes[0].set_title('Andaman & Nicobar Islands')

axes[1].set_title('Andhra Pradesh')

axes[2].set_title('Arunachal Pradesh')

axes[3].set_title('Assam')

axes[4].set_title('Bihar')

axes[5].set_title('Chandigarh')

axes[6].set_title('Chhattisgarh')

axes[7].set_title('Delhi')

axes[8].set_title('Goa')

axes[9].set_title('Gujarat')

axes[10].set_title('Haryana')

axes[11].set_title('Himachal Pradesh')

axes[12].set_title('Jammu & Kashmir')

axes[13].set_title('Jharkhand')

axes[14].set_title('Karnataka')

axes[15].set_title('Kerala')

axes[16].set_title('Ladakh')

axes[17].set_title('Madhya Pradesh')

axes[18].set_title('Maharashtra')

axes[19].set_title('Manipur')

axes[20].set_title('Meghalaya')

axes[21].set_title('Mizoram')

axes[22].set_title('Nagaland')

axes[23].set_title('Odisha')

axes[24].set_title('Puducherry')

axes[25].set_title('Punjab')

axes[26].set_title('Rajashthan')

axes[27].set_title('Tamil Nadu')

axes[28].set_title('Telangana')

axes[29].set_title('Tripura')

axes[30].set_title('Unassigned')

axes[31].set_title('Uttar Pradesh')

axes[32].set_title('Uttarakhand')

axes[33].set_title('West Bengal')





# In[ ]:











df.groupby('State')['Confirmed', 'Cured', 'Deaths'].nunique().plot(kind='bar', figsize=(17,9),color=('blue','darkgreen','r'), grid=False)

plt.title("State-wise recent number of cases")

plt.show()
