#Python Code to generate Plot of Active Corona Cases for States of India



#Import Key Libraries

import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns

import matplotlib.pyplot as plt



# Lines below are just to ignore warnings

import warnings

warnings.filterwarnings('ignore')





#URL to import Corona Dataset for INDIA

#url = 'https://api.covid19india.org/csv/latest/state_wise_daily.csv'

# Read file into a DataFrame: df

#dfIND = pd.read_csv(url)



dfIND= pd.read_csv('../input/india-states/state_wise_daily.csv')

#dataset = dfIND[dfIND['Date_reported'] == df['Date_reported'].max()]

#dataset.reset_index(drop = 'index', inplace = True)





#dfIND.head()

#dfIND.columns





Ind = dfIND[dfIND['Status'] == 'Confirmed'].sum()

Ind.drop(labels=['Date','Status','TT'], inplace = True)

#Get Top States - Total Cases



newpd = pd.DataFrame(data = Ind)

newpd['Total Cases']= dfIND[dfIND['Status'] == 'Confirmed'].sum()



#Get Top States

top_ind = newpd.sort_values(by = ['Total Cases'], ascending = False).head(20)

states = top_ind.index.values



fig, axes = plt.subplots(nrows = 5, ncols = 4, figsize = (30,20))



i = 0

j = 0



for x in states:

    temp = dfIND[dfIND['Status'] == 'Confirmed']

    x_data = temp['Date']

    y_data = temp[x]

    #z_data = temp[' New_deaths']

    axes[j,i].plot(x_data,y_data,'g-')

    axes[j,i].set_title(x)

    axes[j,i].set_xticklabels({})

    axes[j,i].tick_params('y', labelsize = 16)

    i = i + 1        

    if i == 4:

        i = 0

        j = j + 1
