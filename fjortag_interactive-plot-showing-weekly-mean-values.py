#importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import fnmatch
%matplotlib inline
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
#Loading the files
#list of file names
lst_csvs = ['../input/csvs_per_year/csvs_per_year/madrid_' + str(x) + '.csv' for x in range(2001,2019)]
#list of dataframes (years)
all_data = []
for i in range(0,len(lst_csvs)):
    all_data = all_data + [pd.read_csv(lst_csvs[i])]
#list of gases
gases_list = []
for i in range(0, len(all_data)):
    gases_list = gases_list + [all_data[i].columns]
    
gases_list = np.unique(np.concatenate(gases_list))
gases_list = gases_list[0:17]
#This functions add new columns and group data together for each week of the year 
def TransformData(data):
    data['date'] = pd.to_datetime(data['date'])
    data['Month'] = data['date'].apply(lambda x: x.month)
    data['Day'] = data['date'].apply(lambda x: x.day)
    data['Week'] = data['date'].apply(lambda x: x.week)
    data['Year'] = data['date'].apply(lambda x: x.year)
    data = data.groupby(['Week']).mean()
    return data
#The function TransformData is applied to each dataframe (year)
for i in range(0,len(all_data)):
    all_data[i]= TransformData(all_data[i])
#To rename the columns to include the year
for i in range(0,len(all_data)):
    all_data[i].columns = str(2001 + i) + '_' + all_data[i].columns
#Concatenating all dataframes
data_total = pd.concat(all_data,axis=1)
#Displaying the plots
#Double click on the labels and then click to select the years
for i in range(0,len(gases_list)):
    lst = fnmatch.filter(list(data_total.columns), '*' + gases_list[i] )
    data_total[lst].iplot(title = gases_list[i],xTitle='Weeks')
