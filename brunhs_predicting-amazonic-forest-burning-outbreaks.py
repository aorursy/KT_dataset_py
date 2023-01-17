#For our Image

from IPython.display import Image

from IPython.core.display import HTML 



Image(url= "https://media1.s-nbcnews.com/j/newscms/2019_35/2985041/190826-amazon-brazil-fire-cs-834a_eef05addd0d4dc77d766710fa90413d8.fit-760w.jpg", width=950, height=900)
# Loading our packages

import pandas as pd

import matplotlib.pyplot as plt #For our beautiful plots!

import seaborn as sns

import matplotlib.dates as mdates

import matplotlib.ticker as ticker

import warnings

import numpy as np
# Gathering our data

fires_mo = pd.read_excel('../input/inpe-queimadas/fires_mo.xlsx')
fires_mo
#Treating the data



#

#Getting from 1999 to the front

fires_mo_c = fires_mo.iloc[1:,:13]



#Removing 3 last rows

fires_mo_c = fires_mo_c.iloc[:21]



time = fires_mo_c.Ano.loc[0:21]

months = []



for i in range(fires_mo_c.iloc[:,1:].shape[0]):

    months += [fires_mo_c.iloc[i,1:]]    



#Concatenating the lists    

long_ts = pd.DataFrame(pd.concat(months, axis=0))





#Generating the dates again

long_ts = long_ts.set_index(pd.date_range('1999-01','2019-12',freq='MS').strftime("%Y-%b"))

long_ts.columns = ['Fires']



#Dealing with non-existing values in series

long_ts[long_ts.Fires=='-']=0





#Turning into numerical    

long_ts = long_ts.astype(int)



#Eliminating two last zeros and october because it hasn't ended yet

long_ts = long_ts[:-3]



    

# Building the timeseries

fig, ax = plt.subplots(figsize=(20, 10))

ax.plot(long_ts.index,

        long_ts['Fires'],

        color='purple')

ax.xaxis.set_major_locator(ticker.MultipleLocator(24))







#Ignore it, please

warnings.simplefilter('ignore')



# #Importing some R functions, I didn't find some cool TS treatment ones in python

# import rpy2.robjects as robjects



# # import rpy2's package module

# import rpy2.robjects.packages as rpackages



# # R vector of strings

# from rpy2.robjects.vectors import StrVector



# package_names = ('base', 'seasonal', 'stats')

# if all(rpackages.isinstalled(x) for x in package_names):

#     have_package = True

# else:

#     have_package = False    

# if not have_package:    

#     utils = rpackages.importr('utils')

#     utils.chooseCRANmirror(ind=1)    

#     packnames_to_install = [x for x in package_names if not rpackages.isinstalled(x)]

#     if len(packnames_to_install) > 0:

#         utils.install_packages(StrVector(packnames_to_install))



# base = rpackages.importr('base')

# seasonal = rpackages.importr('seasonal')

# stats = rpackages.importr('stats')



# from rpy2.robjects import pandas2ri



# pandas2ri.activate()



# r_dataframe = pandas2ri.py2ri(long_ts.Fires)



# fires_ts_r = stats.ts(r_dataframe, start=2003, freq=12)



# fires_des_r = seasonal.final(seasonal.seas(fires_ts_r))



# import rpy2.robjects as ro



# fires_des_r_df = ro.DataFrame(base.list(fires_des_r))



# fires_des_pd_df = pandas2ri.ri2py(fires_des_r_df)



# #Plot it in python



# #Making the ts

# fires_des_ts = pd.concat([fires_des_pd_df,pd.DataFrame(long_ts.index)], axis=1)

# fires_des_ts.columns = ['des','tim']





fires_des_ts = pd.read_excel('../input/treated-inpe-series/fires_des_ts.xlsx')





# Building the timeseries

fig, ax = plt.subplots(figsize=(20, 10))

ax.plot(fires_des_ts['tim'],

        fires_des_ts['des'],

        color='purple')

ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

#Ignore it, please

warnings.simplefilter('ignore')





#Getting from 1999 to 2019 total fires outbreaks

fires_mo_tot = pd.DataFrame(fires_mo.Total[1:22])



#Setting the index

fires_mo_tot = fires_mo_tot.set_index(time)



# Plotting timeseries

fig, ax = plt.subplots(figsize=(20, 10))

ax.plot(fires_mo_tot.index.astype(int),

        fires_mo_tot['Total'],

        color='purple')

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))





#plotting the max, mean and minimum



fires_stats = np.transpose(fires_mo.iloc[-3:,1:13]).astype('int')

fires_stats.columns = ['Maximum', 'Mean', 'Minimum']



#Generating timestamps

fires_stats = fires_stats.set_index(pd.date_range('1999-01','1999-12',freq='MS').strftime("%b"))





fig, ax = plt.subplots(figsize=(20, 10))

ax.plot(fires_stats.index,

        fires_stats['Maximum'],

        fires_stats.index,

        fires_stats['Mean'],

        fires_stats.index,

        fires_stats['Minimum'], label='sine')

ax.legend(('Maximum','Mean', 'Minimum'))



#How does the 2019 outbreaks are doing compared to the other years?



fires_comps = np.transpose(fires_mo.iloc[2:23,1:11]).astype('int')

fires_comps.columns = fires_mo.iloc[2:23,0]





list_comps = []

#Looping to see the compared years with 2019

for i in range(fires_comps.shape[1]):

    list_comps += [round((fires_comps.iloc[:,20]/fires_comps.iloc[:,i])*100,2)]

    

df_comps = pd.concat(list_comps, axis=1)

df_comps.columns = fires_mo.iloc[2:23,0]



#Generating timestamps

df_comps = df_comps.set_index(pd.date_range('1999-01','1999-10',freq='MS').strftime("%b")).iloc[:,:-1]



df_comps