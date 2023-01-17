# 1.0 Call libraries

%reset -f

# 1.1 For data manipulations

import numpy as np

import pandas as pd

import datetime

# 1.2 For plotting

import matplotlib.pyplot as plt

#import matplotlib

#import matplotlib as mpl     # For creating colormaps

import seaborn as sns

# 1.3 For data processing

from sklearn.preprocessing import StandardScaler

# 1.4 OS related

import os





from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio

pio.templates.default = "plotly_dark"

import warnings

warnings.filterwarnings('ignore')





# 1.5 Go to folder containing data file

os.chdir("../input")

os.listdir()            # List all files in the folderagegp = pd.read_csv("AgeGroupDetails.csv")

#.6 Read All Indices Files



indiavix = pd.read_csv("INDIAVIX.csv")

nifty50=   pd.read_csv("NIFTY 50.csv")

niftynext50 =pd.read_csv('NIFTY NEXT 50.csv')

nifty100 =pd.read_csv("NIFTY 100.csv")

nifty500 =pd.read_csv("NIFTY 500.csv")

niftyauto = pd.read_csv("NIFTY AUTO.csv")

niftybank = pd.read_csv("NIFTY BANK.csv")

niftyfmcg = pd.read_csv("NIFTY FMCG.csv")

niftyit =   pd.read_csv("NIFTY IT.csv")

niftymetal =pd.read_csv("NIFTY METAL.csv")

niftymidcap150=pd.read_csv("NIFTY NEXT 50.csv")

niftypharma =  pd.read_csv("NIFTY PHARMA.csv")

niftysmallcap= pd.read_csv("NIFTY SMALLCAP 250.csv")



# Adding column year,month,week columns in all Indices

nifty50['year'] = pd.DatetimeIndex(nifty50['Date']).year

nifty50['month'] = pd.DatetimeIndex(nifty50['Date']).month

nifty50['week'] = pd.DatetimeIndex(nifty50['Date']).week







niftynext50['year'] = pd.DatetimeIndex(niftynext50['Date']).year

niftynext50['month'] = pd.DatetimeIndex(niftynext50['Date']).month

niftynext50['week'] = pd.DatetimeIndex(niftynext50['Date']).week



nifty100['year'] = pd.DatetimeIndex(nifty100['Date']).year

nifty100['month'] = pd.DatetimeIndex(nifty100['Date']).month

nifty100['week'] = pd.DatetimeIndex(nifty100['Date']).week



nifty500['year'] = pd.DatetimeIndex(nifty500['Date']).year

nifty500['month'] = pd.DatetimeIndex(nifty500['Date']).month

nifty500['week'] = pd.DatetimeIndex(nifty500['Date']).week



niftyauto['year'] = pd.DatetimeIndex(niftyauto['Date']).year

niftyauto['month'] = pd.DatetimeIndex(niftyauto['Date']).month

niftyauto['week'] = pd.DatetimeIndex(niftyauto['Date']).week



niftybank['year'] = pd.DatetimeIndex(niftybank['Date']).year

niftybank['month'] = pd.DatetimeIndex(niftybank['Date']).month

niftybank['week'] = pd.DatetimeIndex(niftybank['Date']).week



niftyfmcg['year'] = pd.DatetimeIndex(niftyfmcg['Date']).year

niftyfmcg['month'] = pd.DatetimeIndex(niftyfmcg['Date']).month

niftyfmcg['week'] = pd.DatetimeIndex(niftyfmcg['Date']).week



niftyit['year'] = pd.DatetimeIndex(niftyit['Date']).year

niftyit['month'] = pd.DatetimeIndex(niftyit['Date']).month

niftyit['week'] = pd.DatetimeIndex(niftyit['Date']).week





niftymetal['year'] = pd.DatetimeIndex(niftymetal['Date']).year

niftymetal['month'] = pd.DatetimeIndex(niftymetal['Date']).month

niftymetal['week'] = pd.DatetimeIndex(niftymetal['Date']).week



niftymidcap150['year'] = pd.DatetimeIndex(niftymidcap150['Date']).year

niftymidcap150['month'] = pd.DatetimeIndex(niftymidcap150['Date']).month

niftymidcap150['week'] = pd.DatetimeIndex(niftymidcap150['Date']).week



niftypharma['year'] = pd.DatetimeIndex(niftypharma['Date']).year

niftypharma['month'] = pd.DatetimeIndex(niftypharma['Date']).month

niftypharma['week'] = pd.DatetimeIndex(niftypharma['Date']).week



niftysmallcap['year'] = pd.DatetimeIndex(niftysmallcap['Date']).year

niftysmallcap['month'] = pd.DatetimeIndex(niftysmallcap['Date']).month

niftysmallcap['week'] = pd.DatetimeIndex(niftysmallcap['Date']).week



indiavix['year'] = pd.DatetimeIndex(indiavix['Date']).year

indiavix['month'] = pd.DatetimeIndex(indiavix['Date']).month

indiavix['week'] = pd.DatetimeIndex(indiavix['Date']).week

#Grouping of various indices data Yearwise, Monthwise and Weekwise



# Grouping of Week wise Nifty Data

grpnifty50 = nifty50.groupby(['year','month','week'],as_index = False)

grpnifty50.groups

gpnifty50 =grpnifty50.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise Nifty Next50 Data

grpniftynext50 = niftynext50.groupby(['year','month','week'],as_index = False)

grpniftynext50.groups

gpniftynext50 =grpniftynext50.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise Nifty100 Data

grpnifty100 = nifty100.groupby(['year','month','week'],as_index = False)

grpnifty100.groups

gpnifty50 =grpnifty50.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise Nifty500 Data

grpnifty500 = nifty500.groupby(['year','month','week'],as_index = False)

grpnifty500.groups

gpnifty500 =grpnifty500.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise Nifty auto Data

grpniftyauto = niftyauto.groupby(['year','month','week'],as_index = False)

grpniftyauto.groups

gpniftyauto =grpniftyauto.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise Niftybank Data

grpniftybank = niftybank.groupby(['year','month','week'],as_index = False)

grpniftybank.groups

gpniftybank =grpniftybank.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise Niftyfmcg Data

grpniftyfmcg = niftyfmcg.groupby(['year','month','week'],as_index = False)

grpniftyfmcg.groups

gpniftyfmcg =grpniftyfmcg.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise Niftyit Data

grpniftyit = niftyit.groupby(['year','month','week'],as_index = False)

grpniftyit.groups

gpniftyit =grpniftyit.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise niftymetal Data

grpniftymetal = niftymetal.groupby(['year','month','week'],as_index = False)

grpniftymetal.groups

gpniftymetal =grpniftymetal.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise niftymidcap150 Data

grpniftymidcap150 = niftymidcap150.groupby(['year','month','week'],as_index = False)

grpniftymidcap150.groups

gpniftymidcap150 =grpniftymidcap150.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise niftypharma Data

grpniftypharma = niftypharma.groupby(['year','month','week'],as_index = False)

grpniftypharma.groups

gpniftypharma =grpniftypharma.agg({'Close':np.mean,'Volume':np.mean,'Turnover':np.mean,'P/E':np.mean,'P/B':np.mean})



# Grouping of Month wise niftysmallcap Data

grpniftysmallcap = niftysmallcap.groupby(['year','month','week'],as_index = False)

grpniftysmallcap.groups

gpniftysmallcap =grpniftysmallcap.agg({'Close':np.mean})



# Grouping of Month wise IndiaVix Data

grpindiavix = indiavix.groupby(['year','month','week'],as_index = False)

grpindiavix.groups

gpindiavix =grpindiavix.agg({'Close':np.mean})





# Changing Column names of various indices group dataframes



nifty50_col_names  = {

                 'Close' :  'Nifty50close',

                 'P/E'   :  'Nifty50PE',

                 'P/B'   :  'Nifty50PB',

                 'Volume':  'Nifty50Volume',

                 'Turnover':'Nifty50turnover'}



gpnifty50.rename(

         columns = nifty50_col_names,

         inplace = True

         )



niftyauto_col_names  = {

                 'Close' :  'niftyautoclose',

                 'P/E'   :  'niftyautoPE',

                 'P/B'   :  'niftyautoPB',

                 'Volume':  'niftyautoVolume',

                 'Turnover':'niftyautoturnover'}



gpniftyauto.rename(

         columns = niftyauto_col_names,

         inplace = True

         )



niftymetal_col_names  = {

                 'Close' :  'niftymetalclose',

                 'P/E'   :  'niftymetalPE',

                 'P/B'   :  'niftymetalPB',

                 'Volume':  'niftymetalVolume',

                 'Turnover':'niftymetalturnover'}



gpniftymetal.rename(

         columns = niftymetal_col_names,

         inplace = True

         )





niftyit_col_names  = {

                 'Close' :  'niftyitclose',

                 'P/E'   :  'niftyitPE',

                 'P/B'   :  'niftyitPB',

                 'Volume':  'niftyitVolume',

                 'Turnover':'niftyitturnover'}



gpniftyit.rename(

         columns = niftyit_col_names,

         inplace = True

         )





niftypharma_col_names  = {

                 'Close' :  'niftypharmaclose',

                 'P/E'   :  'niftypharmaPE',

                 'P/B'   :  'niftypharmaPB',

                 'Volume':  'niftypharmaVolume',

                 'Turnover':'niftypharmaturnover'}



gpniftypharma.rename(

         columns = niftypharma_col_names,

         inplace = True

         )







niftybank_col_names  = {

                 'Close' :  'niftybankclose',

                 'P/E'   :  'niftybankPE',

                 'P/B'   :  'niftybankPB',

                 'Volume':  'niftybankVolume',

                 'Turnover':'niftybankturnover'}

gpniftybank.rename(

         columns = niftybank_col_names,

         inplace = True

         )



niftyfmcg_col_names  = {

                 'Close' :  'niftyfmcgclose',

                 'P/E'   :  'niftyfmcgPE',

                 'P/B'   :  'niftyfmcgPB',

                 'Volume':  'niftyfmcgVolume',

                 'Turnover':'niftyfmcgturnover'}



gpniftyfmcg.rename(

         columns = niftyfmcg_col_names,

         inplace = True

         )



niftysmallcap_col_names  = {

                 'Close' :  'niftysmallcapclose',

                 'P/E'   :  'niftysmallcapPE',

                 'P/B'   :  'niftysmallcapPB',

                 'Volume':  'niftysmallcapVolume',

                 'Turnover':'niftysmallcapturnover'}



gpniftysmallcap.rename(

         columns = niftysmallcap_col_names,

         inplace = True

         )



niftymidcap150_col_names  = {

                 'Close' :  'niftymidcap150close',

                 'P/E'   :  'niftymidcap150PE',

                 'P/B'   :  'niftymidcap150PB',

                 'Volume':  'niftymidcap150Volume',

                 'Turnover':'niftymidcap150turnover'}



gpniftymidcap150.rename(

         columns = niftymidcap150_col_names,

         inplace = True

         )



niftynext50_col_names  = {

                 'Close' :  'niftynext50close',

                 'P/E'   :  'niftynext50PE',

                 'P/B'   :  'niftynext50PB',

                 'Volume':  'niftynext50Volume',

                 'Turnover':'niftynext50turnover'}



gpniftynext50.rename(

         columns = niftynext50_col_names,

         inplace = True

         )



indiavix_col_names  = {

                 'Close' :  'indiavixclose',

                 }



gpindiavix.rename(

         columns = indiavix_col_names,

         inplace = True

         )

# Joining Grouped data of Various Indices

#Indices_Tables = [gpnifty50,gpniftyauto,gpniftymetal,gpniftyit,gpniftypharma,gpniftyfmcg,gpniftysmallcap,gpniftymidcap150,gpniftynext50]

#All_Indices = pd.concat(Indices_Tables,axis =1,keys =['year','month','week'],join ='outer',ignore_index = True,how='left')





tempindex  = gpnifty50.merge(gpniftyauto,on=['year','month','week'],how = 'left')

tempindex1 = tempindex.merge(gpniftymetal,on=['year','month','week'],how = 'left')

tempindex2 = tempindex1.merge(gpniftyit,on=['year','month','week'],how = 'left')

tempindex3 = tempindex2.merge(gpniftypharma,on=['year','month','week'],how = 'left')

tempindex4 = tempindex3.merge(gpniftyfmcg,on=['year','month','week'],how = 'left')

tempindex5 = tempindex4.merge(gpniftybank,on=['year','month','week'],how='left')



tempindex6 = tempindex5.merge(gpniftysmallcap,on=['year','month','week'],how = 'left')

tempindex7 = tempindex6.merge(gpniftymidcap150,on=['year','month','week'],how = 'left')

All_Indices = tempindex7.merge(gpniftynext50,on=['year','month','week'],how = 'left')

               
#cutting of Nifty Data by Recession ERA's

l2 = ['Before2003','Before2008','Before2016','Before2020','After2020']

All_Indices['ERA'] = pd.cut(All_Indices['year'],

                        bins = [1999,2002,2007,2015,2019,2020],labels = l2)



#Dividing the Data Era Wise

All_Indices_Before2003 = All_Indices.loc[All_Indices['ERA'] == 'Before2003']

All_Indices_Before2008 = All_Indices.loc[All_Indices['ERA'] == 'Before2008']

All_Indices_Before2016 = All_Indices.loc[All_Indices['ERA'] == 'Before2016']

All_Indices_Before2020 = All_Indices.loc[All_Indices['ERA'] == 'Before2020']

All_Indices_After2020 =  All_Indices.loc[All_Indices['ERA'] == 'After2020']
#See How Nifty Data work year wise using line chart

columns = ['Nifty50close','niftyitclose','niftypharmaclose','niftyfmcgclose','niftybankclose','niftymetalclose',

           'niftysmallcapclose','niftymidcap150close']

#How Indices Chart Behaves in period from year 2000 to 2020   

fig =plt.figure(figsize= (20,25))

for i in range(len(columns)):

    plt.subplot(4,2,i+1).set_title(columns[i],fontdict={'size':18,'weight':'bold','color':'blue'})

    sns.lineplot(x="year",y=All_Indices[columns[i]],data = All_Indices)
#How Price Earning Ratio(P/E) behaves Heapmap behaves from year 2000 to 2020

grouped = gpnifty50.groupby(['year', 'Nifty50PE'])

df_wqd = grouped['Nifty50close'].mean().unstack()

sns.heatmap(df_wqd, cmap = plt.cm.Spectral)
#How Price to Book Value(PB) behaves Heapmap behaves from year 2000 to 2020

grouped = gpnifty50.groupby(['year', 'Nifty50PB'])

df_wqd = grouped['Nifty50close'].mean().unstack()

sns.heatmap(df_wqd, cmap = plt.cm.Spectral)

#How Indices chart behaves in period 2000 to 2002 December.

fig =plt.figure(figsize= (20,25))

for i in range(len(columns)):

    plt.subplot(4,2,i+1)

    sns.lineplot(x="year",y=All_Indices_Before2003[columns[i]],data = All_Indices_Before2003).set_title(columns[i],fontdict={'size':18,'weight':'bold','color':'blue'})
#How Indices chart behaves in period 2003 to 2007 December.

fig =plt.figure(figsize= (15,20))

for i in range(len(columns)):

    plt.subplot(4,2,i+1)

    sns.lineplot(x="year",y=All_Indices_Before2008[columns[i]],data = All_Indices_Before2008).set_title(columns[i],fontdict={'size':18,'weight':'bold','color':'blue'})
#How Indices chart behaves in period 2008 to 20016 December.

fig =plt.figure(figsize= (20,25))

for i in range(len(columns)):

    plt.subplot(4,2,i+1)

    sns.lineplot(x="year",y=All_Indices_Before2016[columns[i]],data = All_Indices_Before2016).set_title(columns[i],fontdict={'size':18,'weight':'bold','color':'blue'})
#How Indices Chart Behaves in period Jan 2017 to December 2019   

fig =plt.figure(figsize= (20,25))

for i in range(len(columns)):

    plt.subplot(4,2,i+1)

    sns.lineplot(x="year",y=All_Indices_Before2020[columns[i]],data = All_Indices_Before2020).set_title(columns[i],fontdict={'size':18,'weight':'bold','color':'blue'})
#How Indices Chart Behaves in period January 2020 to May 2020   

fig =plt.figure(figsize= (20,25))

for i in range(len(columns)):

    plt.subplot(4,2,i+1)

    sns.lineplot(x="week",y=All_Indices_After2020[columns[i]],data = All_Indices_After2020).set_title(columns[i],fontdict={'size':18,'weight':'bold','color':'blue'})