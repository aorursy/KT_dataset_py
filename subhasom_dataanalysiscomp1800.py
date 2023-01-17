import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



#Makre sure files are in right place

data_dc = pd.read_csv('../input/DailyCustomers.csv')

data_sm = pd.read_csv('../input/StoreMarketing.csv')

data_ov = pd.read_csv('../input/StoreOverheads.csv')

data_sz = pd.read_csv('../input/StoreSize.csv')

data_st = pd.read_csv('../input/StoreStaff.csv')
data_dc.info()

data_sm.info()

data_ov.info()

data_sz.info()

data_st.info()
data_dc.head()
# prepare data frame

df = data_dc.iloc[:100,:]



data = []



# Creating trace

for col in data_dc.columns: 

    if(col != 'Date'):

        exec('yax = df.'+col)

        data.append(go.Scatter(x = df.Date, y = yax, mode = "lines+markers", name = col))



layout = dict(title = 'Top 100 Daily Customers Visit', xaxis= dict(title= 'Customer Visit',ticklen= 5,zeroline= False))

fig = dict(data = data, layout = layout)

iplot(fig)
fig = ff.create_distplot([data_dc.QSN],['QSN'],bin_size=5)

iplot(fig, filename='Basic Distplot for QSN')
hist_data = []

group_labels = []

for col in data_dc.columns:

    if(col != 'Date'):

        exec('zax = data_dc.'+col)

        hist_data.append(zax)

        group_labels.append(col)

        

fig = ff.create_distplot(hist_data, group_labels, bin_size=5)

iplot(fig, filename='Customer Frequency in different Company')
dv = data_dc.drop(columns="Date")

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(background_color='white', width=512, height=384).generate(" ".join(dv))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
# create trace

data_ov = data_ov.rename(columns={'Overheads (£)':'Overheads'})

data = []

data.append(go.Bar(x = data_ov.Id, y = data_ov.Overheads, name = "Overheads", text = 'Overhead Cost'))

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# create trace

data_sm = data_sm.rename(columns={'Marketing (£)':'Marketing'})

data = []

data.append(go.Bar(x = data_sm.Id, y = data_sm.Marketing, name = "Marketing", marker = dict(color = 'rgba(255, 105, 24, 1)'), text = 'Marketing Cost'))

layout = go.Layout(barmode = "relative")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# prepare data

dataframe = data_dc

dataa = dataframe.loc[:,["CFG","MAJ", "SGA"]]

dataa["index"] = np.arange(1,len(dataa)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(dataa, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
data_sz.sort_values(by=['Id'], inplace=True)

data_st.sort_values(by=['Id'], inplace=True)

frames = [data_sz, data_st]

data_szst = pd.merge(data_sz,

                 data_st[['Id', 'Staff']],

                 on='Id')

data_szst = data_szst.rename(columns={'Size (msq)':'Size'})

data_szst.head()
data = []



# Creating trace

data.append(go.Scatter(x = data_szst.Size, y = data_szst.Staff, mode = "markers", marker = dict(size = data_szst.Staff*2), text= data_szst.Id))

layout = dict(hovermode= 'closest', title = 'Sapce of store vs staff number for all companies',

              xaxis= dict(title= 'Size per square Metter',ticklen= 5,zeroline= False, showgrid=False, showline=False),

              yaxis= dict(title= 'No of Staff',ticklen= 5,zeroline= False, showgrid=False, showline=False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)