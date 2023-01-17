



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data_call = pd.read_csv('../input/service-requests-received-by-the-oakland-call-center.csv', parse_dates = ['DATETIMEINIT' , 'DATETIMECLOSED'])

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt

#data_call.head()

#data_call.groupby('REQCATEGORY')['REQUESTID'].count()





#data_call.info(verbose=True)

#data_call.head()



df_cat=pd.DataFrame({'CATEGORY':data_call['REQCATEGORY'].value_counts().index, 'VAL':data_call['REQCATEGORY'].value_counts().values}) 
data_sort_cat = data_call.sort_values(by=['REQCATEGORY'])

#data_sort_cat1 = data_sort_cat[(data_sort_cat.STATUS !='UNFUNDED')|(data_sort_cat.STATUS !='CANCEL')]

#data_sort_cat1

data_sort_cat1 = data_sort_cat.query("STATUS not in ['UNFUNDED','CANCEL','Cancel']")

#data_sort_cat1 

ds=data_sort_cat1[['REQCATEGORY','DESCRIPTION','STATUS']].groupby(['REQCATEGORY','DESCRIPTION','STATUS']).count()

#data_sort_cat1[['STATUS']].groupby(['STATUS']).count()

data_sort_cat2 = data_sort_cat1.query("REQCATEGORY in ['ILLDUMP']")

#data_sort_cat2
df_cat=pd.DataFrame({'CATEGORY':data_sort_cat1['REQCATEGORY'].value_counts().index, 'VAL':data_sort_cat1['REQCATEGORY'].value_counts().values}) 

import plotly.plotly as py

import plotly.graph_objs as go



label = df_cat['CATEGORY']

#data_call['REQCATEGORY']

#df_cat['CATEGORY']

values = df_cat['VAL'] 

#data_call['REQCATEGORY'].value_counts()

#df_cat['VAL']



#trace = go.Pie(labels=labels, values=values)



#py.iplot([trace], filename='pie_chart')

plt.rcdefaults()

plt.rcParams.update({'font.size': 8})

fig, ax = plt.subplots()



index = np.arange(len(label))



ax.barh(index, values, align='center',

        color='green', ecolor='black')

ax.set_yticks(index)

ax.set_yticklabels(label)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Number of events',fontsize=10)

ax.set_ylabel('Category', fontsize=10)

ax.set_title('Event categories',fontsize=10)



plt.barh(index, values, align='center',

        color='green', ecolor='black')



#plt.set_yticks(index)

#plt.set_yticklabels(values)

#plt.invert_yaxis()

#plt.ylabel('Category', fontsize=10)

#plt.xlabel('Number of events', fontsize=10)

#plt.xticks(index, label, fontsize=8, rotation=90)

#plt.title('Event categories')

plt.show()
data_sort_call = data_call.query("STATUS not in ['UNFUNDED','CANCEL','Cancel']")

#data_sort_call.head()

#data_sort_call.count()
#data_sort_call.DATETIMEINIT.min()
#data_sort_call.DATETIMEINIT.max()
import calendar

import datetime
dateinit = pd.DataFrame(data_sort_call.DATETIMEINIT)

#dateinit
dateinit['year']=pd.DatetimeIndex(dateinit['DATETIMEINIT']).year

dateinit['month']=pd.DatetimeIndex(dateinit['DATETIMEINIT']).month

dateinit['date']=pd.DatetimeIndex(dateinit['DATETIMEINIT']).date

#dateinit.head()


#dateinit = dateinit.reset_index().set_index('year', drop=False)

#dateinit.set_index(['year','month','DATETIMEINIT'])

#dateinit.set_index(['date','year','month','DATETIMEINIT'])



#dateinit.head()

#dateinit[['DATETIMEINIT']].groupby(['date'])

#dateinit[['DATETIMEINIT']].groupby(['year','month']).count()

#dateinit.groupby('year','month').count()
DI = dateinit[['date','year','month','DATETIMEINIT']].groupby(['year','month']).count()

#DI = dateinit



#dateinit['count_year']=pd.DataFrame(dateinit[['year','DATETIMEINIT']].groupby(['year']).count())

#DI.head()
from pandas.tools.plotting import radviz,scatter_matrix,bootstrap_plot,parallel_coordinates

import brewer2mpl

from matplotlib import rcParams



#colorbrewer2 Dark2 qualitative color table

dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors



rcParams['figure.figsize'] = (10, 6)

rcParams['figure.dpi'] = 150

#rcParams['axes.color_cycle'] = dark2_colors

rcParams['lines.linewidth'] = 2

rcParams['axes.facecolor'] = 'white'

rcParams['font.size'] = 14

rcParams['patch.edgecolor'] = 'white'

rcParams['patch.facecolor'] = dark2_colors[0]

rcParams['font.family'] = 'StixGeneral'



df = pd.DataFrame(DI, index=DI.index);

#df = df.cumsum();



plt.figure(); 

df.plot(); 

plt.legend(loc='best');
#DIN = dateinit[['date','DATETIMEINIT']].groupby(['date'],axis=0).count().to_frame()

DIN = dateinit.groupby('date')['DATETIMEINIT'].count().to_frame().reset_index()



#DIN.rename(columns= {0: 'date',1: 'DATETIMEINIT'})

#DIN.set_index(['date','DATETIMEINIT'])

#DIN.head()

#pd.DataFrame({'date':DIN.index, 'DATETIMEINIT':DIN.values})

# import plotly

import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis

data = [go.Scatter(x=DIN.date, y=DIN.DATETIMEINIT)]



# specify the layout of our figure

layout = dict(title = "Number of calls per Month",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)