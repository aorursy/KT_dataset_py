

#importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import geopandas as gpd

from matplotlib import cm

import matplotlib

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING





from IPython.display import HTML,display

import warnings

warnings.filterwarnings("ignore")







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

sc1 = pd.read_csv('../input/crime-in-india/crime/crime/02_01_District_wise_crimes_committed_against_SC_2001_2012.csv')



sc13= pd.read_csv("../input/crime-in-india/crime/02_01_District_wise_crimes_committed_against_SC_2013.csv")

sc13.columns



sc13 = sc13[['STATE/UT', 'DISTRICT', 'Year', 'Murder', 'Rape',

       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt','Prevention of atrocities (POA) Act',

       'Protection of Civil Rights (PCR) Act',

        'Other Crimes Against SCs']]

#combining 2 CSV files



frames = [sc1 , sc13]



sc = pd.concat(frames)



sc['STATE/UT'] = sc['STATE/UT'].str.capitalize()

sc['DISTRICT'] = sc['DISTRICT'].str.capitalize()





sc['STATE/UT'].unique()



sc['STATE/UT'].replace(

    to_replace='Delhi ut',

    value='Delhi',

    inplace=True

)



sc['STATE/UT'].replace(

    to_replace='A&n islands',

    value='A & n islands',

    inplace=True

)









sc['STATE/UT'].replace(

    to_replace='D&n haveli',

    value='D & n haveli',

    inplace=True

)

sc['STATE/UT'].unique()
sc.head()
yearw = sc[sc.DISTRICT == 'Total']

yearw = yearw.groupby(['Year'])['Murder', 'Rape',

       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',

       'Prevention of atrocities (POA) Act',

       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()

yearw['sum'] = yearw.drop('Year', axis=1).sum(axis=1)

yearw = yearw[['Year','sum']]
scy = sc[sc.DISTRICT == 'Total']

scy = scy.groupby(['Year'])['Murder', 'Rape',

       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',

       'Prevention of atrocities (POA) Act',

       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()



crimes = ['Murder', 'Rape',

       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',

       'Prevention of atrocities (POA) Act',

       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs']



fig = go.Figure()

fig.add_trace(go.Scatter(x= scy['Year'], y= scy['Murder'],

                    name='Murder',line=dict(color='pink', width=4)))

fig.add_trace(go.Scatter(x= scy['Year'], y= scy['Rape'],

                    name='Rape',line=dict(color='green', width=4)))

fig.add_trace(go.Scatter(x= scy['Year'], y= scy['Kidnapping and Abduction'],

                    name='Kidnapping and Abduction',line=dict(color='orange', width=4)))

fig.add_trace(go.Scatter(x= scy['Year'], y= scy['Dacoity'],

                    name='Dacoity',line=dict(color='yellow', width=4)))

fig.add_trace(go.Scatter(x= scy['Year'], y= scy['Robbery'],

                    name='Robbery',line=dict(color='black', width=4)))

fig.add_trace(go.Scatter(x= scy['Year'], y= scy['Arson'],

                    name='Arson',line=dict(color='skyblue', width=4)))

fig.add_trace(go.Scatter(x= scy['Year'], y= scy['Hurt'],

                    name='Hurt',line=dict(color='royalblue', width=4)))

fig.add_trace(go.Scatter(x= scy['Year'], y= scy['Prevention of atrocities (POA) Act'],

                    name='Atrocities',line=dict(color='firebrick', width=4)))

fig.add_trace(go.Scatter(x= scy['Year'], y= scy['Protection of Civil Rights (PCR) Act'],

                    mode='lines+markers',

                    name='Civil Rights Violations'))

fig.add_trace(go.Scatter(x= scy['Year'], y= scy['Other Crimes Against SCs'],

                    name='Other Crimes',line=dict(color='red', width=4)))



fig.update_layout(uniformtext_minsize= 20,

    title_text="Total Crimes Against Scs 2001-2013",

    

                 )

    

fig.show()



scy2 = sc[sc.DISTRICT == 'Total']

scy2 = scy2.groupby(['Year'])['Murder', 'Rape',

       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',

       'Prevention of atrocities (POA) Act',

       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()



#Plotting Graphs

import itertools

sns.set_context("talk")

plt.style.use("fivethirtyeight")

palette = itertools.cycle(sns.color_palette("dark"))

columns = ['Murder', 'Rape',

       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',

       'Prevention of atrocities (POA) Act',

       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs']

plt.figure(figsize=(20,30))

plt.style.use('fivethirtyeight')

for i,column in enumerate(columns):

    plt.subplot(5,2,i+1)

    ax= sns.barplot(data= scy2,x='Year',y= column ,color=next(palette) )

    plt.xlabel('')

    plt.ylabel('')

    plt.title(column,size = 20)

    for p in ax.patches:

             ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),

                 textcoords='offset points')

    

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()

scy = scy.append(scy.sum().rename('total'))

scy['Year'].replace(26091, 'Total', inplace=True)

scy = scy[scy['Year'] == 'Total']

scy_t = scy.T.reset_index()

scy_t
import plotly.graph_objects as go



labels = ['Murder', 'Rape','Kidnapping', 'Dacoity', 'Robbery', 'Arson', 'Hurt','Atrocities  Act',

         'Civil Rights Act', 'Other Crimes']

values = [8576, 17991, 5305, 440,1015,2906, 54055 , 138533, 4332,176488]



fig = go.Figure(data=[go.Pie(labels=labels, values=values ,textinfo='label+percent',

                              )])

fig.update_layout(

    uniformtext_minsize= 20,

    title_text="Distribution of Crimes Against Scs 2001-2013",

    paper_bgcolor='rgb(233,233,233)',

    autosize=False,

    width=700,

    height=700)

fig.show()

stateyr = sc[sc.DISTRICT == 'Total']

stateyr = stateyr.groupby(['Year','STATE/UT'])['Murder', 'Rape',

       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',

       'Prevention of atrocities (POA) Act',

       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()
stateyr['sum'] =  stateyr.iloc[:, 2:].sum(axis=1)
stateyr2 = stateyr.groupby('STATE/UT')['sum'].sum().reset_index()
stateyr2 = stateyr2.sort_values('sum', ascending = False)
plt.figure(figsize = (12,12))

sns.set_context("talk")

plt.style.use("fivethirtyeight")

ax = sns.barplot(x = 'sum', y = 'STATE/UT', data = stateyr2, palette = 'bright', edgecolor = 'black')

plt.title('Total crimes against SCs (2001- 2013)')

for p in ax.patches:

        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")

         
states = ['Uttar pradesh','Rajasthan' ,'Madhya pradesh' , 'Andhra pradesh', 'Bihar', 'Karnataka' , 'Odisha' , 'Tamil nadu','Gujarat', 'Maharashtra']

sns.set_context("talk")

plt.style.use("fivethirtyeight")

plt.figure(figsize = (23,28))



for i, s in enumerate(states):

    plt.subplot(5,2,i+1)

    stateyr3 = stateyr[stateyr['STATE/UT'] == s]

    ax = sns.barplot(x = 'Year' , y = 'sum' , data = stateyr3,ci=None , palette = 'colorblind' , edgecolor = 'blue')

    plt.title(s , size = 25)

    for p in ax.patches:

             ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),

                 textcoords='offset points')

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

scs = sc[sc.DISTRICT == 'Total']

scs = scs.groupby(['STATE/UT'])['Murder', 'Rape',

       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',

       'Prevention of atrocities (POA) Act',

       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()



scs1 = scs[(scs.Murder > 100) & (scs.Rape > 100)]

sns.set_context("talk")



plt.figure(figsize=(20,30))

plt.style.use('fivethirtyeight')



for i,column in enumerate(columns):

    scs1 = scs1.sort_values(column,ascending = False)

    plt.subplot(5,2,i+1)

    ax = sns.barplot(data= scs1,x= column ,y='STATE/UT',palette = 'dark' )

    plt.xlabel('')

    plt.ylabel('')

    plt.title(column,size = 20)

    for p in ax.patches:

        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")

         

   

    

    

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()

scs['sum'] = scs.sum(axis = 1)

new_row = scs.iloc[[1]]

scs = scs.append(new_row, ignore_index = True)

scs.at[35, 'STATE/UT']= 'Telangana'

scs.at[9,'STATE/UT'] = 'Nct of Delhi'



gdf = gpd.read_file("../input/india-states/Igismap/Indian_States.shp")



gdf.st_nm = gdf.st_nm.str.lower()

scs['STATE/UT'] = scs['STATE/UT'].str.lower()



merged = gdf.merge(scs , left_on='st_nm', right_on='STATE/UT')

merged1 = merged.drop(['STATE/UT'], axis=1)

import pysal.viz.mapclassify 

import mapclassify

figsize = (25, 23)

merged1['coords'] = merged1['geometry'].apply(lambda x: x.representative_point().coords[:])

merged1['coords'] = [coords[0] for coords in merged1['coords']]

colors = 8



import pylab as plot

params = {'legend.fontsize': 20,

          'legend.handlelength': 2}

plot.rcParams.update(params)



ax= merged1.dropna().plot(column= 'sum', figsize=figsize, scheme= 'User_Defined',cmap = 'YlGn',edgecolor='black',k = colors,legend = True, classification_kwds=dict(bins=[5000,10000,20000,40000,60000,90000]) )

ax.set_title(" Total Cases", size = 25)

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['sum'], horizontalalignment='center', bbox={'facecolor': 'yellow', 'alpha':0.8, 'pad': 2, 'edgecolor':'black'})



ax.get_legend().set_bbox_to_anchor((0.8, 0.4))

ax.get_legend().set_title('Number of cases')



ax.set_title("Total cases" , size = 30)

ax.axis('off')

leg = ax.get_legend()

for lbl in leg.get_texts():

    label_text = lbl.get_text()

    lower = label_text.split()[0]

    upper = label_text.split()[2]

    new_text = f'{float(lower):,.0f} - {float(upper):,.0f}'

    lbl.set_text(new_text)





plt.axis('equal')



plt.show()
figsize = (25, 20)



cmap = 'YlGn'

ax= merged1.dropna().plot(column= 'Murder', cmap= cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title(" Murder Cases", size = 25)

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Murder'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = matplotlib.colors.Normalize(vmin=merged1['Murder'].min(), vmax= merged1['Murder'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
from matplotlib.colors import Normalize

from matplotlib import cm



sns.set_context("poster")

sns.set_style("darkgrid")

plt.style.use('fivethirtyeight')



figsize = (25, 20)





ax= merged1.dropna().plot(column= 'Rape', cmap= cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title(" Cases of Rape", size = 25)

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Rape'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = Normalize(vmin=merged1['Rape'].min(), vmax= merged1['Rape'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
figsize = (25, 20)





ax= merged1.dropna().plot(column= 'Kidnapping and Abduction', cmap= cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title(" Kidnapping and Abduction Cases" , size = 25)

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Kidnapping and Abduction'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = Normalize(vmin=merged1['Kidnapping and Abduction'].min(), vmax= merged1['Kidnapping and Abduction'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
figsize = (25, 20)





ax= merged1.dropna().plot(column= 'Dacoity', cmap=cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title("Dacoity Cases", size = 25)

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Dacoity'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = Normalize(vmin=merged1['Dacoity'].min(), vmax= merged1['Dacoity'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
figsize = (25, 20)





ax= merged1.dropna().plot(column= 'Robbery', cmap=cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title(" Robbery Cases", size = 25)

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Robbery'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = Normalize(vmin=merged1['Robbery'].min(), vmax= merged1['Robbery'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
figsize = (25, 20)





ax= merged1.dropna().plot(column= 'Arson', cmap=cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title(" Arson Cases", size = 25)

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Arson'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = Normalize(vmin=merged1['Arson'].min(), vmax= merged1['Arson'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
figsize = (25, 20)

plt.style.use("fivethirtyeight")

cmap1 = 'cool'

ax= merged1.dropna().plot(column= 'Hurt', cmap=cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title(" Hurt Cases" , size = 25)

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Hurt'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = Normalize(vmin=merged1['Hurt'].min(), vmax= merged1['Hurt'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
figsize = (25, 20)





ax= merged1.dropna().plot(column= 'Protection of Civil Rights (PCR) Act', cmap=cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title(" Protection of Civil Rights (PCR) Act Cases" , size = 25)

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Protection of Civil Rights (PCR) Act'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = Normalize(vmin=merged1['Protection of Civil Rights (PCR) Act'].min(), vmax= merged1['Protection of Civil Rights (PCR) Act'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
figsize = (25, 20)





ax= merged1.dropna().plot(column= 'Prevention of atrocities (POA) Act', cmap=cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')



ax.set_title(" Prevention of atrocities (POA) Act Cases" , size = 25)

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Prevention of atrocities (POA) Act'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = Normalize(vmin=merged1['Prevention of atrocities (POA) Act'].min(), vmax= merged1['Prevention of atrocities (POA) Act'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
merged1['coords'] = merged1['geometry'].apply(lambda x: x.representative_point().coords[:])

merged1['coords'] = [coords[0] for coords in merged1['coords']]

figsize = (25, 20)





ax= merged1.dropna().plot(column= 'Other Crimes Against SCs', cmap= cmap, figsize=figsize, scheme='equal_interval' ,edgecolor='black')

for idx, row in merged1.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Other Crimes Against SCs'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})





ax.set_title(" Other Crimes Against SCs cases" , size = 25)





norm = Normalize(vmin=merged1['Other Crimes Against SCs'].min(), vmax= merged1['Other Crimes Against SCs'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
scd = sc[sc.DISTRICT != 'Total']

scd = scd.groupby(['DISTRICT'])['Murder', 'Rape',

       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',

       'Prevention of atrocities (POA) Act',

       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()





sns.set_context("talk")



plt.figure(figsize=(20,30))

plt.style.use('fivethirtyeight')



for i,column in enumerate(columns):

    scd1 = scd.sort_values(column,ascending = False)

    scd1 = scd1.head(10)

    plt.subplot(5,2,i+1)

    ax= sns.barplot(data= scd1,x= column ,y='DISTRICT' )

    plt.xlabel('')

    plt.ylabel('')

    plt.title(column,size = 20)

    for p in ax.patches:

        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")

         

    

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()



scd['sum'] = scd['Murder']+scd['Rape']+scd['Kidnapping and Abduction']+scd['Dacoity']+scd['Robbery']+scd['Arson']+scd['Hurt']+scd['Prevention of atrocities (POA) Act']+scd['Protection of Civil Rights (PCR) Act']+scd['Other Crimes Against SCs']
mostviolent = scd.groupby(['DISTRICT'])['sum'].sum().sort_values(ascending = False).reset_index()

mostviolent = mostviolent.head(15)
import plotly.graph_objects as go





# Use textposition='auto' for direct text

fig = go.Figure(data=[go.Bar(

            x= mostviolent['DISTRICT'], y= mostviolent['sum'],

            text= mostviolent['sum'],

            textposition='auto',marker_color='rgb(255, 22, 22)'

        )])

fig.update_layout(title_text='Most Violent Districts')



fig.show()
scsd = sc[sc.DISTRICT!= 'Total']

scsd = scsd.groupby(['STATE/UT', 'DISTRICT'])['Murder', 'Rape',

       'Kidnapping and Abduction', 'Dacoity', 'Robbery', 'Arson', 'Hurt',

       'Prevention of atrocities (POA) Act',

       'Protection of Civil Rights (PCR) Act', 'Other Crimes Against SCs'].sum().reset_index()





states = ['Rajasthan', 'Maharashtra', 'Andhra pradesh', 'Uttar pradesh', 'Bihar','Madhya pradesh']

sns.set_context("talk")

sns.set_style("darkgrid")

plt.style.use('fivethirtyeight')

plt.figure(figsize=(20,20))

for i , state in enumerate(states):

    scsd1 = scsd[scsd['STATE/UT'] == state].sort_values('Rape', ascending = False)

    scsd1 = scsd1.head(10)

    plt.subplot(3,2,i+1)

    ax = sns.barplot(data= scsd1,x= 'Rape' ,y= 'DISTRICT',palette = 'bright' )

    plt.xlabel('Rape Cases')

    plt.ylabel('')

    plt.title(state.capitalize(),size = 20)

    for p in ax.patches:

        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")

         

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()

    
plt.figure(figsize=(20,20))

plt.style.use('fivethirtyeight')

for i , state in enumerate(states):

    scsd1 = scsd[scsd['STATE/UT'] == state].sort_values('Murder', ascending = False)

    scsd1 = scsd1.head(10)

    plt.subplot(3,2,i+1)

    ax = sns.barplot(data= scsd1,x= 'Murder' ,y= 'DISTRICT',palette = 'colorblind' )

    plt.xlabel('Murders')

    plt.ylabel('')

    plt.title(state.capitalize(),size = 20)

    for p in ax.patches:

        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")

         

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()
plt.figure(figsize=(20,20))

plt.style.use('fivethirtyeight')

for i , state in enumerate(states):

    scsd1 = scsd[scsd['STATE/UT'] == state].sort_values('Hurt', ascending = False)

    scsd1 = scsd1.head(10)

    plt.subplot(3,2,i+1)

    ax = sns.barplot(data= scsd1,x= 'Hurt' ,y= 'DISTRICT', palette = 'bright')

    plt.xlabel('Hurts')

    plt.ylabel('')

    plt.title(state.capitalize(),size = 20)

    for p in ax.patches:

        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")

         

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()
plt.style.use('Solarize_Light2')

plt.figure(figsize=(20,20))

for i , state in enumerate(states):

    scsd1 = scsd[scsd['STATE/UT'] == state].sort_values('Prevention of atrocities (POA) Act', ascending = False)

    scsd1 = scsd1.head(10)

    plt.subplot(3,2,i+1)

    ax = sns.barplot(data= scsd1,x= 'Prevention of atrocities (POA) Act' ,y= 'DISTRICT', palette = 'dark' )

    plt.xlabel('Prevention of atrocities (POA) Act')

    plt.ylabel('')

    plt.title(state.capitalize(),size = 20)

    for p in ax.patches:

        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")

         

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()
plt.figure(figsize=(20,20))

plt.style.use('classic')

for i , state in enumerate(states):

    scsd1 = scsd[scsd['STATE/UT'] == state].sort_values('Other Crimes Against SCs', ascending = False)

    scsd1 = scsd1.head(10)

    plt.subplot(3,2,i+1)

    ax = sns.barplot(data= scsd1,x= 'Other Crimes Against SCs' ,y= 'DISTRICT', palette = 'bright' )

    plt.xlabel('Other Crimes Against SCs')

    plt.ylabel('')

    plt.title(state.capitalize(),size = 20)

    for p in ax.patches:

        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")

         

plt.tight_layout()

plt.subplots_adjust(hspace= .3)

plt.show()