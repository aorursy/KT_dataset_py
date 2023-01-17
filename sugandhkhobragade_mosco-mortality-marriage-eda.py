

#importing Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING

import plotly.graph_objects as go

import warnings

warnings.filterwarnings("ignore")







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/mortaliy-moscow-20102020/moscow_stats.csv')

df.head()
df.info()
dfpopy = df.copy()

dfpopy = dfpopy.groupby(['Year'])['NumberOfBirthCertificatesForBoys', 

                                  'NumberOfBirthCertificatesForGirls','StateRegistrationOfBirth',

                                  'StateRegistrationOfDeath',].sum().reset_index()





dfpopy.style.background_gradient(cmap='Greens')
dfpopm = df.copy()

dfpopm = dfpopm.groupby(['Month'])['NumberOfBirthCertificatesForBoys', 

                                   'NumberOfBirthCertificatesForGirls','StateRegistrationOfBirth',

                                   'StateRegistrationOfDeath',].sum().reset_index()

dfpopm.style.background_gradient(cmap='Greens')




fig = go.Figure(data=go.Heatmap(

                   z= df['StateRegistrationOfBirth'],

                   x=df['Month'],

                   y= df['Year'],

                   hoverongaps = False))

fig.update_layout(

    

    title_text= '<b>Birth in Moscow<b>',title_x=0.5,

    paper_bgcolor='aqua',

    plot_bgcolor = "aqua",

    yaxis = dict(

        tickmode = 'array',

        tickvals = [2010, 2011, 2012, 2013, 2014, 

                    2015, 2016, 2017 , 2018, 2019, 2020],

        ticktext = ['2010', '2011', '2012', '2013', '2014', 

                    '2015', '2016', '2017' , '2018', '2019', '2020'],),

     )

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()




fig = go.Figure(data=go.Heatmap(

                   z= df['StateRegistrationOfDeath'],

                   x=df['Month'],

                   y= df['Year'],

                  

                   hoverongaps = False))

fig.update_layout(

    title_text= '<b> Death in Moscow<b>',title_x=0.5,

    paper_bgcolor='aqua',

    plot_bgcolor = "aqua",

    yaxis = dict(

        tickmode = 'array',

        tickvals = [2010, 2011, 2012, 2013, 2014, 2015, 

                    2016, 2017 , 2018, 2019, 2020],

        ticktext = ['2010', '2011', '2012', '2013', '2014', 

                    '2015', '2016', '2017' , '2018', '2019', '2020'],),

    

    

    )

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
df2 = df.copy()

df2['BirthRate'] = df2['StateRegistrationOfBirth'] / df2['TotalPopulationThisYear'] * 1000

df2['DeathRate'] = df2['StateRegistrationOfDeath']/df2['TotalPopulationThisYear'] * 1000

df2['BirthRateF'] = df2['NumberOfBirthCertificatesForBoys'] / df2['TotalPopulationThisYear'] * 1000

df2['BirthRateM'] = df2['NumberOfBirthCertificatesForGirls']/df2['TotalPopulationThisYear'] * 1000
df3 = df2.groupby(['Year'])['BirthRate','DeathRate','BirthRateF','BirthRateM' ,].sum().reset_index()

df4 = df2.groupby(['Month'])['BirthRate','DeathRate','BirthRateF','BirthRateM' ,].sum().reset_index()



fig = go.Figure(data=[

    go.Bar(name='Birth Rate Male', x= df3['Year'], y= df3['BirthRateF']),

    go.Bar(name='Birth Rate Female', x= df3['Year'], y=df3['BirthRateM'])

])

# Change the bar mode

fig.update_layout(barmode='group',

                  

                 xaxis = dict(

        tickmode = 'array',

        tickvals = [2010, 2011, 2012, 2013, 2014, 2015, 

                    2016, 2017 , 2018, 2019, 2020],

        ticktext = ['2010', '2011', '2012', '2013', '2014', 

                    '2015', '2016', '2017' , '2018', '2019', '2020'],),

        

        paper_bgcolor='aqua',

        plot_bgcolor = "aqua",

        title_text= '<b>Birth Rate of Male & Female Year Wise<b>',title_x=0.5,)

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Birth Rate Male', x= df4['Month'], y= df4['BirthRateF']),

    go.Bar(name='Birth Rate Female', x= df4['Month'], y=df4['BirthRateM'])

])

# Change the bar mode

fig.update_layout(barmode='group',

                 xaxis={'categoryorder':'array', 

                        'categoryarray':['January','February',

                                         'March','April', 'May', 

                                         'June', 'July', 'August', 

                                         'September', 'October', 'November', 'December',]},

                 

                 paper_bgcolor='aqua',

                 plot_bgcolor = "aqua",

                 title_text= '<b> Birth Rate of Male & Female Monthwise<b>',title_x=0.5,)

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Birth Rate', x= df3['Year'], y= df3['BirthRate']),

    go.Bar(name='Death Rate', x= df3['Year'], y=df3['DeathRate'])

])

# Change the bar mode

fig.update_layout(barmode='group',

                 xaxis = dict(

        tickmode = 'array',

        tickvals = [2010, 2011, 2012, 2013, 2014, 

                    2015, 2016, 2017 , 2018, 2019, 2020],

        ticktext = ['2010', '2011', '2012', '2013', 

                    '2014', '2015', '2016', '2017' , 

                    '2018', '2019', '2020'],),

        paper_bgcolor='aqua',

        plot_bgcolor = "aqua",

        title_text= '<b> Birth Rate & Death Rate Yearwise<b>',title_x=0.5,)



fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Birth Rate', x= df4['Month'], y= df4['BirthRate']),

    go.Bar(name='Death Rate', x= df4['Month'], y=df4['DeathRate'])

])

# Change the bar mode

fig.update_layout(barmode='group',

                  xaxis={'categoryorder':'array', 

                         'categoryarray':['January','February','March',

                                          'April', 'May', 'June', 'July', 

                                          'August', 'September', 'October',

                                          'November', 'December',]},

                 paper_bgcolor='aqua',

                 plot_bgcolor = "aqua",

                 title_text= '<b>Birth Rate & Death Rate Month wise<b>',title_x=0.5,)

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
dfm = df.groupby(['Year'])['StateRegistrationOfMarriage'].sum().reset_index()

#dfm = dfm.loc[dfm['Year'] != 2020]

sns.set_context("talk")

plt.style.use("fivethirtyeight")

plt.figure(figsize = (14,10))



ax = sns.barplot(x = 'Year', y = 'StateRegistrationOfMarriage' , data = dfm, ci = None, palette = 'Accent', edgecolor='black')

plt.title("Marriages in Moscow over the years", pad = 20)

plt.ylabel('Marriages')

for p in ax.patches:

             ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),

                 textcoords='offset points')
fig = go.Figure(data=go.Heatmap(

                   z= df['StateRegistrationOfMarriage'],

                   x=df['Month'],

                   y= df['Year'],

                   hoverongaps = False))

fig.update_layout(

    title_text= '<b> Marriages in  Moscow<b>',title_x=0.5,

    yaxis = dict(

        tickmode = 'array',

        tickvals = [2010, 2011, 2012, 2013, 2014, 

                    2015, 2016, 2017 , 2018, 2019, 2020],

        ticktext = ['2010', '2011', '2012', '2013', 

                    '2014', '2015', '2016', '2017' , 

                    '2018', '2019', '2020'],),

    paper_bgcolor='aquamarine',

    plot_bgcolor = "aquamarine",

    

    )

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
dfd = df.groupby(['Year'])['StateRegistrationOfDivorce'].sum().reset_index()

#dfd = dfd.loc[dfd['Year'] != 2020]

sns.set_context("talk")

plt.style.use("fivethirtyeight")

plt.figure(figsize = (14,10))



ax = sns.barplot(x = 'Year', y = 'StateRegistrationOfDivorce', data = dfd, ci=None, palette = 'Accent', edgecolor = 'black')

plt.title("Divorces in Moscow over the years", pad =20)

plt.ylabel('Divorces')

for p in ax.patches:

             ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),

                 textcoords='offset points')
fig = go.Figure(data=go.Heatmap(

                   z= df['StateRegistrationOfDivorce'],

                   x=df['Month'],

                   y= df['Year'],

                   hoverongaps = False))

fig.update_layout(

    title_text= '<b> Divorces in Moscow<b>',title_x=0.5,

    yaxis = dict(

        tickmode = 'array',

        tickvals = [2010, 2011, 2012, 2013, 2014, 

                    2015, 2016, 2017 , 2018, 2019, 2020],

        ticktext = ['2010', '2011', '2012', '2013', 

                    '2014', '2015', '2016', '2017' , 

                    '2018', '2019', '2020'],),

    paper_bgcolor='aquamarine',

    plot_bgcolor = "aquamarine",

    

    )

fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))

fig.show()
df6 = df.groupby(['Year'])['StateRegistrationOfAdoption'].sum().reset_index()

df6 = df6.loc[df6['Year'] != 2020]

sns.set_context("talk")

plt.style.use("fivethirtyeight")

plt.figure(figsize = (14,10))

ax = sns.barplot(x = 'Year' , y = 'StateRegistrationOfAdoption', data = df6, ci = None, palette = 'dark')

plt.title('Adoptions in Moscow over the years', pad =20)

plt.ylabel('Adoptions')



for p in ax.patches:

             ax.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),

                 textcoords='offset points')
df7 = df.groupby(['Year'])['StateRegistrationOfNameChange'].sum().reset_index()

df7 = df7.loc[df7['Year'] != 2020]

sns.set_context("talk")

plt.style.use("fivethirtyeight")

plt.figure(figsize = (14,10))

ax = sns.barplot(x = 'Year' , y = 'StateRegistrationOfNameChange', data = df7, ci = None, palette = 'dark')

plt.title('Name Changes in Moscow over the years', pad =20)

plt.ylabel('Name Changes')

for p in ax.patches:

             ax.annotate( "%.f" %p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),

                 textcoords='offset points')



df8 = df.groupby(['Year'])['StateRegistrationOfPaternityExamination'].sum().reset_index()

df8 = df8.loc[df8['Year'] != 2020]

sns.set_context("talk")

plt.style.use("fivethirtyeight")

plt.figure(figsize = (14,10))

ax = sns.barplot(x = 'Year' , y = 'StateRegistrationOfPaternityExamination', data = df8, ci = None, palette = 'dark')

plt.title('Paternity Examinations in Moscow over the years', pad = 20)

plt.ylabel('No. of parenity examination')

for p in ax.patches:

             ax.annotate( "%.f" %p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=15, color='black', xytext=(0, 8),

                 textcoords='offset points')