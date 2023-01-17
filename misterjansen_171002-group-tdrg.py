

# import additional packages



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

import plotly.graph_objs as go

import seaborn as sns

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,30,32,34,36,38,39,40,42,44,46,47,48,50,52,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,80,81,83,85,87,89,91,93,95,97,98,100,101,102,103,104,105,107,108,109,110,111,112,113,114,115,116,117,118,119,120,122,124,125,126,127,128,130,131,132,133,134])

data.head()
# Number of observations per column

data.count()



# et cetera...
# Group data per year for the world

terror_peryear_world = np.asarray(data.groupby('iyear').iyear.count())

terror_years = np.arange(1970, 2016)



# Explore in table iyear and years (is het mogelijk did in één tabel te krijgen??)

print(terror_peryear)

print(terror_years)
# create distinct lines for the different regions

data_north_america = data[(data.region == 1) |  (data.region == 2)]

data_asia = data[(data.region == 4) | (data.region == 5) | (data.region == 6) | (data.region == 7)]

data_oceania = data[(data.region == 12)]

data_europe = data[(data.region == 8) | (data.region == 9)]

data_south_america = data[(data.region == 3)]

data_middle_east_n_africa = data[(data.region == 10)]

data_sub_africa = data[(data.region == 11)]



peryear_north_america = np.asarray(data_north_america.groupby('iyear').iyear.count())

peryear_asia = np.asarray(data_asia.groupby('iyear').iyear.count())

peryear_oceania = np.asarray(data_oceania.groupby('iyear').iyear.count())

peryear_europe = np.asarray(data_europe.groupby('iyear').iyear.count())

peryear_south_america = np.asarray(data_south_america.groupby('iyear').iyear.count())

peryear_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('iyear').iyear.count())

peryear_sub_africa = np.asarray(data_sub_africa.groupby('iyear').iyear.count())
# Create graph for the wold



trace0 = go.Scatter(

         x = terror_years,

         y = terror_peryear_world,

         mode = 'lines',

         line = dict(

             color = 'rgb(240, 140, 45)',

             width = 3),

        name = 'World'

         )



figure = dict(data = [trace0], layout = layout)

iplot(figure)
trace1 = go.Scatter(                             

         x = terror_years,

         y = peryear_north_america,

         mode = 'lines',

         line = dict(

             color = 'rgb(140, 140, 45)',

             width = 3),

        name = 'North- and Central America '

         )

trace2 = go.Scatter(                             

         x = terror_years,

         y = peryear_asia,

         mode = 'lines',

         line = dict(

             color = 'rgb(240, 40, 45)',

             width = 3),

        name = 'Asia'

         )

trace3 = go.Scatter(                             

         x = terror_years,

         y = peryear_oceania,

         mode = 'lines',

         line = dict(

             color = 'rgb(120, 120,120)',

             width = 3),

        name = 'Oceania'

         )

trace4 = go.Scatter(                             

         x = terror_years,

         y = peryear_europe,

         mode = 'lines',

         line = dict(

             color = 'rgb(0, 50, 72)',

             width = 3),

        name = 'Europe'

         )

trace5 = go.Scatter(                             

         x = terror_years,

         y = peryear_south_america,

         mode = 'lines',

         line = dict(

             color = 'rgb(27, 135 , 78)',

             width = 3),

        name = 'South America'

         )

trace6 = go.Scatter(                             

         x = terror_years,

         y = peryear_middle_east_n_africa,

         mode = 'lines',

         line = dict(

             color = 'rgb(230, 230, 230)',

             width = 3),

        name = 'Middle East and North Africa'

         )

trace7 = go.Scatter(                             

         x = terror_years,

         y = peryear_sub_africa,

         mode = 'lines',

         line = dict(

             color = 'rgb(238, 133, 26)',

             width = 3),

        name = 'Sub Saharan Africa'

         )



layout = go.Layout(

         title = 'Terrorist Attacks by Year per region (1970-2015)',

         xaxis = dict(

             rangeslider = dict(thickness = 0.05),

             showline = True,

             showgrid = False

         ),

         yaxis = dict(

             range = [0.1, 7500],

             showline = True,

             showgrid = False)

         )



data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]



figure = dict(data = data, layout = layout)

iplot(figure)
# create arrays with the values 0 and 1



crit1_north_america = np.asarray(data_north_america.groupby('crit1').crit1.count())

crit1_asia = np.asarray(data_asia.groupby('crit1').crit1.count())

crit1_oceania = np.asarray(data_oceania.groupby('crit1').crit1.count())

crit1_europe = np.asarray(data_europe.groupby('crit1').crit1.count())

crit1_south_america = np.asarray(data_south_america.groupby('crit1').crit1.count())

crit1_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('crit1').crit1.count())

crit1_sub_africa = np.asarray(data_sub_africa.groupby('crit1').crit1.count())



crit2_north_america = np.asarray(data_north_america.groupby('crit2').crit2.count())

crit2_asia = np.asarray(data_asia.groupby('crit2').crit2.count())

crit2_oceania = np.asarray(data_oceania.groupby('crit2').crit2.count())

crit2_europe = np.asarray(data_europe.groupby('crit2').crit2.count())

crit2_south_america = np.asarray(data_south_america.groupby('crit2').crit2.count())

crit2_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('crit2').crit2.count())

crit2_sub_africa = np.asarray(data_sub_africa.groupby('crit2').crit2.count())



crit3_north_america = np.asarray(data_north_america.groupby('crit3').crit3.count())

crit3_asia = np.asarray(data_asia.groupby('crit3').crit3.count())

crit3_oceania = np.asarray(data_oceania.groupby('crit3').crit3.count())

crit3_europe = np.asarray(data_europe.groupby('crit3').crit3.count())

crit3_south_america = np.asarray(data_south_america.groupby('crit3').crit3.count())

crit3_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('crit3').crit3.count())

crit3_sub_africa = np.asarray(data_sub_africa.groupby('crit3').crit3.count())

# Create the arrays for the graphs per region per criterium



regions = ['North- and Central-America', 'Asia', 'Oceania', 'Europe', 'South-America', 'Middle-East and North-Africa', 'Sub-saharan Africa']



# Create array for yes       --> crit2 oceania = 0

crit1_yes = np.append(crit1_north_america[1], crit1_asia[1])

crit1_yes = np.append(crit1_yes,crit1_oceania[1])

crit1_yes = np.append(crit1_yes,crit1_europe[1])

crit1_yes = np.append(crit1_yes,crit1_south_america[1])

crit1_yes = np.append(crit1_yes,crit1_middle_east_n_africa[1])

crit1_yes = np.append(crit1_yes,crit1_sub_africa[1])

crit2_yes = np.append(crit2_north_america[1], crit2_asia[1])

crit2_yes = np.append(crit2_yes, 0)

crit2_yes = np.append(crit2_yes,crit2_europe[1])

crit2_yes = np.append(crit2_yes,crit2_south_america[1])

crit2_yes = np.append(crit2_yes,crit2_middle_east_n_africa[1])

crit2_yes = np.append(crit2_yes,crit2_sub_africa[1])

crit3_yes = np.append(crit3_north_america[1], crit3_asia[1])

crit3_yes = np.append(crit3_yes,crit3_oceania[1])

crit3_yes = np.append(crit3_yes,crit3_europe[1])

crit3_yes = np.append(crit3_yes,crit3_south_america[1])

crit3_yes = np.append(crit3_yes,crit3_middle_east_n_africa[1])

crit3_yes = np.append(crit3_yes,crit3_sub_africa[1])



# Create array for no

crit1_no = np.append(crit1_north_america[0], crit1_asia[0])

crit1_no = np.append(crit1_no,crit1_oceania[0])

crit1_no = np.append(crit1_no,crit1_europe[0])

crit1_no = np.append(crit1_no,crit1_south_america[0])

crit1_no = np.append(crit1_no,crit1_middle_east_n_africa[0])

crit1_no = np.append(crit1_no,crit1_sub_africa[0])

crit2_no = np.append(crit2_north_america[0], crit2_asia[0])

crit2_no = np.append(crit2_no, 0)

crit2_no = np.append(crit2_no,crit2_europe[0])

crit2_no = np.append(crit2_yes,crit2_south_america[0])

crit2_no = np.append(crit2_no,crit2_middle_east_n_africa[0])

crit2_no = np.append(crit2_no,crit2_sub_africa[0])

crit3_no = np.append(crit3_north_america[0], crit3_asia[0])

crit3_no = np.append(crit3_no,crit3_oceania[0])

crit3_no = np.append(crit3_no,crit3_europe[0])

crit3_no = np.append(crit3_no,crit3_south_america[0])

crit3_no = np.append(crit3_no,crit3_middle_east_n_africa[0])

crit3_no = np.append(crit3_no,crit3_sub_africa[0])



# create total observations

total_north_america = sum(crit1_north_america)

total_asia = sum(crit1_asia)

total_oceania = sum(crit1_oceania)

total_europe = sum(crit1_europe)

total_south_america = sum(crit1_south_america)

total_middle_east_n_africa = sum(crit1_middle_east_n_africa)

total_sub_africa = sum(crit1_sub_africa)



total_obs_region = np.append(total_north_america, total_asia)

total_obs_region = np.append(total_obs_region, total_oceania)

total_obs_region = np.append(total_obs_region, total_europe)

total_obs_region = np.append(total_obs_region, total_south_america)

total_obs_region = np.append(total_obs_region, total_middle_east_n_africa)

total_obs_region = np.append(total_obs_region, total_sub_africa)
# Create bar chart with absolute values



trace0 = go.Bar(

    x= regions,

    y= total_obs_region,

    name = 'Observations in region'

)

trace1 = go.Bar(

    x= regions,

    y= crit1_yes,

    name = 'Political, economic, religious, or social'

)

trace2 = go.Bar(

    x= regions,

    y= crit2_yes,

    name = 'Coerce, intimidate, or publicize'

)

trace3 = go.Bar(

    x= regions,

    y= crit3_yes,

    name = 'Outside international humatarian law'

)



layout = go.Layout(

    title = 'Absolute accordance to the 3 criteriums',

    barmode='group'

)



data = [trace0, trace1, trace2, trace3]



figure = dict(data = data, layout = layout)

iplot(figure)
# create relative = crit 1 / total



rel_crit1_yes = crit1_yes / total_obs_region

rel_crit2_yes = crit2_yes / total_obs_region

rel_crit3_yes = crit3_yes / total_obs_region



trace1 = go.Bar(

    x= regions,

    y= rel_crit1_yes,

    name = 'Political, economic, religious, or social'

)

trace2 = go.Bar(

    x= regions,

    y= rel_crit2_yes,

    name = 'Coerce, intimidate, or publicize'

)

trace3 = go.Bar(

    x= regions,

    y= rel_crit3_yes,

    name = 'Outside international humatarian law'

)



layout = go.Layout(

    title = 'relative accordance to the 3 criteriums',

    barmode='group'

)



data = [trace1, trace2, trace3]



figure = dict(data = data, layout = layout)

iplot(figure)
count_year = data.groupby(['iyear']).count()

death_year = data.groupby(['iyear']).mean()



f1 = plt.figure()

ax1 = f1.add_subplot(211)

ax1.plot(count_year.index, count_year.nkill)

ax1.set(title='Total fatalities over time',xlabel='Year',ylabel='Fatalities')



f2 = plt.figure()

ax2 = f2.add_subplot(212)

ax2.plot(death_year.index, death_year.nkill)

ax2.set(title='Average fatalities per terrorist attack',xlabel='Year',ylabel='Fatalities')



plt.show()
# Attack type

data['count'] = 1

by_year = (data.groupby('iyear').agg({'count':'sum'}))

attack_type = data.groupby('attacktype1')['count'].count().reset_index()

total = attack_type['count'].sum()

attack_type['Percentage'] = attack_type.apply(lambda x : (x['count']/total) * 100, axis=1)



plt.figure(figsize=[16,8])

sns.pointplot(x='attacktype1', y='Percentage', data=attack_type, color='yellow', rotation=30)

plt.xlabel('Attack Type', size=16)

plt.ylabel('Percentage', size=16)



data_matrix = [['Number', 'Attack_Type']

               ['1', 'Assassination'],

               ['2', 'Armed_Assault'],

               ['3', 'Bombing/Explosion'],

               ['4', 'Hijacking'],

               ['5', 'Hostage_Taking(barricade_Incident)'],

               ['6', 'Hostage_Taking(Kidnapping)'],

               ['7', 'Facility/Infrastructure_Attack'],

               ['8', 'Unarmed_Assault'],

               ['9', 'Unknown']]

               

table = ff.create_table(data_matrix)

py.iplot(table, filename='simple_table')