import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium import plugins

from folium.plugins import HeatMap



import plotly

import plotly.plotly as py

import cufflinks as cf

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()



import os

print(os.listdir("../input"))



df_population = pd.read_csv('../input/population.csv')

df_population.info()
df_population.head(5)
yearPOPULATION_count = df_population.groupby('Year')['Number'].sum()

trace = go.Scatter(

    x = yearPOPULATION_count.index,

    y = yearPOPULATION_count.values,

    mode = 'lines+markers',  

    marker = dict(color = '#CE0000')

)



data = [trace]



layout = go.Layout(title="Population Change From 2013 To 2017")



fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
man = df_population.loc[df_population['Gender'] == 'Male'].groupby(['Year'])['Number'].sum()

woman = df_population.loc[df_population['Gender'] == 'Female'].groupby(['Year'])['Number'].sum()

population_change = yearPOPULATION_count.values[4]-yearPOPULATION_count.values[0]

print("Since 2013, Barcelona's Population changed by:",population_change, 'Citizens' )

female_change = woman.values[4]-woman.values[0]

male_change = man.values[4]-man.values[0]

print("Out of these",population_change, 'People' )

print('There are',female_change,'female And',male_change,'male citizens')



trace_male_change = go.Bar(x = ['Male'],

                y= [male_change],

                name = "Males",

                marker = dict(color = '#000063')

               )



trace_female_change = go.Bar(x = ['Female'],

                y = [female_change],

                name = "Females",

                marker = dict(color = '#CE0000')

               )



data_change = [trace_male_change,trace_female_change]

layout_change = go.Layout(barmode = 'stack',

                   title="Male versus Female Population Increase since 2013",

                      )



fig_change = go.Figure(data=data_change,layout=layout_change)



plotly.offline.iplot(fig_change)
trace_male = go.Bar(x = man.index,

                y= man.values,

                name = "Males",

                marker = dict(color = '#000063')

               )



trace_female = go.Bar(x = woman.index,

                y = woman.values,

                name = "Females",

                marker = dict(color = '#CE0000')

               )



data_fm = [trace_male,trace_female]

layout_fm = go.Layout(barmode = 'group',

                   title="Number of Male versus Number of Female",

                      )



fig_fm = go.Figure(data=data_fm,layout=layout_fm)



plotly.offline.iplot(fig_fm)
df_unemp = pd.read_csv('../input/unemployment.csv')

df_unemp.head(6)
year2017UNEMPLOY_count = df_unemp.loc[df_unemp['Year']==2017].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2016UNEMPLOY_count = df_unemp.loc[df_unemp['Year']==2016].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2015UNEMPLOY_count = df_unemp.loc[df_unemp['Year']==2015].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2014UNEMPLOY_count = df_unemp.loc[df_unemp['Year']==2014].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2013UNEMPLOY_count = df_unemp.loc[df_unemp['Year']==2013].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()





trace_unemp_2013 = go.Bar(x = ['2013'],

                y= [int(year2013UNEMPLOY_count)],

                name = "2013",

                marker = dict(color = '#CE0000')

               )

trace_unemp_2014 = go.Bar(x = ['2014'],

                y= [int(year2014UNEMPLOY_count)],

                name = "2014",

                marker = dict(color = '#000063')

               )

trace_unemp_2015 = go.Bar(x = ['2015'],

                y= [int(year2015UNEMPLOY_count)],

                name = "2015",

                marker = dict(color = '#5A79A5')

               )

trace_unemp_2016 = go.Bar(x = ['2016'],

                y= [int(year2016UNEMPLOY_count)],

                name = "2016",

                marker = dict(color = '#9CAAC6')

               )

trace_unemp_2017 = go.Bar(x = ['2017'],

                y= [int(year2017UNEMPLOY_count)],

                name = "2017",

                marker = dict(color = 'grey')

               )





data_unemp = [trace_unemp_2013,trace_unemp_2014,trace_unemp_2015,trace_unemp_2016,trace_unemp_2017]

layout_unemp = go.Layout(barmode = 'stack',

                   title="Mean number of Registered unemployed across years",

                      )



fig_unemp = go.Figure(data=data_unemp,layout=layout_unemp)



plotly.offline.iplot(fig_unemp)
year2017UNEMPLOY_Male = df_unemp.loc[df_unemp['Gender']=='Male'].loc[df_unemp['Year']==2017].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2016UNEMPLOY_Male = df_unemp.loc[df_unemp['Gender']=='Male'].loc[df_unemp['Year']==2016].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2015UNEMPLOY_Male = df_unemp.loc[df_unemp['Gender']=='Male'].loc[df_unemp['Year']==2015].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2014UNEMPLOY_Male = df_unemp.loc[df_unemp['Gender']=='Male'].loc[df_unemp['Year']==2014].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2013UNEMPLOY_Male = df_unemp.loc[df_unemp['Gender']=='Male'].loc[df_unemp['Year']==2013].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()





year2017UNEMPLOY_Female = df_unemp.loc[df_unemp['Gender']=='Female'].loc[df_unemp['Year']==2017].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2016UNEMPLOY_Female = df_unemp.loc[df_unemp['Gender']=='Female'].loc[df_unemp['Year']==2016].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2015UNEMPLOY_Female = df_unemp.loc[df_unemp['Gender']=='Female'].loc[df_unemp['Year']==2015].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2014UNEMPLOY_Female = df_unemp.loc[df_unemp['Gender']=='Female'].loc[df_unemp['Year']==2014].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



year2013UNEMPLOY_Female = df_unemp.loc[df_unemp['Gender']=='Female'].loc[df_unemp['Year']==2013].loc[df_unemp['Demand_occupation']=='Registered unemployed'].groupby('Month')['Number'].sum().mean()



trace_male2013 = go.Bar(x = ['2013'],

                y= [int(year2013UNEMPLOY_Male)],

                name = "Male (2013)",

                marker = dict(color = '#000063')        

               )



trace_male2014 = go.Bar(x = ['2014'],

                y= [int(year2014UNEMPLOY_Male)],

                name = "Male (2014)",

                marker = dict(color = '#29264a')

               )



trace_male2015 = go.Bar(x = ['2015'],

                y= [int(year2015UNEMPLOY_Male)],

                name = "Male (2015)",

                marker = dict(color = '#5A79A5')

               )



trace_male2016 = go.Bar(x = ['2016'],

                y= [int(year2016UNEMPLOY_Male)],

                name = "Male (2016)",

                marker = dict(color = '#9CAAC6')

               )



trace_male2017 = go.Bar(x = ['2017'],

                y= [int(year2017UNEMPLOY_Male)],

                name = "Male (2017)",

                marker = dict(color = 'grey')

               )



trace_female2013 = go.Bar(x = ['2013'],

                y= [int(year2013UNEMPLOY_Female)],

                name = "Female (2013)",

                marker = dict(color = '#CE0000')

               )



trace_female2014 = go.Bar(x = ['2014'],

                y= [int(year2014UNEMPLOY_Female)],

                name = "Female (2014)",

                marker = dict(color = '#c4352a')

               )

trace_female2015 = go.Bar(x = ['2015'],

                y= [int(year2015UNEMPLOY_Female)],

                name = "Female (2015)",

                marker = dict(color = '#b84c4a')

               )



trace_female2016 = go.Bar(x = ['2016'],

                y= [int(year2016UNEMPLOY_Female)],

                name = "Female (2016)",

                marker = dict(color = '#a85e6a')

               )



trace_female2017 = go.Bar(x = ['2017'],

                y= [int(year2017UNEMPLOY_Female)],

                name = "Female (2017)",

                marker = dict(color = '#f7957d')

               )







data_unemprt = [trace_male2013,trace_male2014,trace_male2015,trace_male2016,trace_male2017,trace_female2013,

          trace_female2014,trace_female2015,trace_female2016,trace_female2017]



layout_unemprt = go.Layout(barmode = 'stack',

                   title="Mean Number Registered Unemployed Among Gender From Each Year",

                      )



fig_unemprt = go.Figure(data=data_unemprt,layout=layout_unemprt)



plotly.offline.iplot(fig_unemprt)
year2017DEEMPLOY_count = df_unemp.loc[df_unemp['Year']==2017].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2016DEEMPLOY_count = df_unemp.loc[df_unemp['Year']==2016].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2015DEEMPLOY_count = df_unemp.loc[df_unemp['Year']==2015].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2014DEEMPLOY_count = df_unemp.loc[df_unemp['Year']==2014].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2013DEEMPLOY_count = df_unemp.loc[df_unemp['Year']==2013].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()





def unemploy_demand(val):

    if pd.isnull(val) == True:

        val = 0

    return val

               

trace_DEemp_2015 = go.Bar(x = ['Year 2015'],

                y= [int(unemploy_demand(year2015DEEMPLOY_count))],

                name = "2015",

                marker = dict(color = '#000063')

               )

trace_DEemp_2016 = go.Bar(x = ['Year 2016'],

                y= [int(unemploy_demand(year2016DEEMPLOY_count))],

                name = "2016",

                marker = dict(color = '#5A79A5')

               )

trace_DEemp_2017 = go.Bar(x = ['Year 2017'],

                y= [int(unemploy_demand(year2017DEEMPLOY_count))],

                name = "2017",

                marker = dict(color = '#9CAAC6')

               )





data_DEemp = [trace_DEemp_2015,trace_DEemp_2016,trace_DEemp_2017]

layout_DEemp = go.Layout(barmode = 'group',

                   title="Mean number of Unemployment demand across years",

                      )



fig_DEemp = go.Figure(data=data_DEemp,layout=layout_DEemp)



plotly.offline.iplot(fig_DEemp)



year2017DEEMPLOY_male = df_unemp.loc[df_unemp['Gender']=='Male'].loc[df_unemp['Year']==2017].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2016DEEMPLOY_male = df_unemp.loc[df_unemp['Gender']=='Male'].loc[df_unemp['Year']==2016].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2015DEEMPLOY_male = df_unemp.loc[df_unemp['Gender']=='Male'].loc[df_unemp['Year']==2015].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2014DEEMPLOY_male = df_unemp.loc[df_unemp['Gender']=='Male'].loc[df_unemp['Year']==2014].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2013DEEMPLOY_male = df_unemp.loc[df_unemp['Gender']=='Male'].loc[df_unemp['Year']==2013].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()





year2017DEEMPLOY_Female = df_unemp.loc[df_unemp['Gender']=='Female'].loc[df_unemp['Year']==2017].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2016DEEMPLOY_Female = df_unemp.loc[df_unemp['Gender']=='Female'].loc[df_unemp['Year']==2016].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2015DEEMPLOY_Female = df_unemp.loc[df_unemp['Gender']=='Female'].loc[df_unemp['Year']==2015].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2014DEEMPLOY_Female = df_unemp.loc[df_unemp['Gender']=='Female'].loc[df_unemp['Year']==2014].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()



year2013DEEMPLOY_Female = df_unemp.loc[df_unemp['Gender']=='Female'].loc[df_unemp['Year']==2013].loc[df_unemp['Demand_occupation']=='Unemployment demand'].groupby('Month')['Number'].sum().mean()





trace_male2015D = go.Bar(x = ['2015'],

                y= [int(year2015DEEMPLOY_male)],

                name = "Male (2015)",

                marker = dict(color = '#000063')

               )



trace_male2016D = go.Bar(x = ['2016'],

                y= [int(year2016DEEMPLOY_male)],

                name = "Male (2016)",

                marker = dict(color = '#000063')

               )



trace_male2017D = go.Bar(x = ['2017'],

                y= [int(year2017DEEMPLOY_male)],

                name = "Male (2017)",

                marker = dict(color = '#000063')

               )



trace_female2015D = go.Bar(x = ['2015'],

                y= [int(year2015DEEMPLOY_Female)],

                name = "Female (2015)",

                marker = dict(color = '#CE0000')

               )



trace_female2016D = go.Bar(x = ['2016'],

                y= [int(year2016DEEMPLOY_Female)],

                name = "Female (2016)",

                marker = dict(color = '#CE0000')

               )



trace_female2017D = go.Bar(x = ['2017'],

                y= [int(year2017DEEMPLOY_Female)],

                name = "Female (2017)",

                marker = dict(color = '#CE0000')

               )







data_Dunemprt = [trace_male2015D,trace_male2016D,trace_male2017D,trace_female2015D,

                trace_female2016D,trace_female2017D]



layout_Dunemprt = go.Layout(barmode = 'stack',

                   title="Mean Number Unemployment demand Among Gender From Each Year",

                      )



fig_Dunemprt = go.Figure(data=data_Dunemprt,layout=layout_Dunemprt)



plotly.offline.iplot(fig_Dunemprt)
df_lifexp = pd.read_csv('../input/life_expectancy.csv')

df_lifexp.head()
df_lifexp.dropna(inplace=True)


lifexp2010female = df_lifexp['2006-2010'].loc[df_lifexp['Gender']=='Female'].mean()

lifexp2010male = df_lifexp['2006-2010'].loc[df_lifexp['Gender']=='Male'].mean()



lifexp2011female = df_lifexp['2007-2011'].loc[df_lifexp['Gender']=='Female'].mean()

lifexp2011male = df_lifexp['2007-2011'].loc[df_lifexp['Gender']=='Male'].mean()



lifexp2012female = df_lifexp['2008-2012'].loc[df_lifexp['Gender']=='Female'].mean()

lifexp2012male = df_lifexp['2008-2012'].loc[df_lifexp['Gender']=='Male'].mean()



lifexp2013female = df_lifexp['2009-2013'].loc[df_lifexp['Gender']=='Female'].mean()

lifexp2013male = df_lifexp['2009-2013'].loc[df_lifexp['Gender']=='Male'].mean()



lifexp2014female = df_lifexp['2010-2014'].loc[df_lifexp['Gender']=='Female'].mean()

lifexp2014male = df_lifexp['2010-2014'].loc[df_lifexp['Gender']=='Male'].mean()



trace_lifeexp2010male = go.Bar(x = ['2006-2010'],

                y= [lifexp2010male],

                name = "Male Life Expectancy (2006-2010)",

                marker = dict(color = '#000063')      

               )



trace_lifeexp2010female = go.Bar(x = ['2006-2010'],

                y= [lifexp2010female],

                name = "Female Life Expectancy (2006-2010)",

                marker = dict(color = '#CE0000')      

               )





trace_lifeexp2011male = go.Bar(x = ['2007-2011'],

                y= [lifexp2011male],

                name = "Male Life Expectancy (2007-2011)",

                marker = dict(color = '#29264a')

               )



trace_lifeexp2011female = go.Bar(x = ['2007-2011'],

                y= [lifexp2011female],

                name = "Female Life Expectancy (2007-2011)",

                marker = dict(color = '#c4352a')

               )





trace_lifeexp2012male = go.Bar(x = ['2008-2012'],

                y= [lifexp2012male],

                name = "Male Life Expectancy (2008-2012)",

                marker = dict(color = '#5A79A5')

                

               )



trace_lifeexp2012female = go.Bar(x = ['2008-2012'],

                y= [lifexp2012female],

                name = "Female Life Expectancy (2008-2012)",

                marker = dict(color = '#b84c4a')

               )





trace_lifeexp2013male = go.Bar(x = ['2009-2013'],

                y= [lifexp2013male],

                name = "Male Life Expectancy (2009-2013)",

                marker = dict(color = '#9CAAC6')

               )



trace_lifeexp2013female = go.Bar(x = ['2009-2013'],

                y= [lifexp2013female],

                name = "Female Life Expectancy (2009-2013)",

                marker = dict(color = '#a85e6a')

               )





trace_lifeexp2014male = go.Bar(x = ['2010-2014'],

                y= [lifexp2014male],

                name = "Male Life Expectancy (2010-2014)",

                marker = dict(color = 'grey')

               )



trace_lifeexp2014female = go.Bar(x = ['2010-2014'],

                y= [lifexp2014female],

                name = "Female Life Expectancy (2010-2014)",

                marker = dict(color = '#f7957d')

               )



data_lifeexp = [trace_lifeexp2010male,trace_lifeexp2010female,trace_lifeexp2011male,trace_lifeexp2011female,

                trace_lifeexp2012male,trace_lifeexp2012female,trace_lifeexp2013male,trace_lifeexp2013female,

                trace_lifeexp2014male,trace_lifeexp2014female]



layout_lifeexp = go.Layout(barmode = 'group',

                   title="Mean Life Expectancy Among Genders Across Years",

                      )



fig_lifeexp = go.Figure(data=data_lifeexp,layout=layout_lifeexp)



plotly.offline.iplot(fig_lifeexp)
df_birth = pd.read_csv('../input/births.csv')

df_death = pd.read_csv('../input/deaths.csv')
df_birth.head(5)
birth_boys_2017 = df_birth.loc[df_birth["Gender"]=="Boys"].loc[df_birth["Year"]==2017].groupby("Neighborhood Code")['Number'].sum().sum()

birth_girls_2017 = df_birth.loc[df_birth["Gender"]=="Girls"].loc[df_birth["Year"]==2017].groupby("Neighborhood Code")['Number'].sum().sum()



birth_boys_2016 = df_birth.loc[df_birth["Gender"]=="Boys"].loc[df_birth["Year"]==2016].groupby("Neighborhood Code")['Number'].sum().sum()

birth_girls_2016 = df_birth.loc[df_birth["Gender"]=="Girls"].loc[df_birth["Year"]==2016].groupby("Neighborhood Code")['Number'].sum().sum()



birth_boys_2015 = df_birth.loc[df_birth["Gender"]=="Boys"].loc[df_birth["Year"]==2015].groupby("Neighborhood Code")['Number'].sum().sum()

birth_girls_2015 = df_birth.loc[df_birth["Gender"]=="Girls"].loc[df_birth["Year"]==2015].groupby("Neighborhood Code")['Number'].sum().sum()



birth_boys_2014 = df_birth.loc[df_birth["Gender"]=="Boys"].loc[df_birth["Year"]==2014].groupby("Neighborhood Code")['Number'].sum().sum()

birth_girls_2014 = df_birth.loc[df_birth["Gender"]=="Girls"].loc[df_birth["Year"]==2014].groupby("Neighborhood Code")['Number'].sum().sum()



birth_boys_2013 = df_birth.loc[df_birth["Gender"]=="Boys"].loc[df_birth["Year"]==2013].groupby("Neighborhood Code")['Number'].sum().sum()

birth_girls_2013 = df_birth.loc[df_birth["Gender"]=="Girls"].loc[df_birth["Year"]==2013].groupby("Neighborhood Code")['Number'].sum().sum()













trace_birth2013male = go.Bar(x = ['2013'],

                y= [birth_boys_2013],

                name = "Male births 2013",

                marker = dict(color = '#000063')      

               )



trace_birth2013female = go.Bar(x = ['2013'],

                y= [birth_girls_2013],

                name = "Female births 2013",

                marker = dict(color = '#CE0000')      

               )





trace_birth2014male = go.Bar(x = ['2014'],

                y= [birth_boys_2014],

                name = "Male births 2014",

                marker = dict(color = '#29264a')

               )



trace_birth2014female = go.Bar(x = ['2014'],

                y= [birth_girls_2014],

                name = "Female births 2014",

                marker = dict(color = '#c4352a')

               )





trace_birth2015male = go.Bar(x = ['2015'],

                y= [birth_boys_2015],

                name = "Male births 2015",

                marker = dict(color = '#5A79A5')

                

               )



trace_birth2015female = go.Bar(x = ['2015'],

                y= [birth_girls_2015],

                name = "Female births 2015",

                marker = dict(color = '#b84c4a')

               )





trace_birth2016male = go.Bar(x = ['2016'],

                y= [birth_boys_2016],

                name = "Male births 2016",

                marker = dict(color = '#9CAAC6')

               )



trace_birth2016female = go.Bar(x = ['2016'],

                y= [birth_girls_2016],

                name = "Female births 2016",

                marker = dict(color = '#a85e6a')

               )





trace_birth2017male = go.Bar(x = ['2017'],

                y= [birth_girls_2017],

                name = "Male births 2017",

                marker = dict(color = 'grey')

               )



trace_birth2017female = go.Bar(x = ['2017'],

                y= [birth_girls_2017],

                name = "Female births 2017",

                marker = dict(color = '#f7957d')

               )



data_births = [trace_birth2013male,trace_birth2013female,trace_birth2014male,trace_birth2014female,trace_birth2015male,trace_birth2015female,

               trace_birth2016male,trace_birth2016female,trace_birth2017male,trace_birth2017female]



layout_births = go.Layout(barmode = 'stack',

                   title="Births Across Years",

                      )



fig_births = go.Figure(data=data_births,layout=layout_births)



plotly.offline.iplot(fig_births)

df_death.head(5)
year2017death = df_death.loc[df_death['Year']==2017].groupby('Age')['Number'].sum().sort_values()

year2016death = df_death.loc[df_death['Year']==2016].groupby('Age')['Number'].sum().sort_values()

year2015death = df_death.loc[df_death['Year']==2015].groupby('Age')['Number'].sum().sort_values()





trace_death2017 = go.Bar(x = year2017death.index,

                y=  year2017death.values,

                name = "2017",

                marker = dict(color = '#CE0000')

               )



trace_death2016 = go.Bar(x = year2016death.index,

                y=  year2016death.values,

                name = "2017",

                marker = dict(color = '#000063')

               )



trace_death2015 = go.Bar(x = year2015death.index,

                y=  year2015death.values,

                name = "2017",

                marker = dict(color = '#5A79A5')

               )



data_deaths = [trace_death2017,trace_death2016,trace_death2015]



layout_deaths = go.Layout(barmode = 'group',

                   title="Deaths In Age Categories",

                      )



fig_deaths = go.Figure(data=data_deaths,layout=layout_deaths)



plotly.offline.iplot(fig_deaths)

total2017death = df_death.loc[df_death['Year']==2015].groupby('Age')['Number'].sum().sum()

total2016death = df_death.loc[df_death['Year']==2016].groupby('Age')['Number'].sum().sum()

total2015death = df_death.loc[df_death['Year']==2015].groupby('Age')['Number'].sum().sum()



total_death = total2017death + total2016death + total2015death
trace_dead2017 = go.Bar(x = ['Year 2017'],

                y =  [total2017death],

                name = "2017",

                marker = dict(color = '#CE0000')

               )



trace_dead2016 = go.Bar(x = ["Year 2016"],

                y= [total2016death],

                name = "2016",

                marker = dict(color = '#000063')

               )



trace_dead2015 = go.Bar(x = ["Year 2015"],

                y=  [total2015death],

                name = "2015",

                marker = dict(color = '#5A79A5')

               )



trace_dead = go.Bar(x = ["Deaths Total"],

                y=  [total_death],

                name = "Total",

                marker = dict(color = 'grey')

               )





data_dead = [trace_dead2015,trace_dead2016,trace_dead2017,trace_dead]



layout_dead = go.Layout(barmode = 'stack',

                   title="Total Number of Deaths",

                      )



fig_dead = go.Figure(data=data_dead,layout=layout_dead)



plotly.offline.iplot(fig_dead)

df_acc = pd.read_csv("../input/accidents_2017.csv")
df_acc.head(10)
victims = df_acc["Victims"].sum()

mild_inj = df_acc["Mild injuries"].sum()

seri_inj = df_acc["Serious injuries"].sum()

vehi = df_acc["Vehicles involved"].sum()



unknown = victims - (mild_inj+seri_inj)



print('Total Number of vehicles involved:',vehi)

print('Total Number of Victims:',victims)

print('Total Number of Victims with mild injuries:',mild_inj)

print('Total Number of Victims with serious injuries:',seri_inj)

print('Total Number of Victims with unknown information',unknown)





coordinates = [41.406141, 2.168594]





map_acc = folium.Map(location=coordinates,

                    zoom_start = 13)



df_cor = df_acc[['Latitude','Longitude']]

cor = [[row['Latitude'],row['Longitude']] for index,row in df_cor.iterrows()]



HeatMap(cor, min_opacity=0.5, radius=14).add_to(map_acc)

map_acc







df_transport = pd.read_csv("../input/transports.csv")

df_transport.head(100)

df_transport.drop('District.Name',axis=1, inplace=True)

df_transport.drop('Neighborhood.Name',axis=1, inplace=True)
underground = df_transport.loc[df_transport["Transport"]=="Underground"]


under_rounded = underground.round(3)

under_rounded.drop_duplicates(["Longitude","Latitude"],keep=False,inplace=True)

under_rounded.drop_duplicates("Station",keep=False,inplace=True)

under_rounded



## This was very tricky as according to official information there are about 141 metro stations in Barcelona. However there are about 463 records after 

## filtering dataframe with transport type. Many of markers on map are placed in almost same position. So I decided to round Lat and Long coordinates and 

## drop all duplicates. With that I sacrificed accuracy on the map for the lack of unnecessary station markers. Still there are some unnessesary markers.





lat = under_rounded['Latitude']

lon = under_rounded['Longitude']

station = under_rounded["Station"]











data = pd.DataFrame({

'lat':lat,

'lon':lon,

'name':station

})

 

# Make an empty map

map_underground = folium.Map(location=coordinates, tiles="Stamen Toner", zoom_start=13)

 

# I can add marker one by one on the map

for i in range(0,len(data)):

    folium.Marker([data.iloc[i]['lat'], data.iloc[i]['lon']], popup=data.iloc[i]['name']).add_to(map_underground)

 



map_underground