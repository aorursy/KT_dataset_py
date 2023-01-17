import pandas as pd #linear algebra

import numpy as np



# for Box-Cox Transformation

from scipy import stats



# for min_max scaling

from mlxtend.preprocessing import minmax_scaling



# plotting modules

import seaborn as sns

import matplotlib.pyplot as plt



# set seed for reproducibility

np.random.seed(0)





from scipy.stats import norm

import plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.tools import FigureFactory as ff

from wordcloud import WordCloud,STOPWORDS

from PIL import Image

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')

data.head()
Temp_data=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv')



Temp_data['dt']=pd.to_datetime(Temp_data.dt)

Temp_data['year']=Temp_data['dt'].map(lambda x: x.year)



#Calculating average year temperature

year_avg=[]

for i in range(1750,2014):

    year_avg.append(Temp_data[Temp_data['year']==i]['LandAverageTemperature'].mean())





years=range(1750,2014)



#calculating 5 years average temperatures

fiveyear=[]

for i in range(1755,2019):

    a=[]

    for j in range(i-5,i):

        a.append(Temp_data[Temp_data['year']==(j-5)]['LandAverageTemperature'].mean())

    fiveyear.append(sum(a)/float(len(a)))



#for plotting

np_year_avg=np.array(year_avg)

np_fiveyear_avg=np.array(fiveyear)

#plotting graphs



plt.figure(figsize=(10,8))

plt.grid()

plt.plot(years,np_fiveyear_avg,'b',label='Five year average temperature')

plt.plot(years,np_year_avg,'g',label='Annual average temperature')

plt.legend(loc='upper left')

plt.title('Global Average Land Temperature (1750-2014)')

plt.xlabel('Years')

plt.ylabel('Temperature')

plt.show()

productivity = pd.read_csv('../input/oecd-productivity-data/level_of_gdp_per_capita_and_productivity.csv')

productivity_a = productivity.loc[productivity['Unit'] == "National currency"]

productivity_b = productivity_a.loc[productivity_a['Subject'] == "Gross Domestic Product (GDP); millions"]

productivity_c = productivity_b.loc[productivity_b['Country']== "Ireland"]

productivity_d = productivity_b.loc[productivity_b['Country']== "Switzerland"]

productivity_e = productivity_b.loc[productivity_b['Country']== "Iceland"]

productivity_f = productivity_b.loc[productivity_b['Country']== "Norway"]

productivity_l = productivity_b.loc[productivity_b['Country']== "Netherlands"]

productivity_h = productivity_b.loc[productivity_b['Country']== "Austria"]

productivity_i = productivity_b.loc[productivity_b['Country']== "Denmark"]

productivity_j = productivity_b.loc[productivity_b['Country']== "Australia"]

productivity_k = productivity_b.loc[productivity_b['Country']== "Germany"]

productivity_g = productivity_b.loc[productivity_b['Country']== "United States"]
import plotly.graph_objs as go



# creating trace1

trace1 = go.Scatter(

                    x = productivity_c.TIME,

                    y = productivity_c.Value,

                    mode = 'lines',

                    name = "Ireland",

                    marker = dict(color="darkblue"),

                    text = productivity_c.Country)

trace2 = go.Scatter(

                    x = productivity_d.TIME,

                    y = productivity_d.Value,

                    mode = 'lines',

                    name = "Switzerland",

                    marker = dict(color="darkseagreen"),

                    text = productivity_d.Country)

trace3 = go.Scatter(

                    x = productivity_e.TIME,

                    y = productivity_e.Value,

                    mode = 'lines',

                    name = "Iceland",

                    marker = dict(color="darkslateblue"),

                    text = productivity_e.Country)

trace4 = go.Scatter(

                    x = productivity_f.TIME,

                    y = productivity_f.Value,

                    mode = 'lines',

                    name = "Norway",

                    marker = dict(color="darkolivegreen"),

                    text = productivity_f.Country)

trace5 = go.Scatter(

                    x = productivity_g.TIME,

                    y = productivity_g.Value,

                    mode = 'lines',

                    name = "Netherlands",

                    marker = dict(color="darkturquoise"),

                    text = productivity_g.Country)

trace6 = go.Scatter(

                    x = productivity_h.TIME,

                    y = productivity_h.Value,

                    mode = 'lines',

                    name = "Austria",

                    marker = dict(color="navy"),

                    text = productivity_h.Country)

trace7 = go.Scatter(

                    x = productivity_i.TIME,

                    y = productivity_i.Value,

                    mode = 'lines',

                    name = "Denmark",

                    marker = dict(color="teal"),

                    text = productivity_i.Country)

trace8 = go.Scatter(

                    x = productivity_j.TIME,

                    y = productivity_j.Value,

                    mode = 'lines',

                    name = "Australia",

                    marker = dict(color="slategrey"),

                    text = productivity_j.Country)

trace9 = go.Scatter(

                    x = productivity_k.TIME,

                    y = productivity_k.Value,

                    mode = 'lines',

                    name = "Germany",

                    marker = dict(color="skyblue"),

                    text = productivity_k.Country)

trace10 = go.Scatter(

                    x = productivity_l.TIME,

                    y = productivity_l.Value,

                    mode = 'lines',

                    name = "United States",

                    marker = dict(color="royalblue"),

                    text = productivity_l.Country)



data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]

layout = dict(title="GDP(In millions) over Time for top 10 OECD Economies",

             xaxis=dict(title='Years', ticklen=100, zeroline=False)

             )

fig = dict(data=data, layout=layout)

iplot(fig)
Temp_data=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')

Temp_data= Temp_data.loc[Temp_data['Country']== "Norway"]





Temp_data['dt']=pd.to_datetime(Temp_data.dt)

Temp_data['year']=Temp_data['dt'].map(lambda x: x.year)



#Calculating average year temperature

year_avg=[]

for i in range(1750,2014):

    year_avg.append(Temp_data[Temp_data['year']==i]['AverageTemperature'].mean())





years=range(1750,2014)



#calculating 5 years average temperatures

fiveyear=[]

for i in range(1755,2019):

    a=[]

    for j in range(i-5,i):

        a.append(Temp_data[Temp_data['year']==(j-5)]['AverageTemperature'].mean())

    fiveyear.append(sum(a)/float(len(a)))



#for plotting

np_year_avg=np.array(year_avg)

np_fiveyear_avg=np.array(fiveyear)

#plotting graphs



plt.figure(figsize=(10,8))

plt.grid()

plt.plot(years,np_fiveyear_avg,'b',label='Five year average temperature')

plt.plot(years,np_year_avg,'g',label='Annual average temperature')

plt.legend(loc='upper left')

plt.title('Norway Average Land Temperature (1750-2014)')

plt.xlabel('Years')

plt.ylabel('Temperature (Celsius)')

plt.show()

productivity_x = productivity_b.loc[productivity_b['Country']== "Mexico"]

x = productivity_x.TIME



trace1 = {

  'x': x,

  'y': productivity_x.Value,

  'name': 'Economic Freedom Score',

  'type': 'bar'

};

trace11 = go.Scatter(

                    x = productivity_x.TIME,

                    y = productivity_x.Value,

                    mode = 'lines',

                    name = "GDP",

                    marker = dict(color="royalblue"),

                    text = productivity_x.Country)

data = [trace1,trace11];

layout = {

  'xaxis': {'title': ' Countries in 2016'},

  'barmode': 'relative',

  'title': 'Personal, Human, and Economic Freedom Rank For Top 10 Global Innovation Index Countries'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
Temp_data=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv')



Temp_data['dt']=pd.to_datetime(Temp_data.dt)

Temp_data['year']=Temp_data['dt'].map(lambda x: x.year)



#Calculating average year temperature

year_avg=[]

for i in range(1750,2014):

    year_avg.append(Temp_data[Temp_data['year']==i]['LandAndOceanAverageTemperature'].mean())





years=range(1750,2014)



#calculating 5 years average temperatures

fiveyear=[]

for i in range(1755,2019):

    a=[]

    for j in range(i-5,i):

        a.append(Temp_data[Temp_data['year']==(j-5)]['LandAndOceanAverageTemperature'].mean())

    fiveyear.append(sum(a)/float(len(a)))



#for plotting

np_year_avg=np.array(year_avg)

np_fiveyear_avg=np.array(fiveyear)

#plotting graphs



plt.figure(figsize=(10,8))

plt.grid()

plt.plot(years,np_fiveyear_avg,'r',label='Five year average temperature')

plt.plot(years,np_year_avg,'b',label='Annual average temperature')

plt.legend(loc='upper left')

plt.title('Global Average Land and Ocean Temperature')

plt.xlabel('Years')

plt.ylabel('Temperature')

plt.show()

import pandas as pd

import re

import numpy as np

import matplotlib.pylab as pylab

pylab.style.use('ggplot')



GSAF = "../input/global-shark-attacks/attacks.csv"



AllData = pd.read_csv(GSAF, encoding = 'ISO-8859-1')

AllData['Date'] = AllData['Date'].astype(str)





def find_year(date): #This function tries to extract the year from the dates column

    try:

        matches = [int(y) for y in list(re.findall(r'.*([1-3][0-9]{3})', date))]

        return int(np.mean(matches)) #Some date values containa  range of years

    except:

        return 0



AllData['Year'] = AllData['Date'].apply(find_year)



Attacks = AllData[AllData['Case Number'].notnull()]



startyear, endyear = 1960, 2017

Attacks_by_year = Attacks['Year'].value_counts().sort_index(ascending = True).ix[startyear:endyear]#Fatal Attacks by year

Attacks_by_year.plot(kind = 'line', title = "Global Shark Attacks By Year", color= "navy")

pylab.savefig("output.png")
import pandas as pd

import re

import numpy as np

import matplotlib.pylab as pylab

pylab.style.use('ggplot')



GSAF = "../input/global-shark-attacks/attacks.csv"



AllData = pd.read_csv(GSAF, encoding = 'ISO-8859-1')

AllData = AllData.loc[AllData['Type']== "Unprovoked"]

AllData['Date'] = AllData['Date'].astype(str)





def find_year(date): #This function tries to extract the year from the dates column

    try:

        matches = [int(y) for y in list(re.findall(r'.*([1-3][0-9]{3})', date))]

        return int(np.mean(matches)) #Some date values containa  range of years

    except:

        return 0



AllData['Year'] = AllData['Date'].apply(find_year)



Attacks = AllData[AllData['Case Number'].notnull()]



startyear, endyear = 1960, 2017

Attacks_by_year = Attacks['Year'].value_counts().sort_index(ascending = True).ix[startyear:endyear]

Attacks_by_year.plot(kind = 'line', title = "Unprovoked Shark Attacks By Year", color='blue')

pylab.savefig("output.png")
attacks = pd.read_csv("../input/global-shark-attacks/attacks.csv", encoding="latin1")

(attacks.groupby("Country").count().iloc[:,0]

        .to_frame().reset_index(level=0).sort_values(by="Case Number", ascending=False))