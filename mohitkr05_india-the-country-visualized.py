import pandas as pd

import numpy as np

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from pylab import rcParams

# figure size in inches

%matplotlib inline

import plotly.graph_objs as go

import plotly

plotly.offline.init_notebook_mode()
df = pd.read_csv('../input/india-data-worldbank/india_data.csv',index_col=0)

df_area = pd.read_csv('../input/area-data/land_area.csv',index_col=0)

df_population = pd.read_csv('../input/worldpopulation/world_population.csv',index_col=0)

df_pop_india = pd.read_csv('../input/indiapopulation/population_india.csv',index_col=0)

df_land_usage = pd.read_csv('../input/india-land-usage/india_land_usage.csv',index_col=0)
df_area.head()
rcParams['figure.figsize'] = 20, 12

x = df_area["2018"].sort_values(ascending=False)

y = x[x > x.mean()]

ax = plt.barh(y.index , y/1000000,alpha=0.7,label="Countries with Area")

highlight = 'India'

pos = y.index.get_loc(highlight)

ax.patches[pos].set_facecolor('#aa3333')

plt.xlabel('Land area (x million sq. km) ')

plt.ylabel('Country Name')

plt.style.use('fivethirtyeight')

plt.xticks(rotation = 90)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()

x = df_area["2018"].sort_values(ascending=False)

data = dict(type = 'choropleth',

            locations = x.index,

            locationmode = 'country names',

            colorscale= 'Portland',

            text= x.index,

            z=x,

            colorbar = {'title':'Area - Million Sq. Km.', 'len':200,'lenmode':'pixels' })

layout = dict(geo = {'scope':'world'},title="Area")

col_map = go.Figure(data = [data],layout = layout)

col_map.show()
df_population.head()
rcParams['figure.figsize'] = 16, 8

dfinworld = df_population[(df_population.index=="World") |(df_population.index=="India") | (df_population.index=="China")].T

dfinworld1 = dfinworld.drop(["Country Code","Indicator Name"])

dfinworld1["RoW"] = dfinworld1["World"] - dfinworld1["India"]

plt.plot(dfinworld1["RoW"],linestyle='solid',marker='o',label="Rest of World Population")

plt.plot(dfinworld1["India"],linestyle='solid',marker='o',label="Indian Population")

plt.plot(dfinworld1["World"],linestyle='dashed',marker='.',label="World Population")

plt.plot(dfinworld1["China"],linestyle='dashed',marker='.',label="China Population")

plt.xlabel('Year', fontsize=20)

plt.ylabel('Population', fontsize=20)

plt.title('Population growth in India compared by the World', fontsize=20)

plt.xticks(rotation = 90)

plt.style.use('fivethirtyeight')

plt.grid(True)

plt.legend()

plt.tight_layout()
rcParams['figure.figsize'] = 20, 12

x = df_population[df_population.index != "World"]["2018"].sort_values(ascending=False)

z = x[x > x.mean()]

ax = plt.barh(z.index , z,alpha=0.7,label="Population of Countries")

highlight = 'India'

pos = z.index.get_loc(highlight)

ax.patches[pos].set_facecolor('#aa3333')

plt.xlabel('Population (x million) ')

plt.ylabel('Country Name')

plt.style.use('fivethirtyeight')

plt.xticks(rotation = 90)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()
x = df_population["2018"]

y = (x/x.max())*100

z = y[y.index != "World"].sort_values(ascending=False)

#Plotting on the WorldMap using plotly

data = dict(type = 'choropleth',

            locations = z.index,

            locationmode = 'country names',

            colorscale= 'Portland',

            text= z.index,

            z=z,

            colorbar = {'title':'Population %', 'len':200,'lenmode':'pixels' })

layout = dict(geo = {'scope':'world'},title="Population around the world")

col_map = go.Figure(data = [data],layout = layout)

col_map.show()
df_land_usage1  = df_land_usage.T

df_land_usage2 = df_land_usage1.fillna(method='bfill')

cols = ['Agricultural_land_percent','Forest_area_percent', 'Terrestrial_protected_areas_percent']

for col in cols:

    plt.plot(df_land_usage2[col],linestyle='solid',marker='o',label=col)

plt.legend()

plt.title("Land Usage Growth")

plt.xlabel('Year')  

plt.ylabel('%age of Area')

plt.xticks(rotation = 90)

plt.legend()

plt.show()  
df1 = df_population["2018"]

df2 = df_area["2018"]
df1.head()
df2.head()
dfn1 = pd.merge(left=df1, right=df2, left_on='Country Name', right_on='Country Name')
dfn1.rename(columns={'2018_x': 'population','2018_y':'area'}, inplace=True)
dfn1.head()
dfn1["density"] = dfn1["population"]/dfn1["area"]
df_pop  = df_pop_india.T
df_pop.isna().sum()
df_pop.fillna(method='ffill', inplace=True)
df_pop.head()
rcParams['figure.figsize'] = 16, 8

width = 0.25 

plt.plot(df_pop["Population, total"],linestyle='solid',marker='o',label="Population, total")

x_indexes = np.arange(len(df_pop.index))

plt.bar(x_indexes-width,df_pop["Population, male"],  width+0.10, label="Population, male")

plt.bar(df_pop.index,df_pop["Population, female"],width+0.10, label="Population, female")

plt.xlabel('Year', fontsize=20)

plt.ylabel('Population', fontsize=20)

plt.title('Population growth in India/ Gender Wise', fontsize=20)

plt.xticks(rotation = 90)

plt.style.use('fivethirtyeight')

plt.grid(True)

plt.legend()

plt.tight_layout()
df_pop.head()
cols = df_pop.columns

for col in cols:

    if (("ages" in col) & ("total" in col)) :

        plt.plot(df_pop[col],linestyle='solid',marker='o',label=col)

plt.xlabel('Year', fontsize=20)

plt.ylabel('Population', fontsize=20)

plt.title('Age distribution variation of Population - All Gender', fontsize=20)

plt.xticks(rotation = 90)

plt.style.use('fivethirtyeight')

plt.grid(True)

plt.legend()

plt.tight_layout()

cats = ['Population ages 00-04, female (% of female population)',

'Population ages 00-04, male (% of male population)',

'Population ages 05-09, female (% of female population)',

'Population ages 05-09, male (% of male population)',

'Population ages 10-14, female (% of female population)',

'Population ages 10-14, male (% of male population)',

'Population ages 15-19, female (% of female population)',

'Population ages 15-19, male (% of male population)',

'Population ages 20-24, female (% of female population)',

'Population ages 20-24, male (% of male population)',

'Population ages 25-29, female (% of female population)',

'Population ages 25-29, male (% of male population)',

'Population ages 30-34, female (% of female population)',

'Population ages 30-34, male (% of male population)',

'Population ages 35-39, female (% of female population)',

'Population ages 35-39, male (% of male population)',

'Population ages 40-44, female (% of female population)',

'Population ages 40-44, male (% of male population)',

'Population ages 45-49, female (% of female population)',

'Population ages 45-49, male (% of male population)',

'Population ages 50-54, female (% of female population)',

'Population ages 50-54, male (% of male population)',

'Population ages 55-59, female (% of female population)',

'Population ages 55-59, male (% of male population)',

'Population ages 60-64, female (% of female population)',

'Population ages 60-64, male (% of male population)',

'Population ages 65-69, female (% of female population)',

'Population ages 65-69, male (% of male population)',

'Population ages 70-74, female (% of female population)',

'Population ages 70-74, male (% of male population)',

'Population ages 75-79, female (% of female population)',

'Population ages 75-79, male (% of male population)',

'Population ages 80 and above, female (% of female population)',

'Population ages 80 and above, male (% of male population)']
df_ages = df_pop[cats]
df_ages
df_ages1 = df_ages.rename(columns=lambda x: x.split(',')[1].split(" ")[1] +" age " + x.split(',')[0].split(" ")[-1])
df_ages2 = df_ages1.T["2018"]
df_ages3 = df_ages2.to_frame()
df_ages3["x"] = df_ages3.index
df_ages3["gender"] = df_ages3["x"].apply(lambda x:x.split(" ")[0])
df_ages3["age-group"] = df_ages3["x"].apply(lambda x:x.split(" ")[-1])
df_ages4 = df_ages3.drop(["x"],axis="columns")
df_ages4.rename(columns={"2018":"percent"},inplace=True)
x = df_ages4[df_ages4.gender == "male"]

y = df_ages4[df_ages4.gender == "female"]

labels = x["age-group"]

plt.bar(labels, x.percent, color="#6c3376" , label = "Males")

plt.bar(labels, y.percent, bottom=x.percent, color="#f3e151" , label = "Females")

plt.xlabel('Age Group')  

plt.ylabel('%age of population')

plt.legend()

plt.show()  

def get_age_low(x):

    if(x == "above"):

        return float(80)

    else:

        return float(x.split("-")[0])

df_ages4["low_age"] = df_ages4["age-group"].apply(lambda x:get_age_low(x))
def annotate_age_type(x):

    if x <= 5:

        return "toddlers"

    elif (x > 5) & (x <= 10):

        return "kids"

    elif (x > 10) & (x <= 19):

        return "teenager"

    elif (x > 19) & (x <= 35):

        return "young"

    elif (x > 35) & (x <= 55):

        return "middle"

    elif (x > 55) & (x <= 70):

        return "old"

    else:

        return "too old"

df_ages4["age_type"] = df_ages4["low_age"].apply(lambda x:annotate_age_type(x))
x = df_ages4[df_ages4.gender == "male"]

y = df_ages4[df_ages4.gender == "female"]

labels = x["age_type"]

plt.bar(labels, x.percent, color="#6c3376" , label = "Males")

plt.bar(labels, y.percent, bottom=x.percent, color="#f3e151" , label = "Females")

plt.xlabel('Age Group')  

plt.ylabel('%age of population')

plt.legend()

plt.show()  
x = dfn1["density"].sort_values(ascending=False)

z = x[x > x.mean()]

ax = plt.barh(z.index , z,alpha=0.7,label="Population Density")

highlight = 'India'

pos = z.index.get_loc(highlight)

ax.patches[pos].set_facecolor('#aa3333')

plt.xlabel('Number of People/ Sq. Km.')

plt.ylabel('Country Name')

plt.style.use('fivethirtyeight')

plt.xticks(rotation = 90)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()