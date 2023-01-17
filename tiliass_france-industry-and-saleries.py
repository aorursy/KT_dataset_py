

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as graph

import plotly.tools as tools

# Any results you write to the current directory are saved as output.
geography = pd.read_csv("../input/name_geographic_information.csv")

industry = pd.read_csv("../input/base_etablissement_par_tranche_effectif.csv")

salary = pd.read_csv("../input/net_salary_per_town_categories.csv")

population = pd.read_csv("../input/population.csv")



#industry.head(5)

salary.head(5)

#industry.info()
# 1

geography["longitude"] = geography["longitude"].apply(lambda x: str(x).replace(',','.'))

mask = geography["longitude"] == '-'

geography.drop(geography[mask].index, inplace=True)

geography.dropna(subset = ["longitude", "latitude"], inplace=True)

geography["longitude"] = geography["longitude"].astype(float)

geography.head(3)

geography.drop_duplicates(subset=["code_insee"], keep="first", inplace=True)

#geography.head(3)

industry.head(10)

industry=industry.drop(industry[industry['CODGEO'].apply(lambda x: str(x).isdigit())==False].index)



industry['CODGEO']=industry['CODGEO'].astype(int)

## MERGING

merged1=pd.merge(geography,industry,left_on='code_insee',right_on='CODGEO')

merged1.head()
geography.info()
salary = salary[salary["CODGEO"].apply(lambda x: str(x).isdigit())]

salary["CODGEO"] = salary["CODGEO"].astype(int)
positions = ["Executive", "Middle manager", "Employee", "Worker"]

woman_positions = ["SNHMFC14", "SNHMFP14", "SNHMFE14", "SNHMFO14"]

woman_salary_positions = salary[woman_positions].mean().tolist()

man_positions = ["SNHMHC14", "SNHMHP14", "SNHMHE14", "SNHMHO14"]

man_salary_positions = salary[man_positions].mean().tolist()

positions_men_women=["SNHMC14", "SNHMP14", "SNHME14", "SNHMO14"]

salary_positions = salary[positions_men_women].mean().tolist()
trace = graph.Bar(x = positions,y = woman_salary_positions,name='Women',

    marker=dict(

        color='rgb(55, 83, 109)'

    ))

trace2 = graph.Bar(x = positions,y = man_salary_positions,name='Men',

    marker=dict(

        color='rgb(26, 118, 255)'

    ))



trace3 = graph.Bar(x = positions,y = salary_positions,name='Mean net salary per hour',

    marker=dict(

        color='rgb(0, 83, 230)'

    ))





data = [trace, trace2, trace3]

#layout = graph.Layout(barmode='stack')

#fig = graph.Figure(data=data, layout=layout)



layout = graph.Layout(

    title='US Export of Plastic Scrap',

    xaxis=dict(

        tickfont=dict(

            size=14,

            color='rgb(107, 107, 107)'

        )

    ),

    yaxis=dict(

        title='USD (millions)',

        titlefont=dict(

            size=16,

            color='rgb(107, 107, 107)'

        ),

        tickfont=dict(

            size=14,

            color='rgb(107, 107, 107)'

        )

    ),

    legend=dict(

        x=1.0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1

)

fig = graph.Figure(data=data, layout=layout)

py.iplot(fig)
data = [graph.Bar(

            x = merged1['nom_région'],

            y = merged1['E14TS500'],

    marker=dict(color='rgb(0, 83, 230)'))]



layout = graph.Layout(

    title='Regions number of big industries (>500)'

)



fig = graph.Figure(data=data, layout=layout)



py.iplot(fig)

data = [graph.Bar(

            x = merged1['nom_région'],

            y = merged1['E14TS200'],

    marker=dict(color='rgb(0, 83, 230)')

    )]



layout = graph.Layout(

    title='Regions number of big industries (>200)'

)



fig = graph.Figure(data=data, layout=layout)



py.iplot(fig)

merged1.head()




data = [graph.Bar(

            x = merged1['chef.lieu_région'],

            y = merged1['E14TS500'],

    marker=dict(color='rgb(0, 83, 230)')

    )]



layout = graph.Layout(title='Contributions by departement')



fig = graph.Figure(data=data, layout=layout)



py.iplot(fig)


#PARIS

Data_paris=merged1[merged1['nom_région']=='Île-de-France']

#Data_paris=merged1[merged1['numéro_département']=='93']

Data_paris.head(20)



data = [graph.Bar(

            x = Data_paris['préfecture'],

            y = Data_paris['E14TS500'],

    marker=dict(

        color=['rgba(222,45,38,0.8)','rgba(204,204,204,1)' ,

               'rgba(204,204,204,1)', 'rgba(204,204,204,1)',

               'rgba(204,204,204,1)']),

            

            

            opacity=0.8

    )]



layout = graph.Layout(

    title='Contributions by cities in Ile de france'

)



fig = graph.Figure(data=data, layout=layout)



py.iplot(fig)


