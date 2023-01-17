#import the packages that you are going to use in the kernel

import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns



#read the datasets 

fifa2020Data=pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')

teamsLeagues=pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/teams_and_leagues.csv')



fifa2020Data.head(2)
#exploration of the datasets

print(fifa2020Data.shape)

print(fifa2020Data.describe())

print(teamsLeagues.info())

print(teamsLeagues.shape)
nationalities=fifa2020Data.groupby(['nationality']).size().reset_index(name='count')

nationalities=nationalities.sort_values('count', ascending=False)

nationalities=nationalities.head(10)
fig = px.bar(nationalities, x = "nationality", y="count", title = "Top 10 of most frequent nationalities in FIFA 2020", color='count')

fig.update_layout(

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)

fig.show()
positions=fifa2020Data.groupby('player_positions').size().reset_index(name='count')

positions=positions.sort_values('count', ascending=False)

positions=positions.head(5)



fig = px.bar(positions, x ="player_positions", y="count", title = "Top 5 of most frequent positions in FIFA 2020", color='count')

fig.update_layout(

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)

fig.show()
overallBiggerThan79=fifa2020Data[fifa2020Data.overall>=79]



highOverallClubs=overallBiggerThan79.groupby('club').size().reset_index(name='count')

highOverallClubs=highOverallClubs.sort_values('count', ascending=False)

highOverallClubs=highOverallClubs.head(10)



fig = px.pie(highOverallClubs, values='count', names='club', title='Teams with more players with a high overall (>=79)',color_discrete_sequence=px.colors.sequential.RdBu)



fig.update_layout(

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)

fig.show()



potentialBiggerThan80=fifa2020Data[(fifa2020Data.potential>=80) & (fifa2020Data.age<=23) ] 



highPotentialClubs=potentialBiggerThan80.groupby('club').size().reset_index(name='count')

highPotentialClubs=highPotentialClubs.sort_values('count', ascending=False)

highPotentialClubs=highPotentialClubs.head(10)



fig = px.pie(highPotentialClubs, values='count', names='club', title='Teams with more young players with a high potential (>=80)',color_discrete_sequence=px.colors.sequential.Plasma_r)



fig.update_layout(

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)

fig.show()
highOverallCountries=overallBiggerThan79.groupby(['nationality','overall']).size().reset_index(name='count')

highOverallCountries

highOverallCountries=pd.DataFrame(highOverallCountries.groupby('nationality')['count'].sum().reset_index())

highOverallCountries



fig = go.Figure(data=go.Choropleth(

            colorscale='Viridis',

            locationmode='country names',

            locations = highOverallCountries['nationality'],

            text = highOverallCountries['nationality'],

            z = highOverallCountries['count'],

            marker_line_color='black',

            autocolorscale=False,

))





fig.update_layout(

   title_text='Countries with more players with a high overall (>=79)',

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",  

)









fig.show()


highPotentialCountries=potentialBiggerThan80.groupby(['nationality','overall']).size().reset_index(name='count')

highPotentialCountries=pd.DataFrame(highPotentialCountries.groupby('nationality')['count'].sum().reset_index())

highPotentialCountries



fig = go.Figure(data=go.Choropleth(

            colorscale='Cividis',

            locationmode='country names',

            locations = highPotentialCountries['nationality'],

            text = highPotentialCountries['nationality'],

            z = highPotentialCountries['count'],

            marker_line_color='black',

            autocolorscale=False,

))





fig.update_layout(

   title_text='Countries with more young players with a high potential (>=80)',

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",  

)









fig.show()
firstDataset=fifa2020Data[(fifa2020Data.value_eur < 500000) & (fifa2020Data.potential >=82) & (fifa2020Data.age <=23)]







fig = px.bar(firstDataset,

             x='potential',

             y='short_name',

             hover_data=['age','value_eur','player_positions','long_name','wage_eur'],

             title='Young players with a value of less than 500,000 euros and potential >=82',

              color='potential',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    'categoryorder':'total ascending'

    

    }

    

)









# plot

fig.show()
secondDataset=fifa2020Data[(fifa2020Data.value_eur >= 1000000) & (fifa2020Data.value_eur <= 5000000) & (fifa2020Data.potential >=82) & (fifa2020Data.age <=23)]



sDataset=secondDataset.nlargest(20,['potential'])#I did this to select the best 15 players 





fig = px.bar(sDataset,

             x='potential',

             y='short_name',

             hover_data=['age','value_eur','player_positions','long_name','wage_eur'],

             title='Top 20 of young players with a value between 1,000,000-5,000,000 euros and potential >=82',

              color='potential',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    'categoryorder':'total ascending'

    

    

    }

    

)









# plot

fig.show()

thirdDataset=fifa2020Data[(fifa2020Data.value_eur >= 6000000) & (fifa2020Data.value_eur <= 10000000) & (fifa2020Data.potential >=82) & (fifa2020Data.age <=23)]



tDataset=thirdDataset.nlargest(20,['potential'])



fig = px.bar(tDataset,

             x='potential',

             y='short_name',

             hover_data=['age','value_eur','player_positions','long_name'],

             title='Top 20 of young players with a value between 6,000,000-10,000,000 euros and potential>=82',

              color='potential',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    'categoryorder':'total ascending'

    

    }

    

)









# plot

fig.show()
fourthDataset=fifa2020Data[(fifa2020Data.value_eur >= 11000000) & (fifa2020Data.value_eur <= 20000000) & (fifa2020Data.potential >=82) & (fifa2020Data.age <=23)]



foDataset=fourthDataset.nlargest(20,['potential'])



fig = px.bar(foDataset,

             x='potential',

             y='short_name',

             hover_data=['age','value_eur','player_positions','long_name'],

             title='Top 20 of young players with a value between 11,000,000-20,000,000 euros and potential >=82',

              color='potential',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    'categoryorder':'total ascending'

    

    }

    

)









# plot

fig.show()
fifthDataset=fifa2020Data[(fifa2020Data.value_eur >= 21000000) & (fifa2020Data.value_eur <= 40000000) & (fifa2020Data.potential >=82) & (fifa2020Data.age <=23)]



fiDataset=fifthDataset.nlargest(20,['potential'])



fig = px.bar(fiDataset,

             x='potential',

             y='short_name',

             hover_data=['age','value_eur','player_positions','long_name'],

             title='Top 20 of young players with a value between 21,000,000-40,000,000 euros and potential >=82',

              color='potential',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    'categoryorder':'total ascending'

    

    }

    

)









# plot

fig.show()
lastDataset=fifa2020Data[(fifa2020Data.value_eur >= 41000000) & (fifa2020Data.potential >=82) & (fifa2020Data.age <=23)]



laDataset=lastDataset.nlargest(20,['potential'])



fig = px.bar(laDataset,

             x='potential',

             y='short_name',

             hover_data=['age','value_eur','player_positions','long_name'],

             title='Top 20 of young players with value of more than 40,000,000 euros and potential >=82',

              color='potential',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0,0,0)',

   plot_bgcolor='rgb(0,0,0)',

    font_family="Courier New",

    font_color="white",

    title_font_family="Courier New",

    title_font_color="white",

    legend_title_font_color="white",

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    'categoryorder':'total ascending'

    

    }

    

)









# plot

fig.show()