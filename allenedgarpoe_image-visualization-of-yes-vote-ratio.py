import os

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected = True)
state_file = os.path.join("../input/general-assembly","states.csv")

state_df = pd.read_csv(state_file) #none 값이 들어간 행 삭제

change_name = {'United States of America': "United States", "South Korea":"Korea, South", "North Korea":"Korea, North","German Democratic Republic":'Germany',"Myanmar":'Burma',"Yemen People's Republic":'Yemen'}

country_list= list(set(state_df['state_name'].dropna().values))
yes_vote_ratio = dict()

for country in country_list:

    new_df = state_df[state_df['state_name']==country]

    ratio = sum(new_df['yes_votes'])/sum(new_df['all_votes'])

    if country in change_name:

        country = change_name[country]

    yes_vote_ratio[country] = ratio
yes_vote_ratio
countries_df = pd.DataFrame.from_dict(yes_vote_ratio, orient = 'index', columns=['yes_vote_ratio'])

code_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')



def return_country_code(con):

    if con in code_df['COUNTRY'].values:

        return code_df[code_df['COUNTRY'] == con]['CODE'].values[0]



countries_df['Country'] = countries_df.index

countries_df['Code'] = countries_df['Country'].apply(return_country_code)

countries = countries_df.dropna()
data=dict(

    type = 'choropleth',

    locations = countries['Code'],

    z = countries['yes_vote_ratio'],

    text = countries['Country'],

    colorscale = 'YlOrRd',

    marker_line_color='darkgray',

    marker_line_width=0.7,

    colorbar_title = "yes_vote_ratio",

)



layout = dict(title_text='Yes vote ratio per States',

    geo=dict(

        showframe=False,

        showcoastlines=True,

        projection_type='equirectangular'

    ))



fig = go.Figure(data = [data], layout = layout)

iplot(fig)
top_countries = ["China","France","United States of America","Russia","United Kingdom"]

fig, axs = plt.subplots(nrows=1, ncols=5, sharex = False, sharey = True, figsize=(25,4))

for i, country in enumerate(top_countries):

    new_df = state_df[state_df['state_name'] == country]

    ax = axs[i]

    ax.set_xlim(1945,2015)

    ax.plot(new_df['year'], new_df['yes_votes'].values/new_df['all_votes'].values)

    ax.set_title(str(country))

plt.show()
resolution_df = pd.read_csv("../input/general-assembly/resolutions.csv").dropna()

data_df = pd.read_csv("../input/general-assembly/votes.csv")

merge_df = pd.merge(data_df, resolution_df[['resolution', 'vote_date',"colonization","human_rights","israel_palestine", "disarmament", "nuclear_weapons", "economic_development"]], on="resolution")



vote_respond = {1:"Yes", 2:"No", 3:"Abstain", 8:"Absent", 9:"Not a Member"}

new_vote = [vote_respond[i] for i in merge_df['vote']]

merge_df['vote'] = new_vote

merge_df.tail()