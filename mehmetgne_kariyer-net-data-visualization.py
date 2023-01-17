!pip install folium
# importing libraries



from __future__ import print_function

from ipywidgets import interact, interactive, fixed, interact_manual

from IPython.core.display import display, HTML



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import folium

import plotly.graph_objects as go

import seaborn as sb

import ipywidgets as widgets
#importing dataset

df = pd.read_csv('../input/kariyernet-turkey-professions-informations/cleandataset.csv')

type(df)
#sorting dataset by mean salary

sorted_job_df = df.sort_values('mean_salary', ascending=False)

sorted_job_df.head(15)
#function for fill background of salaries columns

fig = go.FigureWidget( layout=go.Layout() )

def highlight_col(x):

    dg = 'background-color: darkgray'

    a = 'background-color: aqua'

    t = 'background-color: teal'

    df1 = pd.DataFrame('', index=x.index, columns=x.columns)

    df1.iloc[:, 1] = dg

    df1.iloc[:, 2] = a

    df1.iloc[:, 3] = t

    

    return df1



#displaying first five rows of costumized dataset

sorted_job_df.head(5).style.apply(highlight_col, axis=None)
#function for plotting mean salary for each profession by interactive dashboard via 'plotly' library.

def bubble_chart(n):

    fig = px.scatter(sorted_job_df.head(n), x="job_name", y="mean_salary", size="mean_salary", color="job_name",

               hover_name="job_name", size_max=60)

    fig.update_layout(

    title=str(n) +" Most Winning Profession (Average)",

    xaxis_title="Professions",

    yaxis_title="Mean Salary",

    width = 700

    )

    fig.show();



interact(bubble_chart, n=10)

ipywLayout = widgets.Layout(border='solid 2px green')

ipywLayout.display='none'

widgets.VBox([fig], layout=ipywLayout)
#bar plot for most winning professions

px.bar(

    sorted_job_df.head(10),

    x = "job_name",

    y = "mean_salary",

    labels= {'job_name': 'Professions',

            'mean_salary': 'Mean Salary (₺)'},

    title= "Top 10 most earning professions", # the axis names

    color_discrete_sequence=["aqua"], 

    height=500,

    width=800

)
sorted_job_adv = df.sort_values('job_advertisement', ascending=False)

sorted_job_adv.head(15)
px.bar(

    sorted_job_adv.head(15),

    x = "job_name",

    y = "job_advertisement",

    labels={'job_name':' Professions',

           'job_advertisement': 'Job Advertisements'},

    title= "Top 15 most wanted professions", # the axis names

    color_discrete_sequence=["blue"],

    height=500,

    width=800,

).update_layout(showlegend=False)
#to look at correlations between numeric columns

numeric_df  = df[['mean_salary','max_salary','min_salary','total_participants','number_of_entries ','male_percentage','female_percentage','job_advertisement']]

sb.pairplot(numeric_df)
#concat skills columns and make them array to sum

#there is '0' value we extract its by beginning one

most_skill = df[['first_skill','second_skill','third_skill','fourth_skill','fiveth_skill']].values.flatten()

most_skill = pd.value_counts(most_skill)

most_skill = most_skill[1:]

most_skill
#plotting most skill wia plolty

px.bar(

    most_skill.head(15),

    title= "Top 15 most wanted professions", # the axxis names

    labels={'value':'# of skills', 'index':'Skills'},

    color_discrete_sequence=["black"],

    height=500,

    width=800

).update_layout(showlegend=False)
#concat schools columns and make them array to sum

#there is '0' value we extract its by beginning one

most_school = df[['first_school','second_school','third_school','fourth_school','fiveth_school']].values.flatten()

most_school = pd.value_counts(most_school)

most_school = most_school[1:]

most_school
#plotting most school wia plolty

px.bar(

    most_school.head(15),

    title= "Top 15 preferable schools in professions", # the axxis names

    labels={'value':'# of schools', 'index':'Schools'},

    color_discrete_sequence=["gray"],

    height=500,

    width=800

).update_layout(showlegend=False)
#concat department columns and make them array to sum

#there is '0' value we extract its by beginning one

most_grad = df[['first_grad','second_grad','third_grad','fourth_grad','fiveth_grad']].values.flatten()

most_grad = pd.value_counts(most_grad)

most_grad = most_grad[1:]

most_grad
#plotting most department wia plolty

px.bar(

    most_grad.head(15),

    title= "Top 15 preferable department in professions", # the axxis names

    labels={'value':'# of department', 'index':'Departments'},

    color_discrete_sequence=["lightgray"],

    height=500,

    width=800

).update_layout(showlegend=False)