import numpy as np

import pandas as pd

import os



import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt
df = pd.read_csv('../input/Munich-Meets-African-Lit.csv')
df.head(5)
df.tail(5)
df_countries = df.groupby(['Country'])

x = df_countries['Country'].count().reset_index(name='Count').sort_values(['Count'], ascending=0)



fig = go.Figure( go.Bar(

              x = x.Count[::-1],

              y = x.Country[::-1],

              orientation='h', 

              opacity=0.5, 

              marker=dict(color='rgba(207, 0, 15, 1)')

                        )

               )



fig.update_traces(marker_line_color='rgb(10,48,107)')

fig.update_layout(title_text='Number of a Country\'s Appearances on the Book Club List', 

                  xaxis_title='Count', 

                  yaxis_title='Countries') 
px.pie(df_countries, x['Country'], x['Count'], 

       color_discrete_sequence=px.colors.sequential.Rainbow_r, 

       title="Proportion of a Country's Appearances on the Book Club List")

df_countries = df.groupby(['Region'])

x = df_countries['Region'].count().reset_index(name='Count').sort_values(['Count'], ascending=0)



px.pie(df_countries, x['Region'], x['Count'], 

       color_discrete_sequence=px.colors.sequential.Hot, 

       title="Proportion of a African Regional Appearances on the Book Club List")

df_gender = df.groupby(['Author Gender'])

x = df_gender['Author Gender'].count().reset_index(name='Count').sort_values(['Count'], ascending=0)





px.pie(df_gender, x['Author Gender'], x['Count'], 

       color_discrete_sequence=px.colors.sequential.Bluered, 

       title="Gender binary split of selected Authors")

df_publisher = df.groupby(['Publisher'])

x = df_publisher['Publisher'].count().reset_index(name='Count').sort_values(['Count'], ascending=0)



fig = go.Figure( go.Bar(

              x = x.Count[::-1],

              y = x.Publisher[::-1],

              orientation='h', 

              opacity=0.5, 

              marker=dict(color='rgba(60, 179, 113, 1)')

                        )

               )



fig.update_traces(marker_line_color='rgb(60, 179, 113)')

fig.update_layout(title_text='Publishers representing authors on Book List', 

                  xaxis_title='Count', 

                  yaxis_title='Publishing Houses') 
# The years are read in as DD.MM.YY format 

# We can get the full names of the years by switching from YY to YYYY



years = [ x[-2:] for x in df['Date Published'] ]



years_full = []

for y in years: 

    if 0 < float(y) < 20: 

        years_full.append('20'+y)

    else: 

        years_full.append('19'+y)



print(years)

print("Confusing much? Let's modify the array!\n")

print(years_full)

print("Much better!")
years_full = sorted(years_full)



plt.figure(figsize=(10,5))

def plot(x, x_label, color, r):

    fs = 14

    plt.hist(x, align='mid', rwidth=r, color=color)

    plt.xlabel(x_label, fontsize=fs)

    plt.ylabel('Count', fontsize=fs)

    plt.xticks(fontsize=10)

    

    

plot(years_full, 'Year Published', '#5726D8', 0.5)

plt.title('Distribution of Publication Years for Books listed')

plt.show()
year_meeting = [ x[-2:] for x in df['Date of Meeting'] ]

year_meeting_full = [ '20'+y for y in year_meeting ]



plt.figure(figsize=(10,5))

plot(years_full, 'Year Published', '#5726D8', 0.2)

plot(year_meeting_full, 'Year of Meeting', '#D826AA',10.)

plt.show()