# Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import plotly.express as px

from tabulate import tabulate
# Data Load

url = '../input/statewise-players-for-the-khelo-india-scheme/Session_245_AU3036_1.1.csv'

df = pd.read_csv(url, header='infer')



# Replace null with 0

df = df.fillna(0)



# Drop Columns

df.drop(['Sl.No','Total'], axis=1, inplace=True)

df.drop([16], axis=0, inplace=True)



# Set Index

df.set_index('Sports Discipline', drop=True, inplace=True)
# Utility Function



def analyse(sport):

    

    count = []   # define empty list to store player count

    

    for col in df.columns:

        if df[col][sport] > 0:

            count.append([col,df[col][sport]])

        else:

            None

    

    player_count = pd.DataFrame(data=count, columns=['State','Count'])

    player_count.sort_values(by=['Count'], inplace=True, ascending=False)

    player_count.reset_index(drop=True, inplace=True)

    

    print(f"Total Selected/Identified for {sport}:", player_count.Count.sum(),"\n")

    print("Number of players selected/identified per state:\n")

    print(tabulate(player_count, headers=['Sno.','State','PlayerCount'], tablefmt='pretty'))

    print()

    

    fig=px.pie(player_count,values='Count',names='State', color_discrete_sequence=px.colors.sequential.RdBu, hover_data=['Count'])

    

    fig.update_layout(title="% of Players Identified/Selected State-Wise in "+sport,title_x=0,font=dict(size=15))

    fig.update_traces(textfont_size=12,textinfo='label+percent')

    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

    fig.show()

    #count = []

    #del player_count

    
analyse('Archery')
analyse('Athletics')
analyse('Badminton')
analyse('Basketball')
analyse('Boxing')
analyse('Football')
analyse('Gymnastics')
analyse('Hockey')
analyse('Judo')
analyse('Kabaddi')
analyse('Kho-kho')
analyse('Shooting')
analyse('Swimming')
analyse('Volleybal')
analyse('Weigtlifting')
analyse('Wrestling')