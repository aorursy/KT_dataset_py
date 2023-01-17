import pandas as pd

import numpy as np

df=pd.read_csv("../input/nyc-jobs.csv")

df.head()
#Combine the positions from the same agency

agency_dic=dict()

agency_list=list()

positions=list()

for i,j in zip(df['Agency'],df['# Of Positions']): 

            if i not in agency_dic.keys(): agency_dic[i]=j

            else: 

                k=agency_dic[i]

                agency_dic.update({i:j+k})

                

#Create a dataframe of Agency and No. of Positions

agency=list()

counts=list()

other_list=list()

other_count=0

for s,t in agency_dic.items():

    if t>20:

        agency.append(s)

        counts.append(t)

    else:

        other_count=other_count+t  #Agencies with less than 10 postings are ploted as "Other".

        other_list.append(s)

agency.append("Other")

counts.append(other_count)

d={"Agency":agency, "Positions":counts}



df1=pd.DataFrame(d)    

df2=df1.sort_values(by=['Positions'],ascending=False) 
#Create bar chart

import plotly.express as px

fig2 = px.bar(df2, x="Agency", y="Positions", color='Agency', 

              title='NYC Hiring Agency',

              barmode='relative',hover_name='Agency',

              #template='plotly_white', #see the template list by running this: import plotly.io as pio,list(pio.templates

              color_discrete_sequence=px.colors.qualitative.Bold, #if the coumn specified in color= is not numeric discrete color is used. If numeric, color_continuous_scale is used.

             )

fig2.update_layout(showlegend=False, template="plotly_dark")

fig2.show()