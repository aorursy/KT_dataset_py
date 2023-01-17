import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#Load dataset (or use the code on GITHUB with the link above to retrieve information for anyone)

df_award = pd.read_csv("../input/spielberg_awards.csv", sep=",", engine='python')

df_award[:5]
#count won & nominated (project anything that is not WON to NOMINATED)

df_award["simplified_outcome"]="Nominated"

df_award.loc[df_award["outcome"]=="Won","simplified_outcome"]="Won"

group = df_award.groupby(by=["year","simplified_outcome"]).count()



#Populate the number of won & nominated from first year of award to last year

min_year=df_award["year"].min()

max_year=df_award["year"].max()+1



won_serie=pd.Series(index=range(min_year,max_year))

nominated_serie=pd.Series(index=range(min_year,max_year))



for i in range(min_year,max_year):

    won_serie[i]=0

    nominated_serie[i]=0



for i in range(len(group.index)):

    if group.index[i][1]=="Won":

        won_serie[group.index[i][0]]=group.iloc[i]["name"]

    if group.index[i][1]=="Nominated":

        nominated_serie[group.index[i][0]]=group.iloc[i]["name"]

        

#Plot it

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

plt.bar(range(min_year,max_year), won_serie, color = 'red')

plt.bar(range(min_year,max_year), nominated_serie, color = 'orange', bottom = won_serie)

plt.legend(('won','nominated'))

plt.title("Critical acclaim career profile for Steven Spielberg")

plt.show()