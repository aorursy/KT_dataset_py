#For Kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Import Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium 
#Read DataFrame

df=pd.read_csv("../input/covid19-corona-virus-india-dataset/2020_03_11.csv")

#df=pd.read_csv("2020_03_11.csv")

print("Shape of df: ",df.shape)



#Print Head 

df.head()
#Renaming the columns

df.rename(columns={'Name of State / UT':'State','Total Confirmed cases (Indian National)':'ConfInd',

           'Total Confirmed cases ( Foreign National )':'ConfFor'}, inplace=True)

df
#Sorting the dataframe by ascending order of Total Cases

df["ConfTot"]=df.ConfInd+df.ConfFor

df=df.sort_values(by=["ConfTot"], ascending=False).reset_index()
#Print the status of COVID-19 in India

t_tot=df.ConfTot.sum()

f_tot=df.ConfFor.sum()

i_tot=df.ConfInd.sum()

print(f"Total Confirmed Cases :{t_tot}\nIndian Nationals :{i_tot}\nForeign Nationals :{f_tot}")
#Coordinates for the centre of India

lat_ind=20.5937

lon_ind=78.9629



ind_map = folium.Map(location=[lat_ind, lon_ind], tiles='cartodbpositron',

               min_zoom=4, max_zoom=6, zoom_start=4)



for i in range(df.shape[0]):

    folium.CircleMarker([df.Latitude[i], df.Longitude[i]], 

                        radius=5+(15*df.ConfTot[i]/df.ConfTot.max()),

                        color='None',

                        fill_color='red',fill_opacity=0.4,

                        tooltip=f"Region : {df.State[i]}<br>Confirmed Cases : {df.ConfTot[i]}"

                       ).add_to(ind_map)



ind_map
#Stacked Bar-Graph

plt.figure(figsize=(10,6))

plt.title("Total Patients by State")

sns.set_style(style="whitegrid")

sns.barplot(df.State, df.ConfInd, color='red', label="Indian National")

sns.barplot(df.State, df.ConfFor, color='blue', label="Foreign National",bottom=df.ConfInd)

plt.legend()

plt.xticks(rotation=90)

plt.show()
#Donut plot State wise Split

x=df.ConfTot

labels=df.State

explode=np.zeros(df.shape[0],)

explode=explode+0.1





plt.figure(figsize=(8,8))

plt.title("Patients by State", fontsize=16)

plt.pie(x, labels=labels, explode=explode,wedgeprops=dict(width=0.5),

        autopct='%1.1f%%', startangle=0, )

#plt.legend()

plt.show()
#Pie chart Indians Vs Foreigners

x=[i_tot,f_tot]

labels=["Indians","Foreign Nationals"]

explode=[0.1,0.1]



plt.figure(figsize=(8,8))

plt.title("Indians vs Foreigners")

plt.pie(x, labels=labels, explode=explode,

        autopct='%1.1f%%')

plt.legend()

plt.show()