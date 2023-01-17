import pandas as pd

import numpy as np



%matplotlib notebook

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import pandas_profiling
# Seaborn advanced                                                                                                                                                           

sns.set(style='ticks',          # 'ticks', 'darkgrid'                                                                                                                        

        palette='colorblind',   # 'colorblind', 'pastel', 'muted', 'bright'                                                                                                  

        #palette=sns.color_palette('Accent'),   # 'Set1', 'Set2', 'Dark2', 'Accent'                                                                                          

        rc = {                                                                                                                                                               

           'figure.autolayout': True,   # Automaticky nastaví velikost grafu, aby se vešel do obrazu                                                                         

           'figure.figsize': (10, 8),    # Velikost obrázku - šířka, výška (v palcích)                                                                                       

           'legend.frameon': True,      # Rámeček okolo legendy                                                                                                              

           'patch.linewidth': 2.0,      # Velikost čáry okolo rámečku                                                                                                        

           'lines.markersize': 6,       # Velikost bodů                                                                                                                      

           'lines.linewidth': 2.0,      # Tloušťka čar                                                                                                                       

           'font.size': 20,             # Velikost hodnot na osách                                                                                                           

           'legend.fontsize': 20,       # Velikost textu v legendě                                                                                                           

           'axes.labelsize': 16,        # Velikost názvů os                                                                                                                  

           'axes.titlesize': 22,        # Velikost nadpisu                                                                                                                   

           'axes.grid': True,           # Mřížka                                                                                                                             

           'grid.color': '0.9',         # Světlost čar mřížky - 1 = bílá, 0 = černá                                                                                          

           'grid.linestyle': '-',       # Typ čárkování mřížka                                                                                                               

           'grid.linewidth': 1.0,       # Tloušťka čar mřížky                                                                                                                

           'xtick.labelsize': 20,       # Velikost popisů na x-ové ose                                                                                                       

           'ytick.labelsize': 20,       # Velikost popisů na y-ové ose                                                                                                       

           'xtick.major.size': 8,       # Velikost čárek na x-ové ose                                                                                                        

           'ytick.major.size': 8,       # Velikost čárek na y-ové ose                                                                                                        

           'xtick.major.pad': 10.0,     # Vzdálenost čísel na x-ové ose od osy                                                                                               

           'ytick.major.pad': 10.0,     # Vzdálenost čísel na y-ové ose od osy                                                                                               

           }                                                                                                                                                                 

       )                                                                                                                                                                     

plt.rcParams['image.cmap'] = 'viridis'  
plays = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv")
# Get players out of dataset of plays and rename columns

players = plays[["DisplayName", "PlayerHeight", "PlayerWeight", "PlayerBirthDate", "PlayerCollegeName"]].drop_duplicates()

players.rename(columns={"DisplayName":"Name", "PlayerHeight": "Height", "PlayerWeight": "Weight", "PlayerBirthDate":"BirthDate", "PlayerCollegeName":"College"}, inplace=True)

# Transfer to SI units

players["Weight"] = players["Weight"] * 0.45359237   # In kilograms

players["Height"] = players["Height"].str.split("-", expand=True)[0].astype(float) * 30.48 + players["Height"].str.split("-", expand=True)[1].astype(float) * 2.54   # in centimeters

# Set birthdate column as datetime and add age in years

players["BirthDate"] = pd.to_datetime(players['BirthDate'])

players["Age"] = (pd.Timestamp('20191010') - players["BirthDate"]).dt.days / 365   # In years

# Compute body mass index (BMI) and add classification

players["BMI"] = players["Weight"] / (players["Height"] / 100)**2    # Weight has to be in kilograms, height in meters (factor 100 to transfer from centimeters)

players["ObesityClassification"] = pd.cut(players["BMI"], bins=[0, 18.5, 25, 30, 35, 40, 100], labels=["Underweight", "Normal weight", "Pre-obesity", "Obesity class 1", "Obesity class 2", "Obesity class 3"])
players
print(f"Players are from {len(players['College'].unique())} different colleges.")

print(f"The youngest player is {players.loc[players['Age'] == players['Age'].min(), 'Name'].values[0]} with age {round(players['Age'].min(), 2)} years from {players.loc[players['Age'] == players['Age'].min(), 'College'].values[0]}.")

print(f"The oldest player is {players.loc[players['Age'] == players['Age'].max(), 'Name'].values[0]} with age {round(players['Age'].max(), 2)} years from {players.loc[players['Age'] == players['Age'].max(), 'College'].values[0]}.")

print(f"The most lightweight player is {players.loc[players['Weight'] == players['Weight'].min(), 'Name'].values[0]} with weight {round(players['Weight'].min(), 2)} kg from {players.loc[players['Weight'] == players['Weight'].min(), 'College'].values[0]}.")

print(f"The heaviest is {players.loc[players['Weight'] == players['Weight'].max(), 'Name'].values[0]} with weight {round(players['Weight'].max(), 2)} kg from {players.loc[players['Weight'] == players['Weight'].max(), 'College'].values[0]}.")

print(f"The shortest player is {players.loc[players['Height'] == players['Height'].min(), 'Name'].values[0]} with height {round(players['Height'].min(), 2)} cm from {players.loc[players['Height'] == players['Height'].min(), 'College'].values[0]}.")

print(f"The tallest player is {players.loc[players['Height'] == players['Height'].max(), 'Name'].values[0]} with height {round(players['Height'].max(), 2)} cm from {players.loc[players['Height'] == players['Height'].max(), 'College'].values[0]}.")
players.describe()
plt.figure()

sns.distplot(players["Weight"])

plt.xlabel("Weight [kg]")

plt.title("Weight of players")

plt.savefig("WeightHist.png")

plt.show()
plt.figure()

sns.distplot(players["Height"], bins=16)

plt.xlabel("Heigh [cm]")

plt.title("Height of players")

plt.savefig("HeightHist.png")

plt.show()
plt.figure()

sns.distplot(players["Age"])

plt.xlabel("Age [years]")

plt.title("Age of players")

plt.savefig("AgeHist.png")

plt.show()
plt.figure()

sns.distplot(players["BMI"])

plt.xlabel("BMI [kg $\cdot$ m$^{-2}$]")

plt.title("Body mass index of players")

plt.savefig("BMIHist.png")

plt.show()
colormap = dict(zip(players["ObesityClassification"].unique(), sns.color_palette()))

fig = plt.figure()

ax = plt.subplot(111)

sns.scatterplot(players["Weight"], players["Height"], hue=players["ObesityClassification"], linewidth=0, alpha=0.6, ax=ax)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles=handles[1:], labels=labels[1:], prop={'size': 16}, loc="lower right")

plt.xlabel("Weight")

plt.ylabel("Height")

plt.savefig("WeightVsHeight.png")

plt.show()
plt.figure(figsize=(7.5,7.5))

sns.pairplot(players, height=4.5, aspect=1, plot_kws={"linewidth":0, "alpha":0.3})

plt.savefig("ScatterMatrix.png")
plt.figure(figsize=(8,8))

sns.heatmap(players.corr(), cmap="RdYlGn", annot=True)

plt.title("Correlation heatmap")

plt.savefig("CorrelationHeatmap.png")

plt.show()