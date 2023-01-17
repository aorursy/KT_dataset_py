# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import random

import math

import plotly.express as px
# PARAMETRES : 



# Nombre d'agents 

population = 500



# Proportion de la population infectee a l'etat initial

prop_infected_depart = 0.15



# Probabilite qu'un 'clean' se fasse infecter lorsqu'il est sur la meme cellule qu'un 'sick'

infectiousness = 0.98



# Taille de la grille (carree)

grid_size = 40



# Probabilite de mourir au prochain mois lorsqu'on a le virus 

proba_die = 0.05



# Probabilite de guerir au prochain mois lorsqu'on a le virus

proba_heal = 0.05



# Probabilite de rester malade au prochain mois 

proba_remain = 1-proba_die-proba_heal



# Parametres de la loi normale qui sert a generer les ages a la situation initiale 

mu_lifetime_years = 80

sigma_lifetime_years = 10



# Proportion de nouveaux nes tous les mois par rapport a la population qui a plus de 18 ans 

 #(explications dans le rapport)

births_per_month_part = 0.02



# Nombre de mois generes par le model (10 ans)

duration_model_months = 10*12

agent_id = range(1,population+1)

pos_x = np.random.randint(1, grid_size+1, size = population)

pos_y = np.random.randint(1, grid_size+1, size = population)

status = np.concatenate((np.repeat("sick", prop_infected_depart*population), np.repeat("clean", np.ceil(population-(prop_infected_depart*population)))), axis=None)

age_months = np.random.randint(0,((12*80)+1), size = population)

lifetime_months = np.floor(np.random.normal(mu_lifetime_years*12, sigma_lifetime_years*12, population))

time = np.repeat(0, population)



data = {'agent_id': agent_id, 'pos_x': pos_x, 'pos_y': pos_y, 'status': status, 

        'age_months': age_months, 'lifetime_months': lifetime_months, 'time': time} 

df_0 = pd.DataFrame(data)



# Au cas où il y ait des ages superieurs a l'age limite 

df_0.age_months.where(age_months < lifetime_months, 0, inplace = True)
df_0.dtypes
df_0 = df_0.astype({"time": 'int32', "lifetime_months": 'int32', 

                   "agent_id": 'int16', "pos_x": 'int16', 

                   "pos_y": 'int16', "age_months": 'int32', "agent_id": 'int16'})

df_0.dtypes
df = df_0

df
liste = []

liste.append(df_0)

liste[0]
for i in range(1, duration_model_months+1) : 

    

    liste.append(liste[i-1].copy())

    

    # Ajout d'un mois pour le monde (time) : 

    liste[i].time += 1

    

    # Ajout d'un mois d'age pour chaque agent : 

    liste[i].age_months += 1

    

    # Naissances : 

    births_per_month = int(np.ceil(len(liste[i-1].loc[liste[i-1].age_months > 18*12])*births_per_month_part))

    agent_id = range(max(liste[i-1].agent_id)+1, max(liste[i-1].agent_id)+1+births_per_month)

    pos_x = np.random.randint(1, grid_size+1, size = births_per_month)

    pos_y = np.random.randint(1, grid_size+1, size = births_per_month)

    status = np.repeat("clean", births_per_month) 

    age_months = np.repeat(0, births_per_month)

    lifetime_months = np.floor(np.random.normal(mu_lifetime_years*12, sigma_lifetime_years*12, births_per_month)) 

    time = np.repeat(max(liste[i-1].time)+1, births_per_month)

    data = {'agent_id': agent_id, 'pos_x': pos_x, 'pos_y': pos_y, 'status': status, 

            'age_months': age_months, 'lifetime_months': lifetime_months, 'time': time} 

    df1 = pd.DataFrame(data)

    df1 = df1.astype({"time": 'int32', "lifetime_months": 'int32', 

                      "agent_id": 'int16', "pos_x": 'int16', 

                      "pos_y": 'int16', "age_months": 'int32'})

    liste[i] = pd.concat([liste[i], df1], ignore_index=True)

    

    # Sortir les morts de la grille : 

    liste[i].loc[(liste[i].status == "dead_other") | (liste[i].status == "dead_virus"), 'pos_x'] = grid_size+1

    liste[i].loc[(liste[i].status == "dead_other") | (liste[i].status == "dead_virus"), 'pos_y'] = grid_size+1

    

    # Si un agent depasse son 'lifetime_months', alors il meurt de vieillesse : 

    liste[i].loc[liste[i].age_months >= liste[i].lifetime_months, 'status'] = "dead_other"

    

    # Markov Chain : probabilités de changer d'état lorsqu'on est malade : 

    for agent in range(0, len(liste[i])) : 

        if liste[i].loc[agent, 'status'] == "sick" : 

            liste[i].loc[agent, 'status'] = random.choices(['sick', 'dead_virus', 'immunized'], weights=[proba_remain, proba_die, proba_heal], k=1)

#    liste[i].loc[liste[i].status == "sick", 'status'] = random.choices(['sick', 'dead_virus', 'immunized'], weights=[proba_remain, proba_die, proba_heal], k=1)

    

    # Random walk sauf pour les morts : haut, bas, gauche, droite, haut-droite, haut-gauche, bas-droite, bas-gauche : 

    #CHECKPOINT 

    for j in range(0, len(liste[i])) :

        if liste[i].loc[j, 'status'] not in ["dead_other", "dead_virus"] : 

            liste[i].loc[j, 'pos_x'] += np.random.randint(-1, 2)

            liste[i].loc[j, 'pos_y'] += np.random.randint(-1, 2)

        

    

    # Si un agent sort du carré, il revient par l'autre coté : 

    

    liste[i].loc[(liste[i].status != "dead_other") & (liste[i].status != "dead_virus") & (liste[i].pos_x == grid_size+1), 'pos_x'] = 1

    liste[i].loc[(liste[i].status != "dead_other") & (liste[i].status != "dead_virus") & (liste[i].pos_x == 0), 'pos_x'] = grid_size

    liste[i].loc[(liste[i].status != "dead_other") & (liste[i].status != "dead_virus") & (liste[i].pos_y == grid_size+1), 'pos_y'] = 1

    liste[i].loc[(liste[i].status != "dead_other") & (liste[i].status != "dead_virus") & (liste[i].pos_y == 0), 'pos_y'] = grid_size

    

    

    

    

  # Si deux agents sont sur les memes coordonnées, alors le non-malade a une proba 'infectiousness' d'etre contaminé : 

    a = liste[i].loc[(liste[i].duplicated(['pos_x', 'pos_y'], keep=False))].reset_index()

    for row in range(0, (len(a)-1)) :

        for row1 in range(row+1, len(a)) : 

            if (a.loc[row, 'pos_x'] == a.loc[row1, 'pos_x']) and (a.loc[row, 'pos_y'] == a.loc[row1, 'pos_y']) : 

                if (a.loc[row, 'status'] == "clean") and (a.loc[row1, 'status'] == "sick") :

                    liste[i].loc[liste[i].agent_id == a.loc[row, 'agent_id'], 'status'] = random.choices(['sick', 'clean'], weights = [infectiousness, 1-infectiousness], k=1)

                if (a.loc[row1, 'status'] == "clean") and (a.loc[row, 'status'] == "sick") : 

                    liste[i].loc[liste[i].agent_id == a.loc[row1, 'agent_id'], 'status'] = random.choices(['sick', 'clean'], weights = [infectiousness, 1-infectiousness], k=1)

            elif (a.loc[row, 'pos_x'] in [a.loc[row1, 'pos_x']-1, a.loc[row1, 'pos_x']+1]) and (a.loc[row, 'pos_y'] in [a.loc[row1, 'pos_y']-1, a.loc[row1, 'pos_y']+1]) : 

                if (a.loc[row, 'status'] == "clean") and (a.loc[row1, 'status'] == "sick") :

                    liste[i].loc[liste[i].agent_id == a.loc[row, 'agent_id'], 'status'] = random.choices(['sick', 'clean'], weights = [infectiousness/2, 1-infectiousness], k=1)

                if (a.loc[row1, 'status'] == "clean") and (a.loc[row, 'status'] == "sick") : 

                    liste[i].loc[liste[i].agent_id == a.loc[row1, 'agent_id'], 'status'] = random.choices(['sick', 'clean'], weights = [infectiousness/2, 1-infectiousness], k=1)



df = pd.concat(liste, ignore_index=True)
#status = np.repeat(df['status'].unique(), len(np.repeat(range(0, duration_model_months+1), len(df['status'].unique())))/len(df['status'].unique()))

status = np.tile(df['status'].unique(), int(len(np.repeat(range(0, duration_model_months+1), len(df['status'].unique())))/len(df['status'].unique())))

agent_id = np.repeat(len(df.index) + 1, len(status))

pos_x = np.repeat(grid_size+1, len(status))

pos_y = np.repeat(grid_size+1, len(status))

age_months = np.repeat(0, len(status))

lifetime_months = np.repeat(1, len(status))

time = np.repeat(range(0, duration_model_months+1), len(df['status'].unique()))





data = {'agent_id': agent_id, 'pos_x': pos_x, 'pos_y': pos_y, 'status': status, 

        'age_months': age_months, 'lifetime_months': lifetime_months, 'time': time} 

df1 = pd.DataFrame(data)





df = pd.concat([df, df1], ignore_index=True)
px.scatter(df, x="pos_x", y="pos_y", animation_frame="time", animation_group="agent_id",

           color="status", hover_name="agent_id",

           log_x=False, range_x=[1,grid_size], range_y=[1,grid_size], 

          color_discrete_map = {"sick": "red", "clean": "deepskyblue", "immunized":"lawngreen", "dead_virus":"fuchsia", "dead_other":"black"})
evol = df.groupby(["time", "status"]).size().reset_index(name="count")



# Suppression des agents fictifs :

evol['count'] = evol['count']-1
px.line(evol, x="time", y="count", color='status', 

        color_discrete_map = {"sick": "red", "clean": "deepskyblue", "immunized":"lawngreen", "dead_virus":"fuchsia", "dead_other":"black"})