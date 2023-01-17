 %matplotlib inline



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sqlite3

from numpy import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Siguiendo el paso 2 de CRISP-DM, efectuamos un conteo de las filas que contienen al menos un NULL

# en la base de datos. El número de filas por tabla viene dado en la descripción de la base de

# datos, por lo que no hay que calcularlo.

with sqlite3.connect('../input/database.sqlite') as con:

    player_attributes_null_amount = pd.read_sql_query("SELECT COUNT (*) from Player_Attributes WHERE id IS NULL OR player_fifa_api_id IS NULL OR player_api_id IS NULL OR date IS NULL OR overall_rating IS NULL OR potential IS NULL OR preferred_foot IS NULL OR attacking_work_rate IS NULL OR defensive_work_rate IS NULL OR crossing IS NULL OR finishing IS NULL OR heading_accuracy IS NULL OR short_passing IS NULL OR volleys IS NULL OR dribbling IS NULL OR curve IS NULL OR free_kick_accuracy IS NULL OR long_passing IS NULL OR ball_control IS NULL OR acceleration IS NULL OR sprint_speed IS NULL OR agility IS NULL OR reactions IS NULL OR balance IS NULL OR shot_power IS NULL OR jumping IS NULL OR stamina IS NULL OR strength IS NULL OR long_shots IS NULL OR aggression IS NULL OR interceptions IS NULL OR positioning IS NULL OR vision IS NULL OR penalties IS NULL OR marking IS NULL OR standing_tackle IS NULL OR sliding_tackle IS NULL OR gk_diving IS NULL OR gk_handling IS NULL OR gk_kicking IS NULL OR gk_positioning IS NULL OR gk_reflexes IS NULL", con)

    player_null_amount = pd.read_sql_query("SELECT COUNT (*) from Player WHERE id IS NULL OR player_api_id IS NULL OR player_name IS NULL OR player_fifa_api_id IS NULL OR birthday IS NULL OR height IS NULL OR weight IS NULL", con)

    match_null_amount = pd.read_sql_query("SELECT COUNT (*) from Match WHERE id IS NULL OR country_id IS NULL OR league_id IS NULL OR season IS NULL OR stage IS NULL OR date IS NULL OR match_api_id IS NULL OR home_team_api_id IS NULL OR away_team_api_id IS NULL OR home_team_goal IS NULL OR away_team_goal IS NULL OR home_player_X1 IS NULL OR home_player_X2 IS NULL OR home_player_X3 IS NULL OR home_player_X4 IS NULL OR home_player_X5 IS NULL OR home_player_X6 IS NULL OR home_player_X7 IS NULL OR home_player_X8 IS NULL OR home_player_X9 IS NULL OR home_player_X10 IS NULL OR home_player_X11 IS NULL OR away_player_X1 IS NULL OR away_player_X2 IS NULL OR away_player_X3 IS NULL OR away_player_X4 IS NULL OR away_player_X5 IS NULL OR away_player_X6 IS NULL OR away_player_X7 IS NULL OR away_player_X8 IS NULL OR away_player_X9 IS NULL OR away_player_X10 IS NULL OR away_player_X11 IS NULL OR home_player_Y1 IS NULL OR home_player_Y2 IS NULL OR home_player_Y3 IS NULL OR home_player_Y4 IS NULL OR home_player_Y5 IS NULL OR home_player_Y6 IS NULL OR home_player_Y7 IS NULL OR home_player_Y8 IS NULL OR home_player_Y9 IS NULL OR home_player_Y10 IS NULL OR home_player_Y11 IS NULL OR away_player_Y1 IS NULL OR away_player_Y2 IS NULL OR away_player_Y3 IS NULL OR away_player_Y4 IS NULL OR away_player_Y5 IS NULL OR away_player_Y6 IS NULL OR away_player_Y7 IS NULL OR away_player_Y8 IS NULL OR away_player_Y9 IS NULL OR away_player_Y10 IS NULL OR away_player_Y11 IS NULL OR home_player_1 IS NULL OR home_player_2 IS NULL OR home_player_3 IS NULL OR home_player_4 IS NULL OR home_player_5 IS NULL OR home_player_6 IS NULL OR home_player_7 IS NULL OR home_player_8 IS NULL OR home_player_9 IS NULL OR home_player_10 IS NULL OR home_player_11 IS NULL OR away_player_1 IS NULL OR away_player_2 IS NULL OR away_player_3 IS NULL OR away_player_4 IS NULL OR away_player_5 IS NULL OR away_player_6 IS NULL OR away_player_7 IS NULL OR away_player_8 IS NULL OR away_player_9 IS NULL OR away_player_10 IS NULL OR away_player_11 IS NULL OR goal IS NULL OR shoton IS NULL OR shotoff IS NULL OR foulcommit IS NULL OR card IS NULL OR cross IS NULL OR corner IS NULL OR possession IS NULL OR B365H IS NULL OR B365D IS NULL OR B365A IS NULL OR BWH IS NULL OR BWD IS NULL OR BWA IS NULL OR IWH IS NULL OR IWD IS NULL OR IWA IS NULL OR LBH IS NULL OR LBD IS NULL OR LBA IS NULL OR PSH IS NULL OR PSD IS NULL OR PSA IS NULL OR WHH IS NULL OR WHD IS NULL OR WHA IS NULL OR SJH IS NULL OR SJD IS NULL OR SJA IS NULL OR VCH IS NULL OR VCD IS NULL OR VCA IS NULL OR GBH IS NULL OR GBD IS NULL OR GBA IS NULL OR BSH IS NULL OR BSD IS NULL OR BSA IS NULL", con)

    league_null_amount = pd.read_sql_query("SELECT COUNT (*) from League WHERE id IS NULL OR country_id IS NULL OR name IS NULL", con)

    country_null_amount = pd.read_sql_query("SELECT COUNT (*) from Country WHERE id IS NULL or name IS NULL", con)

    team_null_amount = pd.read_sql_query("SELECT COUNT (*) from Team WHERE id IS NULL OR team_api_id IS NULL OR team_fifa_api_id IS NULL OR team_long_name IS NULL OR team_short_name IS NULL", con)

    team_attributes_null_amount = pd.read_sql_query("SELECT COUNT (*) from Team_Attributes WHERE id IS NULL OR team_fifa_api_id IS NULL OR team_api_id IS NULL OR date IS NULL OR buildUpPlaySpeed IS NULL OR buildUpPlaySpeedClass IS NULL OR buildUpPlayDribbling IS NULL OR buildUpPlayDribblingClass IS NULL OR buildUpPlayPassing IS NULL OR buildUpPlayPassingClass IS NULL OR buildUpPlayPositioningClass IS NULL OR chanceCreationPassing IS NULL OR chanceCreationPassingClass IS NULL OR chanceCreationCrossing IS NULL OR chanceCreationCrossingClass IS NULL OR chanceCreationShooting IS NULL OR chanceCreationShootingClass IS NULL OR chanceCreationPositioningClass IS NULL OR defencePressure IS NULL OR defencePressureClass IS NULL OR defenceAggression IS NULL OR defenceAggressionClass IS NULL OR defenceTeamWidth IS NULL OR defenceTeamWidthClass IS NULL OR defenceDefenderLineClass IS NULL", con)

con.close()

print ("Número de filas (de 183978) con algún NULL en la tabla Player_Attributes:")    

print (player_attributes_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 11060) con algún NULL en la tabla Player:")    

print (player_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 25979) con algún NULL en la tabla Match:")    

print (match_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 11) con algún NULL en la tabla League:")    

print (league_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 11) con algún NULL en la tabla Countries:")    

print (country_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 299) con algún NULL en la tabla Team:")    

print (team_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 1458) con algún NULL en la tabla Team_Attributes:")    

print (team_attributes_null_amount.head())

print ("*************************************************************************")
with sqlite3.connect('../input/database.sqlite') as con:

    player_attributes_all = pd.read_sql_query("SELECT * from Player_Attributes", con)

    player_all = pd.read_sql_query("SELECT * from Player", con)

    match_all = pd.read_sql_query("SELECT * from Match", con)

    league_all = pd.read_sql_query("SELECT * from League", con)

    country_all = pd.read_sql_query("SELECT * from Country", con)

    team_all = pd.read_sql_query("SELECT * from Team", con)

    team_attributes_all = pd.read_sql_query("SELECT * from Team_Attributes", con)

con.close()



print ("Aspecto del contenido de la tabla Player_Attributes:\n")

print (player_attributes_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla Player:\n")    

print (player_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla Match:\n")    

print (match_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla League:\n")    

print (league_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla Countries:\n")    

print (country_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla Team:\n")    

print (team_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla Team_Attributes:\n")    

print (team_attributes_all.head())

print ("*************************************************************************")

leagues_per_country = country_all.merge(league_all,on='id',suffixes=('', '_y'))

print(leagues_per_country)
matches_per_league = match_all[match_all.league_id.isin(leagues_per_country.id)]
# Se seleccionan los campos adecuados para calcular la predecibilidad

matches_per_league = matches_per_league[['id', 'country_id' ,'league_id', 'season', 'stage', 'date','match_api_id', 'home_team_api_id', 'away_team_api_id','B365H', 'B365D' ,'B365A']]



# No se tienen en cuenta filas en las que falten valores

matches_per_league.dropna(inplace=True)



from scipy.stats import entropy



def match_entropy(row):



    odds = [row['B365H'],row['B365D'],row['B365A']]



    # Se transforman los valores de las columnas de apuestas en probabilidades



    probs = [1/o for o in odds]



    # Se normaliza



    norm = sum(probs)



    probs = [p/norm for p in probs]



    return entropy(probs)





# Se calcula la entropía de los partidos



matches_per_league['entropy'] = matches_per_league.apply(match_entropy,axis=1)



# Se calcula la entropía media para cada liga en cada temporada

entropy_means = matches_per_league.groupby(('season','league_id')).entropy.mean()



entropy_means = entropy_means.reset_index().pivot(index='season', columns='league_id', values='entropy')



entropy_means.columns = [leagues_per_country[leagues_per_country.id==x].name.values[0] for x in entropy_means.columns]



# Se echa un vistazo a la tabla de entropías medias recién calculadas

entropy_means.head()
# Se dibuja la gráfica

ax = entropy_means.plot(figsize=(12,8),marker='o')



plt.title('Predecibilidad de ligas', fontsize=16)



plt.xticks(rotation=50)



colors = [x.get_color() for x in ax.get_lines()]

colors_mapping = dict(zip(leagues_per_country.id,colors))



ax.set_xlabel('')



plt.legend(loc='lower left')



ax.annotate('', xytext=(7.2, 1),xy=(7.2, 1.039),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('', xytext=(7.2, 0.96),xy=(7.2, 0.921),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('Menos predecible', xy=(7.3, 1.028), annotation_clip=False,fontsize=14,rotation='vertical')

ax.annotate('Más predecible', xy=(7.3, 0.952), annotation_clip=False,fontsize=14,rotation='vertical')