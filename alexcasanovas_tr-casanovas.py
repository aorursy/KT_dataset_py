# This Python 3 environment comes with many helpful analytics libraries installed
#Importem primeres llibreries

import numpy as np # linear algebra
import pandas as pd # data processing, llegir csv

#Importem el dataset de Fifa i li diem que ens mostri quines classes té, per després poder seleccionar el seu "path"

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Importo llibreries i sklearn, ML program.

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#Carreguem i visualtizem les dades del dataset de fifa i ho guardem dins d'una variable anomenada "data"

data = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')

data.head()

#Davant aquesta massiva taula de font de dades, n'hem d'eliminar aquelles que no són determinants per a determinar la posició d'un jugador. 
#També eliminem el seu nom i el seu ID, ja que el propòsit d'aquesta pràctica és descobrir la posició de certs jugadors mitjançant els seus atributs.





#Escollim aquelles columnes que segons el nostre criteri tenen pes en el resultat final
#Estarà tot referenciat al treball

data = data[['age',
 'height_cm',
 'weight_kg',
 'overall',
 'preferred_foot',
 'weak_foot',
 'skill_moves',
 'work_rate',
 'release_clause_eur',
 'player_positions',
 'team_jersey_number',
 'pace',
 'shooting',
 'passing',
 'dribbling',
 'defending',
 'physic',
 'gk_diving',
 'gk_handling',
 'gk_kicking',
 'gk_reflexes',
 'gk_speed',
 'gk_positioning',
 'player_traits',
 'attacking_crossing',
 'attacking_finishing',
 'attacking_heading_accuracy',
 'attacking_short_passing',
 'attacking_volleys',
 'skill_dribbling',
 'skill_curve',
 'skill_fk_accuracy',
 'skill_long_passing',
 'skill_ball_control',
 'movement_acceleration',
 'movement_sprint_speed',
 'movement_agility',
 'movement_reactions',
 'movement_balance',
 'power_shot_power',
 'power_jumping',
 'power_stamina',
 'power_strength',
 'power_long_shots',
 'mentality_aggression',
 'mentality_interceptions',
 'mentality_positioning',
 'mentality_vision',
 'mentality_penalties',
 'mentality_composure',
 'defending_marking',
 'defending_standing_tackle',
 'defending_sliding_tackle',
 'goalkeeping_diving',
 'goalkeeping_handling',
 'goalkeeping_kicking',
 'goalkeeping_positioning',
 'goalkeeping_reflexes']]



#Netejem els valors nuls (Nan= not a number) per a que no donguin resultats nuls
# i alhora assignem a la variable "player_positions" totes les posicions possibles dels jugadors
# amb les quals treballarem més tard per a quedar-nos amb la primera posició només.
data = data[-pd.isnull(data.player_positions)]
player_positions = data['player_positions']
#Utilitzem la funció str.split per a fer una separació nombrable entre les diferentes posicions dels jugadors
#En aquest cas la separació s'efectúa amb una coma extra, la qual nombrarem més tard per fer les noves columnes
#
player_positions.str.split()
#Efectuem la separació nombrant la coma extra a l'split()
#
player_positions.str.split(',')
#Referenciant novament la coma per crear la separació entre columnes, creem un dataframe amb la funció expand, compatible
#amb el dataframe. D'aquesta manera podrem nombra les 2es i 3es columnes per a eliminar-les i quedar-nos només amb una posició
#
player_positions.str.split(',', expand=True)
#Li donem un valor de variable a la taula per a poder-la manipular.
#De la variable (per tant del dataframe) ens quedem amb la columna 0 (en programació, la primera)
#
data_position = player_positions.str.split(',', expand=True)
player_position = data_position[0]

player_position
#afegim la columna resultant al dataframe s de Fifa sencer com una nova columna
#
data['player_position'] = player_position

data
#Com al dataframe encara hi ha la columna player_positions, que es la que volem modificar, l'hem
#d'extreure i posar la nova: player_position
data = data[['age',
 'height_cm',
 'weight_kg',
 'overall',
 'preferred_foot',
 'weak_foot',
 'skill_moves',
 'work_rate',
 'release_clause_eur',
 'player_position',
 'team_jersey_number',
 'pace',
 'shooting',
 'passing',
 'dribbling',
 'defending',
 'physic',
 'gk_diving',
 'gk_handling',
 'gk_kicking',
 'gk_reflexes',
 'gk_speed',
 'gk_positioning',
 'player_traits',
 'attacking_crossing',
 'attacking_finishing',
 'attacking_heading_accuracy',
 'attacking_short_passing',
 'attacking_volleys',
 'skill_dribbling',
 'skill_curve',
 'skill_fk_accuracy',
 'skill_long_passing',
 'skill_ball_control',
 'movement_acceleration',
 'movement_sprint_speed',
 'movement_agility',
 'movement_reactions',
 'movement_balance',
 'power_shot_power',
 'power_jumping',
 'power_stamina',
 'power_strength',
 'power_long_shots',
 'mentality_aggression',
 'mentality_interceptions',
 'mentality_positioning',
 'mentality_vision',
 'mentality_penalties',
 'mentality_composure',
 'defending_marking',
 'defending_standing_tackle',
 'defending_sliding_tackle',
 'goalkeeping_diving',
 'goalkeeping_handling',
 'goalkeeping_kicking',
 'goalkeeping_positioning',
 'goalkeeping_reflexes']]

data
data['player_position'].replace(['RWB', 'LAM', 'CDM', 'ST', 'LM', 'GK', 'CM', 'RB', 'LWB', 'RW', 'CB', 'CAM', 'LS', 'LW', 'LCM', 'LB', 'LDM', 'LF', 'RM', 'RF', 
                                  'RDM', 'RCB', 'RAM', 'RS', 'LCB', 'CF', 'RCM'],['DEF','MC','MC','DEL','MC','GK','MC','DEF','DEF','DEL',
                                                                                 'DEF', 'MC','DEL','DEL','MC','DEF','MC','DEL','MC','DEL','MC','DEF','MC','DEL',
                                                                                 'DEF','DEL','MC'])

general_position = data['player_position'].replace(['RWB', 'LAM', 'CDM', 'ST', 'LM', 'GK', 'CM', 'RB', 'LWB', 'RW', 'CB', 'CAM', 'LS', 'LW', 'LCM', 'LB', 'LDM', 'LF', 'RM', 'RF', 
                                  'RDM', 'RCB', 'RAM', 'RS', 'LCB', 'CF', 'RCM'],['DEF','MC','MC','DEL','MC','GK','MC','DEF','DEF','DEL',
                                                                                 'DEF', 'MC','DEL','DEL','MC','DEF','MC','DEL','MC','DEL','MC','DEF','MC','DEL',
                                                                                 'DEF','DEL','MC'])

data['general_position'] = general_position

data
data = data[['age',
 'weight_kg',
 'overall',
 'pace',
 'shooting',
 'passing',
 'dribbling',
 'defending',
 'physic',
 'attacking_crossing',
 'attacking_finishing',
 'attacking_heading_accuracy',
 'attacking_short_passing',
 'attacking_volleys',
 'skill_dribbling',
 'skill_curve',
 'skill_fk_accuracy',
 'skill_long_passing',
 'skill_ball_control',
 'movement_acceleration',
 'movement_sprint_speed',
 'movement_agility',
 'movement_reactions',
 'movement_balance',
 'power_shot_power',
 'power_jumping',
 'power_stamina',
 'power_strength',
 'power_long_shots',
 'mentality_aggression',
 'mentality_interceptions',
 'mentality_positioning',
 'mentality_vision',
 'mentality_penalties',
 'mentality_composure',
 'defending_marking',
 'defending_standing_tackle',
 'defending_sliding_tackle',
 'goalkeeping_diving',
 'goalkeeping_handling',
 'goalkeeping_kicking',
 'goalkeeping_positioning',
 'goalkeeping_reflexes',
 'general_position']]

data
data = data.fillna(0)

data


X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

print(Y)
plt.xlabel('Atributs')
plt.ylabel('Posició')



pltX = data.loc[:, 'age']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='age')


pltX = data.loc[:, 'weight_kg']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='weight_kg')

pltX = data.loc[:, 'overall']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='overall')

pltX = data.loc[:, 'pace']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='pace')

pltX = data.loc[:, 'shooting']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='shooting')

pltX = data.loc[:, 'passing']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='passing')

pltX = data.loc[:, 'dribbling']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='dribbling')

pltX = data.loc[:, 'defending']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='defending')

pltX = data.loc[:, 'physic']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='physic')


pltX = data.loc[:, 'attacking_crossing']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='attacking_crossing')

pltX = data.loc[:, 'attacking_finishing']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='attacking_finishing')

pltX = data.loc[:, 'attacking_heading_accuracy']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='attacking_heading_accuracy')

pltX = data.loc[:, 'attacking_short_passing']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='attacking_short_passing')

pltX = data.loc[:, 'attacking_volleys']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='attacking_volleys')

pltX = data.loc[:, 'skill_dribbling']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='skill_dribbling')

pltX = data.loc[:, 'skill_curve']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='skill_curve')

pltX = data.loc[:, 'skill_fk_accuracy']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='skill_fk_accuracy')

pltX = data.loc[:, 'skill_long_passing']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='skill_long_passing')

pltX = data.loc[:, 'skill_ball_control']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='skill_ball_control')

pltX = data.loc[:, 'movement_acceleration']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='movement_acceleration')

pltX = data.loc[:, 'movement_sprint_speed']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='movement_sprint_speed')

pltX = data.loc[:, 'movement_agility']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='movement_agility')

pltX = data.loc[:, 'movement_reactions']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='movement_reactions')

pltX = data.loc[:, 'movement_balance']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='movement_balance')

pltX = data.loc[:, 'power_shot_power']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='power_shot_power')

pltX = data.loc[:, 'power_jumping']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='power_jumping')

pltX = data.loc[:, 'power_stamina']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='power_stamina')

pltX = data.loc[:, 'power_strength']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='power_strength')

pltX = data.loc[:, 'power_long_shots']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='power_long_shots')

pltX = data.loc[:, 'mentality_aggression']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='mentality_aggression')

pltX = data.loc[:, 'mentality_interceptions']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='mentality_interceptions')

pltX = data.loc[:, 'mentality_positioning']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='mentality_positioning')

pltX = data.loc[:, 'mentality_vision']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='mentality_vision')

pltX = data.loc[:, 'mentality_penalties']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='mentality_penalties')

pltX = data.loc[:, 'mentality_composure']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='mentality_composure')

pltX = data.loc[:, 'defending_marking']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='defending_marking')

pltX = data.loc[:, 'defending_standing_tackle']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='defending_standing_tackle')

pltX = data.loc[:, 'defending_sliding_tackle']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='defending_sliding_tackle')

pltX = data.loc[:, 'goalkeeping_diving']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='goalkeeping_diving')

pltX = data.loc[:, 'goalkeeping_handling']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='goalkeeping_handling')

pltX = data.loc[:, 'goalkeeping_kicking']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='goalkeeping_kicking')

pltX = data.loc[:, 'goalkeeping_positioning']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='goalkeeping_positioning')

pltX = data.loc[:, 'goalkeeping_reflexes']
pltY = data.loc[:,'general_position']
plt.scatter(pltX, pltY, label='goalkeeping_reflexes')


plt.legend(loc=4, prop={'size':8})
plt.show()




x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=500000)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(predictions)
print(y_test)


print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))