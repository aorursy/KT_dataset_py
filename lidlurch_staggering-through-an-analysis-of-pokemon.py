#invite all my peeps to the Kaggle party

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



#from bokeh.charts import Bar, TimeSeries, output_file, show, output_notebook

#from bokeh.models import HoverTool, ColumnDataSource

#output_notebook()



pkm = pd.read_csv('../input/pokemon.csv')
pkm.head()
#rearrange the pokemon columns into a more categorical order: names, basic stats, abilities, characteristics

#getting rid of the japanese name, base_egg_steps, base_happiness, weight, height, percentage_male

#... don't really need that in this case

pkm =    pkm        [[ #Name

                      'name', 

                      'pokedex_number', 

                      'type1', 

                      'type2', 

                      #Basic Stats

                      'hp', 

                      'attack',

                      'defense',

                      'sp_attack',

                      'sp_defense',

                      'speed', 

                      'base_total', #all the previous stats added together

                      'abilities',

                      #Attributes

                      'generation',

                      'is_legendary',

                      #"Against" attributes

                      'against_bug',

                      'against_dark',

                      'against_dragon',

                      'against_electric',

                      'against_fairy',

                      'against_fight',

                      'against_fire',

                      'against_flying',

                      'against_ghost',

                      'against_grass',

                      'against_ground',

                      'against_ice',

                      'against_normal',

                      'against_poison',

                      'against_psychic',

                      'against_rock',

                      'against_steel',

                      'against_water']]#.head(151) #we're only counting the Pokemon that really matter.

pkm.head()
pkm['against_average']= (pkm['against_bug']+

                      pkm['against_dark']+

                      pkm['against_dragon']+

                      pkm['against_electric']+

                      pkm['against_fairy']+

                      pkm['against_fight']+

                      pkm['against_fire']+

                      pkm['against_flying']+

                      pkm['against_ghost']+

                      pkm['against_grass']+

                      pkm['against_ground']+

                      pkm['against_ice']+

                      pkm['against_normal']+

                      pkm['against_poison']+

                      pkm['against_psychic']+

                      pkm['against_rock']+

                      pkm['against_steel']+

                      pkm['against_water'])/18
against_avg_srt = pkm.sort_values('against_average', ascending = False)

against_avg_srt[['name', 'against_average']].head()
#plot using the Seaborn "swarmplot"

sns.swarmplot(x='type1', y='against_average', data=against_avg_srt

           ,dodge=True

           , size=7

           #,fit_reg = False

           ,hue = 'type1'

          )

#rotate the tick marks and move the lengend over

plt.xticks(rotation=-45)

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
pkmn_type_colors = ['#B8A038',  # Rock

                    '#78C850',  # Grass

                    '#A8B820',  # Bug

                    '#E0C068',  # Ground

                    '#F85888',  # Psychic

                    '#705848',  # Dark

                    '#98D8D8',  # Ice

                    '#C03028',  # Fighting

                    '#F08030',  # Fire

                    '#7038F8',  # Dragon

                    '#6890F0',  # Water

                    '#A8A878',  # Normal

                    '#A890F0',  # Flying

                    '#A040A0',  # Poison

                    '#B8B8D0',  # Steel

                    '#705898',  # Ghost 

                    '#EE99AC',  # Fairy

                    '#F8D030',  # Electric

                   ]
sns.set_style("whitegrid")

plt.figure(figsize=(16,10))

#plt.ylim(0, 1.5)

sns.swarmplot(x='type1', y='against_average', data=against_avg_srt 

              ,palette = pkmn_type_colors

              ,dodge = True

              ,size = 7

           )

plt.xticks(rotation=-45)

#plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);
sns.set_style("whitegrid")

plt.figure(figsize=(16,10))

#plt.ylim(0, 1.5)

sns.swarmplot(x='type1', y='base_total', data=against_avg_srt 

              ,palette = pkmn_type_colors

              ,dodge = True

              ,size = 7

           )

plt.xticks(rotation=-45)

#plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);

base_total_srt = pkm.sort_values('base_total', ascending = False)

base_total_srt[['name', 'base_total']].head()
pkm =           pkm[[ #Name

                      'name', 

                      'pokedex_number', 

                      'type1', 

                      'type2', 

                      #Basic Stats

                      'hp', 

                      'attack',

                      'defense',

                      'sp_attack',

                      'sp_defense',

                      'speed', 

                      'base_total', #all the previous stats added together

                      'abilities',

                      #Attributes

                      'generation',

                      'is_legendary'

                    ]]
pkm['abilities_cnt']= pkm['abilities'].count()
pkm['abilities_cnt']