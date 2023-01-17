import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
import os

# Get the necessary data
my_data = pd.read_csv('../input/final-pokemon/pokemon_clean _update.csv')

# Add base stats and weakness count to our dataset
my_data['base_stats'] = my_data['attack'] + my_data['defense'] + my_data['hp'] + my_data['sp_attack'] + my_data['sp_defense'] + my_data['speed'] 
my_data['weakness_count'] =  my_data['against_flying'] + my_data['against_ghost'] + my_data['against_ground'] + my_data['against_grass'] + my_data['against_ice'] + my_data['against_normal'] + my_data['against_poison'] + my_data['against_psychic'] +  my_data['against_rock'] + my_data['against_steel'] + my_data['against_water'] + my_data['against_bug'] + my_data['against_fire'] + my_data['against_dragon'] +  my_data['against_dark'] + my_data['against_electric'] + my_data['against_fairy'] + my_data['against_fight']
my_data['mod_stats'] = (my_data['attack'] * 1.5) + my_data['defense'] + my_data['hp'] + (my_data['sp_attack'] * 1.5) + my_data['sp_defense'] + (my_data['speed'] * 2)

#weakness 0.0, .25, 1, 0.5, 2, 4.0
my_data['mod_stats_weakness'] = (my_data['attack'] * 1.5) + my_data['defense'] + my_data['hp'] + (my_data['sp_attack'] * 1.5) + my_data['sp_defense'] + (my_data['speed'] * 2)

my_data = my_data.replace(np.nan,' ',regex=True)

# We organize the data by generation
pokemon_list = []
generation_list = []
i = 0
gen_count = 1

# Lets get the max weakness and least weakness.
max_weakness = my_data['weakness_count'].max();
min_weakness = my_data['weakness_count'].min();

# Process for organizing pokemon by generation and filtering out the legendary ones
while i < len(my_data):
    if  my_data['name'][i] is not None:
        if my_data['generation'][i] == gen_count:
            # Add pokemon if not legendary
            if my_data['is_legendary'][i] == 0:
                # We now use affine transformation in order to create a weakness rating system from 1-10
                weakness_count = (my_data['weakness_count'][i] - min_weakness) * (10 - 1)/ (max_weakness - min_weakness) + 1
                
                # Dealing with pokemon weaknesses
                base_weakness = my_data['mod_stats_weakness'][i]
                big_impact = 10
                small_impact = 5
                
                # Check for weaknesses and reduce base stats if weakness is > 2
                if my_data['against_bug'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_bug'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_dark'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_dark'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_dragon'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_dragon'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_electric'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_electric'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_fairy'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_fairy'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_fight'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_fight'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_flying'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_flying'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_ghost'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_ghost'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_grass'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_grass'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_ground'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_ground'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_ice'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_ice'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_rock'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_rock'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_normal'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_normal'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_psychic'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_psychic'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_poison'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_poison'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_steel'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_steel'][i] == 2:
                    base_weakness -= small_impact
                if my_data['against_water'][i] == 4: 
                    base_weakness -= big_impact
                elif my_data['against_water'][i] == 2:
                    base_weakness -= small_impact

                # Store single pokemon record in dictionary and append to out generation_list array
                pokemon = {'name': my_data['name'][i], 'base stats': my_data['base_stats'][i], 'weakness rating': round(weakness_count, 1), 'type': str(my_data['type1'][i]) +" "+ str(my_data['type2'][i]), 'mod stats': my_data['mod_stats'][i],'true base stats': base_weakness, 'special attack': my_data['sp_attack'][i], 'attack': my_data['attack'][i]}
                generation_list.append(pokemon)
            
            if i == 800:
                tmp_pokemon_list = pd.DataFrame(generation_list)
                pokemon_list.append(tmp_pokemon_list)
                generation_list = []
        else:
            # Save the frame to pokemon_list array
            tmp_pokemon_list = pd.DataFrame(generation_list)
            pokemon_list.append(tmp_pokemon_list)
            
            # Reset generation list!
            generation_list = []
            
            # Go to the next generation
            gen_count +=1  
            i -=1
    i += 1
#lets look at the total base stats
#DataFrame.plot.bar(x=Pokemon, y=Base_Stats, pandas.DataFrame.plot().)
# Any results you write to the current directory are saved as output.
generation_1 = pokemon_list[0].sort_values(by=['true base stats'], ascending = False)
generation_1 = generation_1.drop([100]) #Drop Electrode 
generation_1
generation_1.loc[:129].plot.bar(y='true base stats', x='name')
finalists = pd.DataFrame(columns=generation_1.columns)
finalists = finalists.append(generation_1.loc[:129])
generation_2 = pokemon_list[1].sort_values(by=['true base stats'], ascending = False)
generation_2
generation_2.loc[:8].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_2.loc[:8])

generation_3 = pokemon_list[2].sort_values(by=['true base stats'], ascending = False)
generation_3
generation_3.loc[:98].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_3.loc[:98])
generation_4 = pokemon_list[3].sort_values(by=['true base stats'], ascending = False)
# Drop pokemon who do not make the cut
generation_4 = generation_4.drop([87,73,5]) 
generation_4
generation_4.loc[:32].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_4.loc[:32])
generation_5 = pokemon_list[4].sort_values(by=['true base stats'], ascending = False)
# Drop pokemon who do not make the cut
generation_5 = generation_5.drop([122]) 
generation_5
generation_5.loc[:125].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_5.loc[:125])
generation_6 = pokemon_list[5].sort_values(by=['true base stats'], ascending = False)
# Drop pokemon who do not make the cut
generation_6 = generation_6.drop([65]) 
generation_6
generation_6.loc[:5].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_6.loc[:5])
generation_7 = pokemon_list[6].sort_values(by=['true base stats'], ascending = False)
# Drop pokemon who do not make the cut
generation_7 = generation_7.drop([50]) 
generation_7
generation_7.loc[:8].plot.bar(y='true base stats', x='name')
finalists = finalists.append(generation_7.loc[:8])

finalists = finalists.sort_values(by=['true base stats'], ascending = False)
finalists

# Append the final pokemon for the BEST team!
top_6 = pd.DataFrame(columns=finalists.columns)
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Greninja'])
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Slaking'])
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Garchomp'])
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Wishiwashi'])
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Metagross'])
top_6 = top_6.append(finalists.loc[finalists['name'] == 'Arcanine'])
top_6

top_6.plot.bar(y='true base stats', x='name')
plt.rcParams['figure.figsize'] = (14, 8)