import numpy as np
import pandas as pd
from math import pi

import matplotlib.pyplot as plt
class Similarity():
    
    def __init__(self, name):
        self.player_name = name
    
    def PlayerList(self):    
        data = pd.read_csv('../input/fifa19/data.csv') 
        data = data[data['Overall'] > 82] # Lower Overall
        attributes = ['Name','Nationality','Club','Age','Position','Overall','Potential','Preferred Foot','Value']
        abilities = ['Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys',
                 'Dribbling','Curve','FKAccuracy','LongPassing','BallControl',
                 'Acceleration','SprintSpeed','Agility','Reactions','Balance',
                 'ShotPower','Jumping','Stamina','Strength','LongShots',
                 'Aggression','Interceptions','Positioning','Vision','Penalties',
                 'Composure','Marking','StandingTackle','SlidingTackle']
   
        # Unit directional vector
        AbilitiesData = data[abilities]
        vec_length = np.sqrt(np.square(AbilitiesData).sum(axis=1))
        mat_abt = AbilitiesData.values.reshape(AbilitiesData.shape)
        
        for i in np.arange(AbilitiesData.shape[0]):
                mat_abt[i] = mat_abt[i,:]/vec_length[i]
                
        df_norm = pd.DataFrame(mat_abt, columns=abilities) 
    
        # Inner Product
        compared_player = df_norm[data['Name'] == self.player_name].iloc[0]
        
        data['Inner Product'] = df_norm.dot(compared_player)
        
        threshold_idp = 0.991 
        lower_potential = 85 # High potential
        substitutes = data[(data['Inner Product'] >= threshold_idp) & (data['Potential'] >= lower_potential)]
        
        if substitutes.shape[0] <= 1:
            print('There is no recommendation.')
            
        else:    
            substitutes = substitutes.sort_values(by=['Inner Product'], ascending=False)
            
            # Maximum of Player Recommendations = 3 players
            if substitutes.shape[0] > 4:
                substitutes = substitutes[0:4]
                
            substitutes = substitutes[attributes]
            substitutes.reset_index(drop=True)
            
            # Save the Scout list
            substitutes.to_csv('./scout_list.csv', index=False)
            
            standard_player = data[abilities][data.Name == self.player_name]
            
            for player_list in substitutes['Name'][1:]:
                
                add = data[abilities][data.Name == player_list]
                standard_player = standard_player.append([add])

            player_name = substitutes['Name'].values
            
            return standard_player, abilities, player_name
                   
def RadorChart(graph, abilities, player_name):
    len1 = graph.shape[0]
    len2 = graph.shape[1]
    temp = graph.values.reshape((len1, len2))
    
    tmp = pd.DataFrame(temp, columns = abilities)
    Attributes =list(tmp)
    AttNo = len(Attributes)
    
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111, polar=True)
    
    colors = ['green', 'blue', 'red', 'gold', 'orange', 'lightskyblue', 'black', 'pink']
    
    for i in range(len1):
        values = tmp.iloc[i].tolist() #
        values += values [:1]
    
        angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
        angles += angles [:1]
        
        plt.xticks(angles[:-1],Attributes)
        ax.plot(angles, values, color = colors[i])
        ax.fill(angles, values, colors[i], alpha=0.1)
        plt.figtext(0.8, 0.25-0.025*i, player_name[i], color = colors[i], fontsize=20)
    
    plt.savefig('RadarChart.png')
    plt.show()
    
Scouter = Similarity('L. Messi')
players, abilities, player_names = Scouter.PlayerList()
abilities_view = ['PAS','SHO','SPE','PHY','DRI','DEF']
players['PAS'] = (players['ShortPassing']+players['LongPassing']+players['Crossing'])//3
players['SHO'] = (players['ShotPower']+players['LongShots']+players['Finishing'])//3
players['SPE'] = (players['SprintSpeed']+players['Acceleration']+players['Agility'])//3 
players['PHY'] = (players['Stamina']+players['Strength']+players['Balance']+players['Reactions'])//4
players['DRI'] = (players['Dribbling']+players['BallControl'])//2
players['DEF'] = (players['Marking']+players['StandingTackle']+players['Interceptions'])//3
RadorChart(players[abilities_view], abilities_view, player_names)
