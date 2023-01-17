import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
df = pd.read_csv('../input/fifa-complete.csv')
df.drop(['ID', 'full_name', 'club_logo', 'flag', 'photo'], axis = 1, inplace = True)
df = df[:3]
def plot_players_radar_graph(angles, labels, first_pl_stats, second_pl_stats, num_players, third_pl_stats, mytitle):
    
    fig = plt.figure(figsize=(15,15))
    ax  = fig.add_subplot(111, polar=True)   # Set polar axis
    
    ax.plot(angles, first_pl_stats, 'o-', linewidth=2)  # Draw the plot (or the frame on the radar chart)
    ax.fill(angles, first_pl_stats, alpha=0.15)  #Fulfill the area

    ax.plot(angles, second_pl_stats, ':', linewidth=2, c='m')  # Draw the plot (or the frame on the radar chart)
    ax.fill(angles, second_pl_stats, alpha=0.10, c='m')  #Fulfill the area

    if num_players == 3:
        ax.plot(angles, third_pl_stats, '--', linewidth=2, c='g')  # Draw the plot (or the frame on the radar chart)
        ax.fill(angles, third_pl_stats, alpha=0.05, c='g')  #Fulfill the area

    ax.set_thetagrids(angles * 180/np.pi, labels)  # Set the label for each axis
    ax.tick_params(pad = 25, direction = 'out', labelsize = 'large')

    ax.set_title(mytitle, pad = 30, weight = 'bold')#[df.loc[386,"Name"]])  # Set the pokemon's name as the title

    ax.grid(True)
labels = np.array(['overall', 'potential', 'pac', 'sho', 'pas', 'dri', 'def', 'phy', 'crossing', 'finishing', 'heading_accuracy', 'short_passing', 'long_passing', 'dribbling', 'curve', 'free_kick_accuracy', 'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 'aggression', 'interceptions', 'positioning', 'vision', 'penalties', 'composure'])

ronaldo_stats = df.loc[0,labels].values
messi_stats = df.loc[1,labels].values
neymar_stats = df.loc[2,labels].values

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False) # Set the angle

# close the plot
angles = np.concatenate((angles, [angles[0]]))  # Closed

ronaldo_stats  = np.concatenate((ronaldo_stats,[ronaldo_stats[0]]))  # Closed
messi_stats  = np.concatenate((messi_stats,[messi_stats[0]]))  # Closed
neymar_stats  = np.concatenate((neymar_stats,[neymar_stats[0]]))  # Closed

plot_players_radar_graph( angles= angles,
                          labels= labels,
                          first_pl_stats = ronaldo_stats,
                          second_pl_stats = messi_stats,
                          num_players = 3,
                          third_pl_stats = neymar_stats,
                          mytitle = 'Ronaldo [Blue]  |  Messi [Purple]  |  Neymar [Green]  Skills Statistics')
pos_labels = ['rs', 'rw', 'rf', 'ram', 'rcm', 'rm', 'rdm', 'rcb', 'rb', 'rwb', 'st', 'lw', 'cf', 'cam', 'cm', 'lm', 'cdm', 'cb', 'lb', 'lwb', 'ls', 'lf', 'lam', 'lcm', 'ldm', 'lcb']

ronaldo_pos_stats = df.loc[0,pos_labels].values
messi_pos_stats = df.loc[1,pos_labels].values
neymar_pos_stats = df.loc[2,pos_labels].values

angles = np.linspace(0, 2 * np.pi, len(pos_labels), endpoint=False) # Set the angle

# close the plot
angles = np.concatenate((angles, [angles[0]]))  # Closed

ronaldo_pos_stats  = np.concatenate((ronaldo_pos_stats,[ronaldo_pos_stats[0]]))  # Closed
messi_pos_stats  = np.concatenate((messi_pos_stats,[messi_pos_stats[0]]))  # Closed
neymar_pos_stats  = np.concatenate((neymar_pos_stats,[neymar_pos_stats[0]]))  # Closed

plot_players_radar_graph( angles= angles,
                          labels= pos_labels,
                          first_pl_stats = ronaldo_pos_stats,
                          second_pl_stats = messi_pos_stats,
                          num_players = 3,
                          third_pl_stats = neymar_pos_stats,
                          mytitle = 'Ronaldo [Blue]  |  Messi [Purple]  |  Neymar [Green]  Positions Statistics')
