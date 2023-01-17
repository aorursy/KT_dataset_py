import json

from pandas.io.json import json_normalize

import pandas as pd

with open('../input/FIFA_France_v_Belgium_2018.json') as data_file:    

    data = json.load(data_file)

df = json_normalize(data, sep = "_")
df
# import relevant libraries

%matplotlib inline

import json

from pandas.io.json import json_normalize

import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.patches import Arc, Rectangle, ConnectionPatch

from matplotlib.offsetbox import  OffsetImage

from functools import reduce



def draw_pitch(ax):

    # focus on only half of the pitch

    #Pitch Outline & Centre Line

    Pitch = Rectangle([0,0], width = 120, height = 80, fill = False)

    #Left, Right Penalty Area and midline

    LeftPenalty = Rectangle([0,22.3], width = 14.6, height = 35.3, fill = False)

    RightPenalty = Rectangle([105.4,22.3], width = 14.6, height = 35.3, fill = False)

    midline = ConnectionPatch([60,0], [60,80], "data", "data")



    #Left, Right 6-yard Box

    LeftSixYard = Rectangle([0,32], width = 4.9, height = 16, fill = False)

    RightSixYard = Rectangle([115.1,32], width = 4.9, height = 16, fill = False)





    #Prepare Circles

    centreCircle = plt.Circle((60,40),8.1,color="black", fill = False)

    centreSpot = plt.Circle((60,40),0.71,color="black")

    #Penalty spots and Arcs around penalty boxes

    leftPenSpot = plt.Circle((9.7,40),0.71,color="black")

    rightPenSpot = plt.Circle((110.3,40),0.71,color="black")

    leftArc = Arc((9.7,40),height=16.2,width=16.2,angle=0,theta1=310,theta2=50,color="black")

    rightArc = Arc((110.3,40),height=16.2,width=16.2,angle=0,theta1=130,theta2=230,color="black")

    

    element = [Pitch, LeftPenalty, RightPenalty, midline, LeftSixYard, RightSixYard, centreCircle, 

               centreSpot, rightPenSpot, leftPenSpot, leftArc, rightArc]

    for i in element:

        ax.add_patch(i)
fig=plt.figure() #set up the figures

fig.set_size_inches(7, 5)

ax=fig.add_subplot(1,1,1)

draw_pitch(ax) #overlay our different objects on the pitch

plt.ylim(-2, 82)

plt.xlim(-2, 122)

plt.axis('off')

plt.show()
def player_pass_plot(data, player_name):

    player_pass = df[(df['type_name'] == "Pass") & (df['player_name']==player_name)] # get passing information of player

    pass_column = [i for i in df.columns if i.startswith("pass")]

    player_pass = player_pass[["id", "period", "timestamp", "location", "pass_end_location", "pass_recipient_name"]]

    fig, ax = plt.subplots()

    fig.set_size_inches(7, 5)

    ax.set_xlim([0,120])

    ax.set_ylim([0,80])

    draw_pitch(ax) #overlay our different objects on the pitch

    for i in range(len(player_pass)):

    # can also differentiate by color

        color = "blue" if player_pass.iloc[i]['period'] == 1 else "red"

        ax.annotate("", xy = (player_pass.iloc[i]['pass_end_location'][0], player_pass.iloc[i]['pass_end_location'][1]), xycoords = 'data',

               xytext = (player_pass.iloc[i]['location'][0], player_pass.iloc[i]['location'][1]), textcoords = 'data',

               arrowprops=dict(arrowstyle="->",connectionstyle="arc3", color = "blue"),)

    plt.show()
player_pass_plot(data, 'Paul Pogba')
def player_movt_heat_map(data, player_name):

    fig, ax = plt.subplots()

    fig.set_size_inches(7, 5)

    player_action = df[(df['player_name']==player_name)] # get movement information of Pogba\

    x_coord = [i[0] for i in player_action["location"]]

    y_coord = [i[1] for i in player_action["location"]]

    #shades: give us the heat map we desire

    # n_levels: draw more lines, the larger n, the more blurry it looks

    ax=fig.add_subplot(1,1,1)

    draw_pitch(ax)

    sns.kdeplot(x_coord, y_coord, shade = "True", color = "blue", n_levels=30)

    plt.show()
player_movt_heat_map(df, 'Paul Pogba')
def heat_pass_map(data, player_name):

    pass_data = data[(data['type_name'] == "Pass") & (data['player_name'] == player_name)]

    action_data = data[(data['player_name']==player_name)]

    

    fig, ax = plt.subplots()

    fig.set_size_inches(7, 5)

    ax.set_xlim([0,120])

    ax.set_ylim([0,80])

    draw_pitch(ax)



    for i in range(len(pass_data)):

        # we also differentiate different half by different color

        color = "blue" if pass_data.iloc[i]['period'] == 1 else "red"

        ax.annotate("", xy = (pass_data.iloc[i]['pass_end_location'][0], pass_data.iloc[i]['pass_end_location'][1]), xycoords = 'data',

               xytext = (pass_data.iloc[i]['location'][0], pass_data.iloc[i]['location'][1]), textcoords = 'data',

               arrowprops=dict(arrowstyle="->",connectionstyle="arc3", color = color),)

    x_coord = [i[0] for i in action_data["location"]]

    y_coord = [i[1] for i in action_data["location"]]

    sns.kdeplot(x_coord, y_coord, shade = "True", color = "green", n_levels = 30)

    plt.ylim(0, 80) # need this, otherwise kde plot will go outside

    plt.xlim(0, 120)

    plt.show()
heat_pass_map(df,'Paul Pogba')
def team_shot_chart(data, team_name):

    shot_data = data[(data['type_name'] == "Shot") & (data['team_name'] == team_name)]

   

    fig, ax = plt.subplots()

    fig.set_size_inches(7, 5)

    ax.set_xlim([0,120])

    ax.set_ylim([0,80])

    draw_pitch(ax)

    

    for i in range(len(shot_data)):

        color = "red" if shot_data.iloc[i]['shot_outcome_name'] == "Goal" else "black"

        ax.annotate("", xy = (shot_data.iloc[i]['shot_end_location'][0], shot_data.iloc[i]['shot_end_location'][1]), xycoords = 'data',

           xytext = (shot_data.iloc[i]['location'][0], shot_data.iloc[i]['location'][1]), textcoords = 'data',

           arrowprops=dict(arrowstyle="->",connectionstyle="arc3", color = color),)

    plt.ylim(0, 80)

    plt.xlim(0, 120)

    plt.show()
team_shot_chart(df, 'France')
def team_goal_chart(data, team_name):



    fig, ax = plt.subplots()

    fig.set_size_inches(7, 5)

    ax.set_xlim([0,120])

    ax.set_ylim([0,80])

    draw_pitch(ax)



    shot_data = data[(data['type_name'] == "Shot") & (data['team_name'] == team_name)]



    # draw the scatter plot for goals

    x_coord_goal = [location[0] for i, location in enumerate(shot_data["location"]) if shot_data.iloc[i]['shot_outcome_name'] == "Goal"]

    y_coord_goal = [location[1] for i, location in enumerate(shot_data["location"]) if shot_data.iloc[i]['shot_outcome_name'] == "Goal"]

    

    # shots that end up with no goal

    x_coord = [location[0] for i, location in enumerate(shot_data["location"]) if shot_data.iloc[i]['shot_outcome_name'] != "Goal"]

    y_coord = [location[1] for i, location in enumerate(shot_data["location"]) if shot_data.iloc[i]['shot_outcome_name'] != "Goal"]

    

    # put the two scatter plots on to the pitch

    ax.scatter(x_coord_goal, y_coord_goal, c = 'red', label = 'goal')

    ax.scatter(x_coord, y_coord, c = 'blue', label = 'shots')

    plt.legend(loc='upper right')

    

    plt.show()

    

team_goal_chart(df,'France')
def team_goal_chart_dist(data, team_name):

    shot_data = data[(data['type_name'] == "Shot") & (data['team_name'] == team_name)]

    # draw the scatter plot for goals

    x_coord_goal = [location[0] for i, location in enumerate(shot_data["location"]) if shot_data.iloc[i]['shot_outcome_name'] == "Goal"]

    y_coord_goal = [location[1] for i, location in enumerate(shot_data["location"]) if shot_data.iloc[i]['shot_outcome_name'] == "Goal"]



    # shots that end up with no goal

    x_coord = [location[0] for i, location in enumerate(shot_data["location"]) if shot_data.iloc[i]['shot_outcome_name'] != "Goal"]

    y_coord = [location[1] for i, location in enumerate(shot_data["location"]) if shot_data.iloc[i]['shot_outcome_name'] != "Goal"]

  

    # we use a joint plot to see the density of the shot distribution across the 2 axes of the pitch

    joint_shot_chart = sns.jointplot(x_coord, y_coord, stat_func=None,

                                     kind='scatter', space=0, alpha=0.5)

    joint_shot_chart.fig.set_size_inches(7,5)

    ax = joint_shot_chart.ax_joint



    # overlaying the plot with a pitch

    draw_pitch(ax)

    ax.set_xlim(0,120)

    ax.set_ylim(0,80)







    # put the two scatter plots on to the pitch

    ax.scatter(x_coord, y_coord, c = 'b', label = 'shots')

    ax.scatter(x_coord_goal, y_coord_goal, c = 'r', label = 'goal')



    # Get rid of axis labels and tick marks

    ax.set_xlabel('')

    ax.set_ylabel('')

    joint_shot_chart.ax_marg_x.set_axis_off()

    ax.set_axis_off()

    plt.ylim(-.5, 80)

    plt.show()







team_goal_chart_dist(df, 'France')