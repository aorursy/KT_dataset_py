import numpy as np

import pandas as pd

import random

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/fifa19/data.csv')

df.head()
df = df[['Name', 'Age', 'Overall', 'Value', 'Position']]

df['Value'] = df['Value'].str.replace('€','').str.replace('M',' 1000000').str.replace('K',' 1000')

df['Value'] = df['Value'].str.split(' ', expand=True)[0].astype(float) * df['Value'].str.split(' ', expand=True)[1].astype(float).fillna(0).astype(np.float32)

df = df[df['Position'].notnull()]

df = df[df['Value']>10000]

df.head()
replace_dict = {

    'GK': 'Goalkeeper',

    'ST': 'Center Forward', 

    'CF': 'Center Forward',

    'LF': 'Left Forward', 

    'LS': 'Left Forward',

    'RF': 'Right Forward', 

    'RS': 'Right Forward',

    'RCM': 'Center Half', 

    'LCM': 'Center Half', 

    'LDM': 'Center Half', 

    'CAM': 'Center Half', 

    'CDM': 'Center Half', 

    'RDM': 'Center Half', 

    'CM': 'Center Half',

    'LW': 'Left Half', 

    'LAM': 'Left Half', 

    'LM': 'Left Half',

    'RM': 'Right Half', 

    'RW': 'Right Half', 

    'RAM': 'Right Half',

    'RCB': 'Center Defender',

    'CB': 'Center Defender', 

    'LCB': 'Center Defender',

    'LB': 'Left Defender',  

    'LWB': 'Left Defender',

    'RB': 'Right Defender', 

    'RWB': 'Right Defender'

}



df['Position'] = df['Position'].replace(replace_dict)
df['Position'].unique().tolist()
df
class Player:

    def __init__(self, name, age, overall, value, position):

        self.name = name

        self.age = age

        self.overall = overall

        self.value = value

        self.position = position
def generate_player(player_df, genes, use_best=False):

    gen_df = player_df.sort_values(['Overall'], ascending=False)

    if use_best:

        player = 0

    else:

        player = random.randint(0, len(gen_df)-1)

    player_item = Player(

        gen_df.iloc[player]['Name'], 

        gen_df.iloc[player]['Age'],

        gen_df.iloc[player]['Overall'],

        gen_df.iloc[player]['Value'],

        gen_df.iloc[player]['Position']

    )

    

    return player_item
def create_individual(football_df):

    genes = []

    positions = ['Goalkeeper', 'Left Defender', 'Right Defender', 'Center Defender', 'Left Half', 'Right Half', 'Center Half', 'Left Forward', 'Right Forward', 'Center Forward'] 

    

    for pos in positions:

        player_df = football_df[football_df['Position'] == pos]

        for i in range(2):

            player_item = generate_player(player_df, genes)

            genes.append(player_item)

        if pos == 'Center Defender':

            for i in range(2):

                player_item = generate_player(player_df, genes)

                genes.append(player_item)



    return genes
def fitness(max_money, avg_age, individual):

    score = cost = age = 0

    for player in individual:

        score += player.overall

        cost += player.value

        age += player.age



    score += min(0, max_money-cost)

    age_diff = (avg_age - age / 22) * 10

    score += min(0, age_diff)

    

    if len(list(set([it.name for it in individual]))) != 22:

        score -= 1000000

        

    return score
def mutate(individual, mutation_rate, mutation_best_rate, footbal_df):

    new = []

    for gene in individual:

        player_df = footbal_df[footbal_df['Position'] == gene.position]

        if mutation_best_rate > random.random():

            player_item = generate_player(player_df, individual, use_best=True)

            new.append(player_item)

            continue

        if mutation_rate > random.random():

            player_item = generate_player(player_df, individual)

            new.append(player_item)

        else:

            player_item = generate_player(player_df, individual)

            new.append(player_item)

    return new
MAX_MONEY = 1200000000

AVG_AGE = 31

EPOCHS = 800

CHILDREN = 25

MUTATION_RATE = 0.15

MUTATION_BEST_RATE = 0.1

MUTATION_CHANGE_OVER_EPOCHS = 120

MUTATION_DECREASE = 0.015

INCREASE_IF_NO_IMPROVES = 80

MUTATION_INCREASE = 0.0075

CHILDREN_INCREASE = 2
ind = create_individual(df)

stable_score = 0

best_score = fitness(MAX_MONEY, AVG_AGE, ind)

best_ind = ind

for i in range(EPOCHS):

    improve_flag = False

    if stable_score == INCREASE_IF_NO_IMPROVES:

        stable_score = 0

        print('Mutation rate increased to: ', MUTATION_RATE+MUTATION_INCREASE)

        MUTATION_RATE += MUTATION_INCREASE

        CHILDREN += CHILDREN_INCREASE

    if i % 20 == 0:

        print('Epoch: ', i, best_score)

    if i % MUTATION_CHANGE_OVER_EPOCHS == 0 and i != 0:

        print('Mutation rate decreased to: ', MUTATION_RATE-MUTATION_DECREASE)

        MUTATION_RATE -= MUTATION_DECREASE

    for i in range(CHILDREN):

        child = mutate(ind, MUTATION_RATE, MUTATION_BEST_RATE, df)

        child_score = fitness(MAX_MONEY, AVG_AGE, child)

        if child_score >= best_score:

            best_score = child_score

            best_ind = child

            improve_flag=True

            

    if improve_flag == False:

        stable_score += 1
for player in best_ind:

    print(player.name, player.overall)
def draw_pitch(pitch, line, orientation,view):

    

    orientation = orientation

    view = view

    line = line

    pitch = pitch

    

    if view.lower().startswith("h"):

        fig,ax = plt.subplots(figsize=(20.8, 13.6))

        plt.ylim(98, 210)

        plt.xlim(-2, 138)

    else:

        fig,ax = plt.subplots(figsize=(13.6, 20.8))

        plt.ylim(-2, 210)

        plt.xlim(-2, 138)

    ax.axis('off')



    lx1 = [0, 0, 136, 136, 0]

    ly1 = [0, 208, 208, 0, 0]



    plt.plot(lx1,ly1,color=line,zorder=5)



    lx2 = [27.68, 27.68, 108.32, 108.32] 

    ly2 = [208, 175, 175, 208]

    plt.plot(lx2, ly2, color=line, zorder=5)



    lx3 = [27.68, 27.68, 108.32, 108.32] 

    ly3 = [0, 33, 33, 0]

    plt.plot(lx3,ly3,color=line,zorder=5)



    lx4 = [60.68, 60.68, 75.32, 75.32]

    ly4 = [208, 208.4, 208.4, 208]

    plt.plot(lx4,ly4,color=line,zorder=5)



    lx5 = [60.68,60.68,75.32,75.32]

    ly5 = [0,-0.4,-0.4,0]

    plt.plot(lx5,ly5,color=line,zorder=5)



       #6 yard boxes#

    lx6 = [49.68,49.68,86.32,86.32]

    ly6 = [208,199,199,208]

    plt.plot(lx6,ly6,color=line,zorder=5)



    lx7 = [49.68,49.68,86.32,86.32]

    ly7 = [0,9,9,0]

    plt.plot(lx7,ly7,color=line,zorder=5)



    #Halfway line, penalty spots, and kickoff spot

    lx8 = [0,136] 

    ly8 = [104,104]

    plt.plot(lx8,ly8,color=line,zorder=5)





    plt.scatter(68,186,color=line,zorder=5)

    plt.scatter(68,22,color=line,zorder=5)

    plt.scatter(68,104,color=line,zorder=5)



    circle1 = plt.Circle((68,187), 18.30,ls='solid',lw=3,color=line, fill=False, zorder=1,alpha=1)

    circle2 = plt.Circle((68,21), 18.30,ls='solid',lw=3,color=line, fill=False, zorder=1,alpha=1)

    circle3 = plt.Circle((68,104), 18.30,ls='solid',lw=3,color=line, fill=False, zorder=2,alpha=1)





    ## Rectangles in boxes

    rec1 = plt.Rectangle((40, 175), 60,33,ls='-',color=pitch, zorder=1,alpha=1)

    rec2 = plt.Rectangle((40, 0), 60,33,ls='-',color=pitch, zorder=1,alpha=1)



    ## Pitch rectangle

    rec3 = plt.Rectangle((-1, -1), 140,212,ls='-',color=pitch, zorder=1,alpha=1)



    ax.add_artist(rec3)

    ax.add_artist(circle1)

    ax.add_artist(circle2)

    ax.add_artist(rec1)

    ax.add_artist(rec2)

    ax.add_artist(circle3)   
first_team = []

second_team = []



i = 0

while i < len(best_ind):

    if best_ind[i].overall >= best_ind[i+1].overall:

        first_team.append(best_ind[i])

        second_team.append(best_ind[i+1])

    else:

        first_team.append(best_ind[i+1])

        second_team.append(best_ind[i])

    i+=2

    

first_team.reverse()

second_team.reverse()
draw_pitch("#195905","#faf0e6","v","full")

x = [68, 28, 108, 68, 28, 108, 40, 90, 20, 112, 68]

y = [160, 160, 160, 110, 110, 110, 40, 40, 60, 60, 1]

n = [str(item.name) + ', ' + str(item.overall) for item in first_team]



for i,type in enumerate(n):

    x_c = x[i]

    y_c = y[i]

    plt.scatter(x_c, y_c, marker='o', color='red', edgecolors="black", zorder=10)

    plt.text(x_c-2.5, y_c+1, type, fontsize=16)

draw_pitch("#195905","#faf0e6","v","full")

x = [68, 28, 108, 68, 28, 108, 40, 90, 20, 112, 68]

y = [160, 160, 160, 110, 110, 110, 40, 40, 60, 60, 1]

n = [str(item.name) + ', ' + str(item.overall) for item in second_team]



for i,type in enumerate(n):

    x_c = x[i]

    y_c = y[i]

    plt.scatter(x_c, y_c, marker='o', color='red', edgecolors="black", zorder=10)

    plt.text(x_c-2.5, y_c+1, type, fontsize=16)