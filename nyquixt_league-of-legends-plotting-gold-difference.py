import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/leagueoflegends/LeagueofLegends.csv')
df.head(1)
df.columns
def calculate_gold_diff(gold_list_blue, gold_list_red):

    blue = gold_list_blue[1:-1].split(", ")

    blue = np.array([int(gold) for gold in blue])

    

    red = gold_list_red[1:-1].split(", ")

    red = np.array([int(gold) for gold in red])

    

    return blue - red
def plot_gold(gold_list):

    

    # Change str-like list to list

    if type(gold_list) == str:

        y = gold_list[1:-1].split(", ")

        y = [int(gold) for gold in y]

        y = np.array(y)

    else:

        y = gold_list

    

    fig, ax = plt.subplots(figsize=(15, 10))

    

    # set x-axis to be in the center

    ax.spines['bottom'].set_position('center')

    

    x = np.array([x for x in range(1, len(y) + 1)])

    

    ax.set_ylim([-np.max(y), np.max(y)])

    ax.fill_between(x, 0, y, where=y < 0, color='red')

    ax.fill_between(x, 0, y, where=y > 0, color='blue')
plot_gold(df.iloc[0]['golddiff'])
# Examine gold diff between 2 players of the same role

def plot_gold_diff_role(match_num, role):

    match = df.iloc[match_num] # Access match row

    

    goldblue = match['goldblue' + role]

    goldred = match['goldred' + role]

    

    golddiff = calculate_gold_diff(goldblue, goldred)

    plot_gold(golddiff)

    

plot_gold_diff_role(100, 'ADC')