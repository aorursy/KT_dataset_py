import pandas as pd

races= pd.read_csv("../input/ittalentsracescleand/races_cleaned.csv")
from IPython.display import HTML, display



display(HTML('''<style>

    .widget-label { min-width: 30ex !important; }

    .widget-slider{ min-width: 100ex !important; }

</style>'''))
%matplotlib notebook

from ipywidgets import *

import numpy as np

import matplotlib.pyplot as plt

a = races.challenger.loc[races.winner!=-1].value_counts()

b = races.opponent.loc[races.winner!=-1].value_counts()

# now we combine those columns 

finished_races = a.add(b, fill_value=0)

wins = races.winner.loc[races.winner!=-1].value_counts()

ttt = wins.divide(finished_races).loc[finished_races > 50].nlargest(10).to_frame()

x = [x for x in range(0,10)]

fig = plt.figure(figsize=(10,5))

ax = fig.add_subplot(1, 1, 1)

ax.set_title("Top 10 der Fahrer mit der besten Sieg/Niederlage Quote")

rects = plt.bar(x, ttt.values[x][0])

ax.set_xticks(x)

ax.set_xticklabels(ttt.index)

ax.set_ylim([0.8,1.0])



def update(amount_of_finished_races = 50):

    ttt = wins.divide(finished_races).loc[finished_races > amount_of_finished_races].nlargest(10).to_frame()

    for rect, h in zip(rects, x):

        rect.set_height(ttt.values[h][0])

        ax.set_xticklabels(ttt.index)

    fig.canvas.draw_idle()



interact(update, amount_of_finished_races=IntSlider(min=0, max=200, step=1, value=50));