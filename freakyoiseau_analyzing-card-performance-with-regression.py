import pandas as pd

import ast

import pandas as pd

from os import path

import json

import codecs

import numpy as np

import seaborn as sns

from itertools import chain

from sklearn.linear_model import LogisticRegression



images = json.load(open("../input/clash-images/images.json"))

cards_info = pd.read_csv("../input/clash-royale-card-infos/info_cards.csv")
games_file = open("../input/clash-royale-matches-dataset/matches.txt")

errors = 0

games = []

for i,line in enumerate(games_file.readlines()):

    try:

        dic = dict(ast.literal_eval(line.strip()))

        games.append(dic)

    except:

        errors+=1

    #if i>400000: break # subset data to 400000 otherwise kernel does not run
def parse(line):

    "flatten the json to get it in pandas format"

    keys = line.keys()

    data_out = {}

    for k in keys:

        data_out[k+'_clan'] = line[k]['clan']

        data_out[k+"_deck"] = line[k]['deck']

        data_out[k+'_name']  = line[k]['name']

        data_out[k+'_trophy']  = line[k]['trophy']

        data_out[k+'_deck_list'] = [name for name, level in line[k]['deck']]

        data_out[k+'_level_list'] = [level for name, level in line[k]['deck']]

    return data_out



games_df = pd.DataFrame(games)

add_var = pd.DataFrame(list(games_df.players.map(lambda x: parse(x))))

games_data = pd.concat([games_df.drop("players",axis=1), add_var],axis = 1)

res = pd.DataFrame(list(games_data.result), columns=['left_crowns', 'right_crowns'])

games_data = pd.concat([games_data.drop('result',axis=1), res],axis=1)

games_data.head()
ids = games_data.type =="ladder"

games_data = games_data[ids]



good_players = games_data.left_trophy.astype(int)>3000 # subset best players only

games_data = games_data[good_players]

games_data["mean_trophies"] = (games_data.left_trophy.astype(int)+  games_data.right_trophy.astype(int))/2 # get mean trophy of two players

games_data["diff_trophies"] = (games_data.left_trophy.astype(int)+  games_data.right_trophy.astype(int)) # get trophy difference (probably not so much informative)

games_data['left_diff'] = games_data.left_crowns.astype(int).astype(int) - games_data.right_crowns.astype(int)  # compute difference of column as target

games_data['right_diff'] = games_data.right_crowns.astype(int) - games_data.left_crowns.astype(int) # same for right player



left_deck = games_data[["left_level_list","left_deck_list","left_crowns","left_diff","mean_trophies","diff_trophies"]] # subset useful cplumns

right_deck = games_data[["right_level_list","right_deck_list","right_crowns",'right_diff',"mean_trophies","diff_trophies"]]

right_deck.columns = left_deck.columns # change name of left deck

data = left_deck.append(right_deck) # concatenate rows of right players with rows of left players as if they were independant

# might be a problem...



data.reset_index(drop=True, inplace=True) # reset index to avoid future problems and let's get started
# compute two lists to get all the possible cards played in the dataset

a = data.left_deck_list 

liste = set(chain.from_iterable(a))

liste_2 = [l+"_level" for l in liste]



# map elixir to each card

map_elixir = lambda x: [int(cards_info.loc[cards_info.name==a,"elixir"]) for a in x]

elixir_list = data.left_deck_list.map(map_elixir)

elixir_mean = elixir_list.map(np.mean)

elixir_mean = elixir_list.map(np.mean)

data['elixir_mean'] = elixir_mean

data.head()
from collections import Counter

def map(deck):

    s = Counter(deck)

    res = {k:s[k] for k in liste}

    return res

flat_deck = data.left_deck_list.map(lambda x: map(x))

cards = pd.DataFrame(list(flat_deck))
%%time



def map_again(tup):

    classe = {v:int(k) for k,v in tup}

    dico = {k:(classe[k.split("_")[0]] if k.split("_")[0] in classe else 0) for k in liste_2}

    return dico



format_row = lambda x: list(zip(x['left_level_list'], x["left_deck_list"]))



merged = data.apply(lambda x: format_row(x), axis=1)

marged = merged.map(map_again)

levels = pd.DataFrame(list(marged))
# concatenate all features (aroound 200 features at the end, seems reasonable compared to nb of lines)

X = pd.concat([cards, levels, data.mean_trophies, elixir_mean],axis=1)
# prepare targets for 4 different regressions (3 ordinal logit, and one Binary Logistic)

# the coefficients obtained cannot directly be interpreted in odds ratio (marginal probability), but their amount does not really matter, their comparison

#only is important (add odds ratio later)



y_diff = data.left_diff.astype(int) #ordinal (crowns won - crowns lost) (from -3 to +3)

y_attack = data.left_crowns.astype(int) # ordinal: crowns won: from 0 to 3

y_defense = data.left_crowns.astype(int) - data.left_diff.astype(int) # ordinal crowns lost: from 0 to 3

y_won = data.left_diff.astype(int) > 0 # logistic regression: True/False

targets = [y_diff, y_attack, y_defense, y_won]


from sklearn.linear_model import LogisticRegression



# train the models (would be good to add crossvalidation? not sure)



lr = LogisticRegression(C=1e9)

lr.fit(X,y_won)
def plotting_result(series, name = '', col ="b"):

    pd.Series(series).sort_values().plot(kind = "barh", figsize = (8,15), color =col,title = name);
coef_cards = pd.Series({card:coef for coef,card in zip(lr.coef_[0] , X.columns) if card in liste})

coef_levels = pd.Series({card.split("_")[0] :coef for coef,card in zip(lr.coef_[0] , X.columns) if card in liste_2})

plotting_result(coef_cards)
import matplotlib.image as mpimg

import matplotlib.pylab as plt

from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)
def res_plot(data, images,title = "", offset =0.002):

    f, ax = plt.subplots(figsize = (8,35))

    plt.style.use('seaborn-white')

    dat = pd.Series(data).sort_values()

    sorted_images = [images[i.replace(' ','-').replace('.','')] for i in dat.index]

    my_range = range(len(dat.index))

    ax.hlines(y=my_range, xmin=0, xmax=dat.values, color='skyblue',linewidth = 3)

    #ax.plot(dat, my_range, "o",color='r',markersize = 10)

    ax.set_ylim(-1,76 )

    ax.set_xlim(None,max(dat) +offset)

    ax.set_title(title)

    ax.grid(color='r', linestyle='--', linewidth=2)

    b=0

    for image,y, x in zip(sorted_images, my_range, dat.values):

        build_artist(ax, image, [x , y-0.5])

        b+=0.001

def build_artist(ax, image, xy):

    im = OffsetImage(image, zoom=0.1)

    im.image.axes = ax

    ab = AnnotationBbox(im, xy, pad=0,

                        xycoords='data',

                        frameon=False,

                        box_alignment =(0,0))

    ax.add_artist(ab)
res_plot(coef_cards, images ,title="Overall Performance")
res_plot(coef_levels, images,"Level bonus on card performance",offset = 0.005)