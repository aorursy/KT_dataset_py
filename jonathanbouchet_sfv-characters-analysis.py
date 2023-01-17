import numpy as np

import pandas as pd

import json

import os

import re

import math

from pandas.io.json import json_normalize

import pprint

pp = pprint.PrettyPrinter(indent=4)



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        files.append(os.path.join(dirname, filename))
f = [str(s) for s in files if 'json' in s]



with open(f[0]) as jsn:

    df = json.load(jsn)

print(df.keys())
ab_dict = df['Abigail']

k0s = []

k1s_vals =[]

for k0,v0 in ab_dict.items():

    k0s.append(k0)

    k1s = []

    for k1,v1 in ab_dict[k0].items():

        k1s.append(k1)

    k1s_vals.append(k1s)
print(k0s[0],':' + ','.join(k1s_vals[0]))
print(k0s[1],':' + ','.join(k1s_vals[1]))
def make_stats(name):

    temp_dict = df[name]

    tempo_df = pd.DataFrame([temp_dict['stats']])

    tempo_df['Character'] = name

    #tempo_df['Character'] = list(df.keys())[0]

    return(tempo_df)

    

list_chars = []

for k,v in df.items():

    list_chars.append(make_stats(k))

df_stats = pd.concat(list_chars, axis=0, sort=True)
health_char = df_stats[['Character', 'health']].sort_values(by='health')

plt.figure(figsize=(10,10)) 

sns.barplot(y=health_char['Character'], 

            x= health_char['health'], 

            color="b").set_title('Character\'s health')
print(k0s[0],':' + ','.join(k1s_vals[0]))
ab_dict['moves']['normal'].keys()
ab_dict['moves']['vtOne'].keys()
ab_dict['moves']['vtTwo'].keys()
def make_moves(name):

    # get the name of normal, vt1 and vt2 moves for this character

    normal_moves = list(df[name]['moves']['normal'].keys())

    vt1_moves = list(df[name]['moves']['vtOne'].keys())

    vt2_moves = list(df[name]['moves']['vtTwo'].keys())

    

    # find same moves in the normal and vt1 list

    normal_vt1 = list(set(normal_moves).intersection(set(vt1_moves)))

    

    print("name: {}, total moves {}".format(name, len(normal_vt1)))

    

    normal_damages = []

    vt1_damages = []

    moves_name = []

    

    if len(normal_vt1) >0:

        for m in normal_vt1:

            moves_name.append(df[name]['moves']['normal'][m]['moveName'])

            try:

                normal_damages.append(df[name]['moves']['normal'][m]['damage'])

            except KeyError:

                normal_damages.append(np.nan) 

            try:

                vt1_damages.append(df[name]['moves']['vtOne'][m]['damage'])

            except KeyError:

                vt1_damages.append(np.nan) 



        temp_char = pd.DataFrame(list(zip(moves_name, normal_damages, vt1_damages)), 

                      columns = ['move name','normal damage', 'vt1 damage'])

        temp_char['Character'] = [name] * len(temp_char)

    elif len(normal_vt1) == 0:

        normal_damages.append(np.nan)

        vt1_damages.append(np.nan) 

        moves_name.append(np.nan)

        

        temp_char = pd.DataFrame(list(zip(moves_name, normal_damages, vt1_damages)), 

                      columns = ['move name','normal damage', 'vt1 damage'])

        temp_char['Character'] = [name] * len(temp_char)

    #print(temp_char)

    return(temp_char)

    

all_chars_moves = []

for name in list(df.keys()):

    all_chars_moves.append(make_moves(name))
df_moves = pd.concat(all_chars_moves, axis=0, sort=True)

df_moves.head(6)
# first check if it's nan or a string --> apply some regexp

# then deal with number

def get_val(x):

    #print(x)

    if x == "NaN" or x =="~":

        return np.nan

    elif "(" in str(x):

        f = re.findall('\(([^)]+)', x)

        #print(f[0])

        #if isinstance(f[0], str):

        if f[0] == "air":

            return(np.nan) # "air" case

        else:

            return(int(f[0]))

    elif "x" in str(x) and "*" in str(x):

        d = x.split("x")

        x = d[0]

        y = d[1].split("*")

        mysum = int(x) + int(y[0]) + int(y[1])

        return(mysum)

    elif "*" in str(x):

        return(int(x.split("*")[0]) + int(x.split("*")[1]))

    elif "+" in str(x):

        return(int(x.split("+")[0]) + int(x.split("+")[1]))

    elif "x" in str(x):

        return(int(x.split("x")[0]) + int(x.split("x")[1]))

    elif math.isnan(x):

        return np.nan

    else:

        return(int(x))
df_moves['normal damage num'] = df_moves['normal damage'].apply(get_val)

df_moves['vt1 damage num'] = df_moves['vt1 damage'].apply(get_val)
df_moves[['normal damage','normal damage num','vt1 damage','vt1 damage num']].head(10)