# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv(r"../input/pokemon/PokemonData.csv")

data.head(100)

first_ten_data=data.head(10)

first_ten_data
sort_by_defense = data.sort_values('Defense',ascending=False)

Defense_pokemons= sort_by_defense.head(10)

Defense_pokemons

import plotly.graph_objs as go

trace1 = go.Bar(

                x = data.Name,

                y = data.Speed,

                name = "Speed",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = data.Type1)

data = [trace1]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

fig



Defense_pokemons.Defense.plot(kind='bar', title ="V comp", figsize=(15, 10), legend=True, fontsize=12)


Defense_pokemons.Defense.plot(kind='bar', figsize=(8, 6))
import matplotlib.pyplot as plt

x=Defense_pokemons.Name

y=Defense_pokemons.Defense

fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 20

fig_size[1] = 12

plt.rcParams["figure.figsize"] = fig_size

top10_Defense_pokemons=plt.bar(x,y, width = 0.8,color=['green'])

plt.show() 