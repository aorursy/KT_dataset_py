import numpy as np

import pandas as pd 

data = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')

data.to_csv('pokemon.csv')

data.describe()
%%time

import requests

from PIL import Image

from io import BytesIO

import os



os.mkdir('pokemon-image', 777)



# crawling

for i in range(809):

    r = requests.get(f"https://assets.pokemon.com/assets/cms2/img/pokedex/detail/{i+1:03}.png")

    im = Image.open(BytesIO(r.content))

    im.save(f'./pokemon-image/{i+1:03}.png')

    if i and i % 100 == 0 : 

        print(f"{i+1}th Image Save Compelete")
%%time 



import imageio

import matplotlib.pyplot as plt





generation_range = [

    [1, 151], # Gen1

    [152, 251], #Gen2

    [252, 386], # Gen3

    [387, 493], # Gen4

    [494, 649], # Gen5

    [650, 721], # Gen6

    [722, 809]] # Gen7



def pokemon_dict(gen, rng):    

    ln = rng[1] - rng[0] + 1

    fig = plt.figure(figsize=(10, (ln + 9)//10), dpi=300)

    for j in range(rng[0], rng[1]+1):

        ax = fig.add_subplot((ln + 9)//10, 10, j-rng[0] + 1)

        im = imageio.imread(f'./pokemon-image/{j:03}.png')

        ax.imshow(im)

        ax.axis('off')

    fig.suptitle(f'Generation {gen} Pokemon', fontweight='bold')

    plt.show()
%%time 



for i in range(7):

    pokemon_dict(i+1, generation_range[i])