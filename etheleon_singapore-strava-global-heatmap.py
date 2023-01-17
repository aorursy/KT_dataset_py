import os

#import hdf5



import pandas as pd

import numpy as np

import pickle



from PIL import Image

from io import BytesIO



import urllib

import requests

import matplotlib.pyplot as plt
def downloadTile( lat=1.31214, lon =  103.97219, zoom = 11, verbose=False):

    xtile = int(np.floor(((lon + 180) / 360) * 2**zoom))

    ytile = int(np.floor( (1 - np.log(np.tan(np.deg2rad(lat)) + 1 / np.cos(np.deg2rad(lat))) / np.pi) / 2 * 2**zoom))

    if verbose: 

        print(f'x: {xtile}')

        print(f'y: {ytile}')

    url = f'https://heatmap-external-c.strava.com/tiles/all/hot/{zoom}/{xtile}/{ytile}.png'

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))

    return img





file = open("../input/images.pkl", "rb")

img = pickle.load(file)

images = pickle.load(file)

file.close()
#img = downloadTile(zoom = 11, lon= 103.97219, lat=1.31214)

img

#images = [downloadTile(lat) for lat in [1.31214+i*0.1 for i in range(-3, 3, 1)]]
w=10

h=10

fig=plt.figure(figsize=(8, 8))

columns = 3

rows = 2

for i in range(1, columns*rows +1):

    img = images[i-1]

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

plt.show()