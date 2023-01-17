import pandas as pd

from pandas.io.json import json_normalize

import json

import os



import IPython.display as display

from PIL import Image

from PIL.ImageDraw import Draw
data_dir = '/kaggle/input/chinese-traffic-signs-in-the-natural-scenes/'
df = pd.read_json(os.path.join(data_dir, 'annotations.json'))

df = json_normalize(df['annotations'])

df.head(5)
# Helper method to convert coordinate list into tuple of tuples

def tuplize(coords):

    it = iter(map(float, coords))

    return tuple(zip(it, it))

    

# Test

tuplize(['253', '119', '403', '119', '403', '206', '253', '206'])
for index, row in df.sample(frac=1)[:5].iterrows():

    image = Image.open(os.path.join(data_dir, 'images', row.file))

    draw = Draw(image)

    for sign in row.signs:

        coords = sign['coords']

        draw.polygon(tuplize(coords), outline='#00FF00')

    display.display(image)