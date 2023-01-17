import pandas as pd

from pandas.io.json import json_normalize

import json

import os



import IPython.display as display

from PIL import Image

from PIL.ImageDraw import Draw
data_dir = '/kaggle/input/chinese-traffic-sign-panels/'
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

    # Some panel coordinates are invalid, have to investigate on this

    # Will skip them so far

    panels = list(filter(lambda coords: len(coords)==8, row.panels))

    for coords in panels:

        draw.polygon(tuplize(coords), outline='#00FF00')

    display.display(image)