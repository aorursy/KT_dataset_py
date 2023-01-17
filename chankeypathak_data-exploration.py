# dependencies

import os

import numpy as np

import pandas as pd

import glob

import random

import base64



from PIL import Image

from io import BytesIO

from IPython.display import HTML
pd.set_option('display.max_colwidth', -1)



def get_thumbnail(path):

    if path and os.path.exists(path):

        i = Image.open(path)

        i.thumbnail((150, 150), Image.LANCZOS)

        return i



def image_base64(im):

    if isinstance(im, str):

        im = get_thumbnail(im)

    with BytesIO() as buffer:

        im.save(buffer, 'jpeg')

        return base64.b64encode(buffer.getvalue()).decode()



def image_formatter(im):

    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'



def add_image_path(x):

    image_path = '../input/train/' + x

    if os.path.exists(image_path):

        path = os.path.join(image_path, os.listdir(image_path)[0])

        return path
kin_df = pd.read_csv('../input/train_relationships.csv')

kin_df = kin_df.sample(50)

kin_df.head()
kin_df['p1_path'] = kin_df.p1.apply(lambda x: add_image_path(x))

kin_df['p2_path'] = kin_df.p2.apply(lambda x: add_image_path(x))

kin_df['p1_thumb'] = kin_df.p1_path.map(lambda f: get_thumbnail(f))

kin_df['p2_thumb'] = kin_df.p2_path.map(lambda f: get_thumbnail(f))

kin_df.head()
kin_df.dropna(inplace=True)
# displaying PIL.Image objects embedded in dataframe

HTML(kin_df[['p1', 'p2', 'p1_thumb', 'p2_thumb']].to_html(formatters={'p1_thumb': image_formatter, 'p2_thumb': image_formatter}, escape=False))
# display images specified by path

# HTML(kin_df[['p1', 'p2', 'p1_path', 'p2_path']].to_html(formatters={'p1_path': image_formatter, 'p2_path': image_formatter}, escape=False))