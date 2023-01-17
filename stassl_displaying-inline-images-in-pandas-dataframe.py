import glob
import random
import base64
import pandas as pd

from PIL import Image
from io import BytesIO
from IPython.display import HTML
pd.set_option('display.max_colwidth', -1)

def get_thumbnail(path):
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
dogs = pd.read_csv('../input/labels.csv')
dogs = dogs.sample(20)
dogs['file'] = dogs.id.map(lambda id: f'../input/train/{id}.jpg')
dogs['image'] = dogs.file.map(lambda f: get_thumbnail(f))
dogs.head()
# displaying PIL.Image objects embedded in dataframe
HTML(dogs[['breed', 'image']].to_html(formatters={'image': image_formatter}, escape=False))
# display images specified by path
HTML(dogs[['breed', 'file']].to_html(formatters={'file': image_formatter}, escape=False))
