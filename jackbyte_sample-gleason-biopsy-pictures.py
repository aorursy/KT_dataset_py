import os

import pandas as pd 

import openslide

from PIL import ImageFont

from PIL import ImageDraw

train = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
IMAGE_DIR = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'

def getSampleImageWithInfo(provider='radboud', fontsize=800):

    """Given a 'provider'  this function will return an sample/random image

    from the training data, along with a info dict.

    

    The info dict is also written in the top left corner of the image"""

    query = train.data_provider==provider

    filename = train[query].image_id.sample().values[0] + '.tiff'

    info = train[query].sample().to_dict(orient='list')

    text = out = ' '.join([f'{k.upper()} \t {info[k][0]} \n'for k in info])

    image = openslide.OpenSlide(os.path.join(IMAGE_DIR, filename))

    

    #check if image is to big

    too_big = True

    i = 0

    while too_big:

        w, h = image.level_dimensions[i]

        if w*h<2**26:

            too_big = False

        else:

            i = i + 1

            

    #draw info into image

    image = image.read_region((0,0), i, image.level_dimensions[i])

    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', fontsize)

    draw.text((0, 0), text, (0, 0, 0))

            

    return image, info
for prvdr in ['radboud', 'karolinska']:

    for i in range(10):

        img, info = getSampleImageWithInfo(provider=prvdr)

        img.save(fp=info['image_id'][0] + '.png')