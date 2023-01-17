import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

train = pd.read_csv('../input/train.csv')
from matplotlib import font_manager as fm, rcParams

import math



def read_characters(line):

    if type(line)==float and math.isnan(line):

        return None

    return line.split(' ')



def show_page(page, width=15, height=15):

    im = mpimg.imread('../input/train_images/'+train.loc[page,'image_id']+'.jpg')

    fig,ax = plt.subplots(1)

    ax.imshow(im)

    fig.set_figheight(height)

    fig.set_figwidth(width)

    plt.show()
def find_images(train):

    examples_with_images = train[train.apply(lambda x: isinstance(x.labels, float), axis=1)]

    return examples_with_images

        
import random



random.seed( 7 ) #Just to pick one of my favorites on this kernel



images_indexes = find_images(train).index

choice = random.choice(images_indexes)

print("Page index "+str(choice))

show_page(choice, 25, 25)