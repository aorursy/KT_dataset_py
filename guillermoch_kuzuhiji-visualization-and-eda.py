import os



from collections import Counter



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import skimage

import seaborn as sns



from IPython.display import Image



INPUT_DIR = '/kaggle/input/kuzushiji-recognition'
train_metadata = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'), index_col='image_id')

unicode_translations = pd.read_csv(os.path.join(INPUT_DIR, 'unicode_translation.csv'))
train_metadata.head()
print(f'We have {train_metadata.shape[0]} images in the train dataset')
# There are some images that do not contain any character, let's have a look at one

img_no_chars = train_metadata[train_metadata.labels.isnull()].reset_index().iloc[0].image_id

Image(os.path.join(INPUT_DIR, 'train_images', '{}.jpg'.format(img_no_chars)), width=500)
def show_page(img_id, bounding_boxes=False):

    """ Shows a page (image)

    

    :param img_id - str: ID of the page to show

    :param bounding_boxes - boolean: If True, will drow the bounding boxes on the characters contained in the image

    """

    img = skimage.io.imread(os.path.join(INPUT_DIR, 'train_images', '{}.jpg'.format(img_id)))

    plt.figure(figsize=(15,15))

    plt.imshow(img)

    

    if bounding_boxes:

        def _chunks(l, n):

            for i in range(0, len(l), n):

                yield l[i:i+n]

        ax = plt.gca()

        chars = train_metadata.loc[img_id].labels.split()

        for char, x, y, height, width in chars[::5]:

            ax.add_patch(plt.Rectangle(x, y, height, width, color='blue', fill=False, linewidth=2))

            ax.text(x, y, char, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

            plt.show()
show_page('100241706_00004_2', bounding_boxes=True)
unicode_translations.head()
def num_chars_in_image(labels):

    """ Return the number of characters in an image given the labels string

    

    Labels is a string with all the characters in the image. The format is labels = ['unicode char', 'X', 'Y', 'width', 'height'] for each character.

    So if we jump the list in steps of length 5, we can get all the unicodes

    """

    return 0 if labels is np.nan else len(labels[::5])

    

train_metadata['num_chars'] = train_metadata.labels.apply(lambda l: num_chars_in_image(l))
def count_characters(df):

    counter = Counter()

    for labels in df[df.num_chars > 0].labels:

        for c in labels.split()[::5]:

            counter[c]+=1

    return counter

        

chars_count = count_characters(train_metadata)
print(f'Min characters per image: {train_metadata.num_chars.min()}')

print(f'Max characters per image: {train_metadata.num_chars.max()}')

print(f'Avg characters per image: {np.average(train_metadata.num_chars)} (std of {np.std(train_metadata.num_chars)})')

print(f'Total number of different characters (classes) is {len(chars_count)}')
_ = sns.distplot(train_metadata.num_chars, kde=False)

_ = plt.title('Number of characters per page count distribution')
_ = sns.distplot(train_metadata[train_metadata.num_chars > 0].num_chars, kde=False)

_ = plt.title('Number of characters per page count distribution (no blank pages)')