import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from tensorflow.keras.preprocessing import image

from textwrap import wrap
list_cols = ['subreddit', 'id', 'title', 'ups', 'downs', 'upvote_ratio', 'total_awards_received', 'num_comments', 'created_utc', 'filename']

ac = pd.read_csv('/kaggle/input/doom-crossing/animal_crossing_dataset.csv', usecols = list_cols)

doom = pd.read_csv('/kaggle/input/doom-crossing/doom_crossing_dataset.csv', usecols = list_cols)

data = pd.concat([ac, doom])

data.shape
data.head()
# Titles of images receiving most ups - Animal Crossing

ac.sort_values('ups', ascending=False).head(10)[['title','ups']]
# Titles of images receiving most ups - Doom

doom.sort_values('ups', ascending=False).head(10)[['title','ups']]
ac.downs.value_counts()
doom.downs.value_counts()
data_dir = '/kaggle/input/doom-crossing'



games = {'doom':[], 'animal_crossing': []}



for game in games.keys():

    for dirname, _, filenames in os.walk(os.path.join(data_dir, game)):

        for filename in filenames:

            games[game].append((os.path.join(os.path.join(data_dir, game), filename)))
np.random.seed(1)



# Show 12 random images of doom

row=3; col=4;

plt.figure(figsize=(20,(row/col)*12))

doom_img = np.random.choice(len(games['doom']), row*col)



for x in range(row*col):

    plt.subplot(row,col,x+1)

    img_path = games['doom'][doom_img[x]]

    img = image.load_img(img_path)

    plt.imshow(img)

    

plt.show()
np.random.seed(2)



# Show 12 random images of animal crossing

row=3; col=4;

plt.figure(figsize=(20,(row/col)*12))

ac_img = np.random.choice(len(games['animal_crossing']), row*col)



for x in range(row*col):

    plt.subplot(row,col,x+1)

    img_path = games['animal_crossing'][ac_img[x]]

    img = image.load_img(img_path)

    plt.imshow(img)

    

plt.show()
# Top 12 images that receives most ups - Animal Crossing

_ = ac.sort_values('ups', ascending=False).head(12)[['title','ups','filename']]



plt.figure(figsize=(20, 12))



for index, row in _.iterrows():

    plt.subplot(3,4, index+1)

    img_path = os.path.join('/kaggle/input/doom-crossing/animal_crossing', row['filename'])

    img = image.load_img(img_path)

    plt.imshow(img)

    plt.title('\n'.join(wrap(row['title'], 50)))

    

plt.show()
# Top 12 images that receives most ups - Doom

_ = doom.sort_values('ups', ascending=False).head(12)[['title','ups','filename']]



plt.figure(figsize=(20, 14))



for index, row in _.iterrows():

    plt.subplot(3,4, index+1)

    img_path = os.path.join('/kaggle/input/doom-crossing/doom', row['filename'])

    img = image.load_img(img_path)

    plt.imshow(img)

    plt.title('\n'.join(wrap(row['title'], 50)))

    

plt.show()