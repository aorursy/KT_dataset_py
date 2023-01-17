

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import os

import io

from PIL import Image

import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

import multiprocessing

from sklearn import cluster

import random

%matplotlib inline

new_style = {'grid': False}

plt.rc('axes', **new_style)

random.seed(100);
# Set working directory

os.chdir('../input')
def hamdist(hash_set):

    diffs = 0

    for ch1, ch2 in zip(hash_set[0], hash_set[1]):

        if ch1 != ch2:

            diffs += 1

    return diffs



def dhash(image,hash_size = 16):

    image = image.convert('LA').resize((hash_size+1,hash_size),Image.ANTIALIAS)

    pixels = list(image.getdata())

    difference = []

    for row in range(hash_size):

        for col in range(hash_size):

            pixel_left = image.getpixel((col,row))

            pixel_right = image.getpixel((col+1,row))

            difference.append(pixel_left>pixel_right)

    decimal_value = 0

    hex_string = []

    for index, value in enumerate(difference):

        if value:

            decimal_value += 2**(index%8)

        if (index%8) == 7:

            hex_string.append(hex(decimal_value)[2:].rjust(2,'0'))

            decimal_value = 0

    return ''.join(hex_string)
counter = 1

hash_size = 16
# Lets get the train and test images and their respective gradient hashes

img_id_hash = []

parent_dir = "train"

subdir = os.listdir(os.path.join(parent_dir))

for direc in subdir: 

    try:

        names = os.listdir(os.path.join(parent_dir, direc))

    except:

        continue

    print(counter, direc, parent_dir)

    for name in names:

        imgdata = Image.open(os.path.join(parent_dir, direc, name)).convert("L")

        img_hash = dhash(imgdata, hash_size)

        img_id_hash.append([parent_dir, direc, name, img_hash])

        counter+=1



df = pd.DataFrame(img_id_hash,columns=['ParDirectory' , 'SubDirectory', 'file_name', 'image_hash'])

df.head(2)
# Create the image hash distance matrix for the distances in images

pool = multiprocessing.Pool(1)

distances = np.zeros((df.shape[0], df.shape[0]))

for i, row in df.iterrows():

    #if i % 50 == 0: print i

    all_hashes = [(row['image_hash'], f) for f in df.image_hash.tolist()]

    dists = pool.map(hamdist, all_hashes)

    distances[i, :] = dists

# Get a histogram of the distances

plt.hist(distances.flatten(), bins=50)

plt.title('Histogram of distance matrix')
# Cluster the images - average cluster size ~ 10

cls = cluster.KMeans(n_clusters=int(df.shape[0]/10), n_jobs = 8)

y = cls.fit_predict(distances)
# Lets look at the first 5 clusters to check were grouping similar images

_, ax = plt.subplots(12, 5, figsize=(10, 20))

ax = ax.flatten()

counter = 0

for c in range(20):

    for i, row in df[y==c].iterrows():

        if counter  == len(ax): 

            break

        if row['ParDirectory'] == 'test' :

            imgdata = Image.open(os.path.join(row['ParDirectory'], row['file_name']))

        else:

            imgdata = Image.open(os.path.join(row['ParDirectory'], row['SubDirectory'], row['file_name']))

        axis = ax[counter]

        axis.set_title('Cluster ' + str(c) + ' ' + row['SubDirectory'], fontsize=10)

        axis.imshow(np.asarray(imgdata), interpolation='nearest', aspect='auto')        

        axis.axis('off')

        counter += 1
# Lets pullout some random clusters as a validation set 

random.seed(100)

samp = random.sample(range(len(np.unique(y))), int(len(np.unique(y))/6))

df['Cluster'] = y

df['Validation'] = np.where(df['Cluster'].isin(samp), 1, 0)
# Break down of full data

df.SubDirectory.value_counts()
# Break down of validation data

df[df['Validation'] == 1].SubDirectory.value_counts()
# Use last column as validation indicator

df.to_csv('../working/image_validation_set.csv', index=False)