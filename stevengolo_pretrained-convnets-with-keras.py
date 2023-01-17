# Load packages

import h5py

import os

import random



import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd



from matplotlib.offsetbox import AnnotationBbox, OffsetImage



from skimage.io import imread

from skimage.transform import resize



from sklearn.manifold import TSNE



from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input

from tensorflow.keras.models import load_model, Model

from tensorflow.keras.preprocessing import image



DIR = '../input/pascal-voc-2012/VOC2012/JPEGImages'
# Load the model

model = ResNet50(include_top=True, weights='imagenet')
model.summary()
# Load and show an image

img = imread(f'{DIR}/2007_000250.jpg')



plt.imshow(img)

plt.show()
# Classify the image

img_resize = resize(img, (224, 224), preserve_range=True).astype('float32')

img_batch = preprocess_input(img_resize[np.newaxis])



preds = model.predict(img_batch)

preds_decode = decode_predictions(preds)



for idx, name, pct in preds_decode[0]:

    print(f'{name}: {100*pct:.3}%')
input_layer = model.layers[0].input

output_layer = model.layers[-1].input

base_model = Model(input_layer, output_layer)
print(f'The output shape of the base model is {base_model.output_shape}.')
representation = base_model.predict(img_batch)

print(f'Shape of the representation of the image: {representation.shape}.')

print(f'Proportion of zero valued axis: {100*np.mean(representation[0] == 0):.3}%.')
# Preprocess a sample of images

PATHS = [DIR + '/' + path for path in random.choices(os.listdir(DIR), k=1000)]

IMGS = []

for path in PATHS:

    img = imread(path)

    img_resize = resize(img, (224, 224), preserve_range=True).astype('float32')

    IMGS.append(preprocess_input(img_resize[np.newaxis]))
# Compute representation of sample of images

batch_tensor = np.vstack(IMGS)

out_tensors = base_model.predict(batch_tensor)
print(f'Shape of the representations: {out_tensors.shape}.')

print(f'Proportion of zero values axis for one representation: {100*np.mean(out_tensors[0] == 0):.3}%')
plt.hist(np.mean(out_tensors == 0, axis=1))

plt.show()
# Compute tSNE representation

img_emb_tsne = TSNE(perplexity=30).fit_transform(out_tensors)
plt.figure(figsize=(10, 10))

plt.scatter(img_emb_tsne[:, 0], img_emb_tsne[:, 1])

plt.xticks(()); plt.yticks(());

plt.show()
def imscatter(x, y, paths, ax=None, zoom=1, linewidth=0):

    if ax is None:

        ax = plt.gca()

    x, y = np.atleast_1d(x, y)

    artists = []

    for x0, y0, p in zip(x, y, paths):

        try:

            im = imread(p)

        except:

            print(p)

            continue

        im = resize(im,(224,224))

        im = OffsetImage(im, zoom=zoom)

        ab = AnnotationBbox(im, (x0, y0), xycoords='data',

                            frameon=True, pad=0.1, 

                            bboxprops=dict(edgecolor='red',

                                           linewidth=linewidth))

        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))

    ax.autoscale()

    return artists
fig, ax = plt.subplots(figsize=(50, 50))

imscatter(img_emb_tsne[:, 0], img_emb_tsne[:, 1], PATHS, zoom=0.5, ax=ax)

plt.show()
def most_similar(idx, top_n=5):

    # Compute Euclidean distance between every image reprentations

    # and the image idx.

    dists = np.linalg.norm(out_tensors - out_tensors[idx], axis=1)

    sorted_dists = np.argsort(dists)

    return sorted_dists[2:(top_n + 2)]
IDX = 50



similar = most_similar(IDX)



for s in similar:

    img = imread(PATHS[s])

    plt.imshow(img)

    plt.show()
# Compute the norms of each representation

out_norms = np.linalg.norm(out_tensors, axis=1, keepdims=True)

normed_out_tensors = out_tensors / out_norms
IDX = 50



eucl_dist = np.linalg.norm(out_tensors - out_tensors[IDX], axis=1)

cos_dist = np.dot(normed_out_tensors, normed_out_tensors[IDX])
fig, axes = plt.subplots(ncols=3, figsize=(15, 5))



img = imread(PATHS[s])

axes[0].imshow(img)

axes[0].set_title('Image')



axes[1].hist(eucl_dist)

axes[1].set_title('Euclidean distance')



axes[2].hist(cos_dist)

axes[2].set_title('Cosinus similarities')



plt.show()
items = np.where(cos_dist > 0.5)



for s in items[0]:

    img = imread(PATHS[s])

    plt.imshow(img)

    plt.show()