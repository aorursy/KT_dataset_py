import glob

import cv2

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
df = pd.read_csv('../input/best-artworks-of-all-time/artists.csv')

df = df.drop(columns=['bio', 'wikipedia'])
artists_hist_dict = {}

for ii in glob.glob('../input/best-artworks-of-all-time/images/images/*'):

    blue_hist = []

    red_hist = []

    green_hist = []

    for j in glob.glob(ii + '/*'):

        img = cv2.imread(j)

        for i, col in enumerate(['b', 'g', 'r']):

            hist = cv2.calcHist([img], [i], None, [256], [0, 256])

            if col=='b':

                blue_hist.append(hist)

            elif col=='g':

                green_hist.append(hist)

            elif col=='r':

                red_hist.append(hist)



    blue_hist = sum(blue_hist)/len(blue_hist)

    green_hist = sum(green_hist)/len(green_hist)

    red_hist = sum(red_hist)/len(red_hist)

    artists_hist_dict[j.split('/')[-2]] = [blue_hist, green_hist, red_hist]
equivalent_artists = []

for i in artists_hist_dict:

    score_max = 0

    for j in artists_hist_dict:

        score = 0

        if i!=j:

            for k,l in zip(artists_hist_dict[j], artists_hist_dict[i]):     

                score = score + cv2.compareHist(k, l, cv2.HISTCMP_CORREL)

                score = score/3.0

            if score > score_max:

                score_max = score

                similar_artist = j

    equivalent_artists.append((i,similar_artist, score_max))
def plot(artist_1,  artist_2, title_of_plot):

    blue_hist1, green_hist1, red_hist1 = artists_hist_dict[artist_1]

    blue_hist2, green_hist2, red_hist2 = artists_hist_dict[artist_2]

    fig, axs = plt.subplots(2,figsize=(5,5))

    fig.tight_layout()

    #fig.suptitle('score :' + title_of_plot )

    axs[0].title.set_text(artist_1)

    axs[0].plot(blue_hist1, color = 'b')

    axs[0].plot(green_hist1, color = 'g')

    axs[0].plot(red_hist1, color = 'r')

    axs[1].title.set_text(artist_2)

    axs[1].plot(blue_hist2, color = 'b')

    axs[1].plot(green_hist2, color = 'g')

    axs[1].plot(red_hist2, color = 'r')
for i in equivalent_artists:

    plot(i[0], i[1], str(i[2]))
all_artwork = []

all_labels = []

for ii in glob.glob('../input/best-artworks-of-all-time/images/images/*'):

    for j in glob.glob(ii + '/*'):

        all_artwork.append(cv2.resize(cv2.imread(j), (128,128)))

        all_labels.append(ii.split('/')[-1])
all_artwork = np.asarray(all_artwork)
import gc

gc.collect()
from keras.models import *

from keras.layers import *

from keras.optimizers import *

import keras
inputs = Input(shape=(128,128,3))

##--encoder--##

conv = Conv2D(8, kernel_size=3, activation='relu')(inputs)

conv = MaxPooling2D(pool_size=(2, 2))(conv)

conv = Conv2D(8, kernel_size=3, activation='relu')(conv)

conv = Conv2D(8, kernel_size=3, activation='relu')(conv)

conv = MaxPooling2D(pool_size=(2, 2))(conv)

conv = Conv2D(16, kernel_size=3, activation='relu')(conv)

###

deconv = Conv2DTranspose(16, kernel_size=3, activation='relu')(conv)

deconv = UpSampling2D(size=(2, 2),interpolation='nearest')(deconv)

deconv = Conv2DTranspose(8, kernel_size=3, activation='relu')(deconv)

deconv = Conv2DTranspose(8, kernel_size=3, activation='relu')(deconv)

deconv = UpSampling2D(size=(2, 2),interpolation='nearest')(deconv)

deconv = Conv2DTranspose(8, kernel_size=3, activation='relu')(deconv)

##--get--back--image--##

img = Conv2DTranspose(3, kernel_size=3, activation='relu')(deconv)



model = Model(inputs=inputs, outputs=img)
model.summary()
optimizer = Adam(lr = 0.0001)

model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
gc.collect()
history_model = model.fit(all_artwork/255.0,  all_artwork/255.0, shuffle= True, epochs = 50,  batch_size=8,verbose=1)

print(history_model)
for ind, layer in enumerate(model.layers):

    print(ind," ",layer.name)
import random

for i in [random.randint(0, len(all_artwork)) for i in range(10)]:

    original_art_works =all_artwork[i]

    plt.title('original')

    plt.imshow(original_art_works)

    plt.show()

    reconstructed_art_works = np.squeeze(model.predict(np.asarray([original_art_works/255.0])))

    plt.title('reconstruction')

    plt.imshow(reconstructed_art_works)

    plt.show()
model_output = model.get_layer(model.layers[6].name).output

get_latent_vector = Model(inputs=model.input, outputs=model_output)
gc.collect()
latent_vectors = []

latent_labels = []

all_artwork = []

for ii in glob.glob('../input/best-artworks-of-all-time/images/images/*'):

    for j in glob.glob(ii + '/*'):

        all_artwork.append(cv2.resize(cv2.imread(j), (128,128)))

        latent_vectors.append(np.squeeze(get_latent_vector.predict(np.asarray([cv2.resize(cv2.imread(j), (128,128))/255.0]))))

        latent_labels.append(ii.split('/')[-1])
import random



c = list(zip(all_artwork,  latent_vectors, latent_labels))



random.shuffle(c)



all_artwork, latent_vectors, latent_labels = zip(*c)
gc.collect()
lv = []

for i in range(len(latent_vectors)):

    lv.append(np.reshape(latent_vectors[i],  (27*27*16)))
latent_labels = np.asarray(latent_labels)
import pandas as pd

import seaborn as sn

from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, random_state=0,  perplexity=100, learning_rate=10.0)

data = tsne_model.fit_transform(np.asarray(lv[0:2000]))

print(data.shape)

tsne_data = np.vstack((data.T, latent_labels[0:2000])).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

g = sn.FacetGrid(tsne_df, hue="label",  size=10).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

g.set(xticklabels=[], yticklabels=[])

plt.show()
lv1 = []

for i in range(len(all_artwork)):

    lv1.append(np.reshape(all_artwork[i],  (128*128*3)))
import pandas as pd

import seaborn as sn

from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, random_state=0,  perplexity=100, learning_rate=10.0)

data = tsne_model.fit_transform(np.asarray(lv1[0:2000])/255.0)

print(data.shape)

tsne_data = np.vstack((data.T, latent_labels[0:2000])).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

g = sn.FacetGrid(tsne_df, hue="label",  size=10).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

g.set(xticklabels=[], yticklabels=[])

plt.show()
import matplotlib.pyplot as plt



for kk in range(3):

    plt.imshow(all_artwork[kk])

    plt.show()

    fig, ax = plt.subplots(nrows=4, ncols=4, figsize = (10,10))



    i = 0

    for row in ax:

        for col in row:

            col.imshow(latent_vectors[kk][:,:,i])

            i = i+1



    plt.show()