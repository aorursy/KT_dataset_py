#matplotlib inline



import matplotlib.pyplot as plt



import seaborn as sb

import numpy as np

import pandas as pd



plt.style.use('ggplot')
from sklearn.manifold import TSNE

from matplotlib import offsetbox



def virtualize_2d(X, y_pred, dataset, cmap, n_clusters):

    X_embedded = TSNE(

        n_components=2, 

        perplexity=20, 

        verbose=True, 

        n_iter=3000

    ).fit_transform(X)

    # normalize

    X_min, X_max = np.min(X_embedded, 0), np.max(X_embedded, 0)

    X_norm = (X_embedded - X_min) / (X_max - X_min)

    # draw picture with ground truth image and prediction result

    palette = np.array(sb.color_palette("hls", n_clusters))

    plt.figure(figsize=(20, 15))

    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=palette[y_pred])

    ax = plt.subplot(111)

    shown_images = np.array([[1., 1.]])

    for i in range(dataset.data.shape[0]):

        imagebox = offsetbox.AnnotationBbox(

            offsetbox.OffsetImage(

                dataset.images[i], 

                cmap=cmap

            ),

            X_norm[i],

            bboxprops=dict(edgecolor=palette[y_pred[i]])

        )

        ax.add_artist(imagebox)

    plt.show()
from sklearn.datasets import load_digits



digits = load_digits()

n_samples, h, w = digits.images.shape

plt.imshow(digits.images[10], cmap=plt.cm.gray)

print(digits.target[10])
from sklearn.cluster import KMeans

from sklearn.metrics.cluster import v_measure_score



X = digits.data

y_true = digits.target

n_clusters = 10

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

y_pred = kmeans.labels_

score = v_measure_score(y_true, y_pred)

print('cluster={}, v_measure_score={}'.format(n_clusters, score))

virtualize_2d(X, y_pred, digits, plt.cm.gray_r, n_clusters)
from sklearn.datasets import fetch_lfw_people



lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape

plt.imshow(lfw_people.images[10], cmap=plt.cm.gray)

print(lfw_people.target_names[lfw_people.target[10]])
from sklearn.cluster import KMeans

from sklearn.metrics.cluster import v_measure_score



X = lfw_people.data

y_true = lfw_people.target

n_clusters = 7

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

y_pred = kmeans.labels_

score = v_measure_score(y_true, y_pred)

print('cluster={}, v_measure_score={}'.format(n_clusters, score))

virtualize_2d(X, y_pred, lfw_people, plt.cm.gray, n_clusters)
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.metrics.cluster import v_measure_score



X = lfw_people.data

y_true = lfw_people.target

n_clusters = 7

n_components = 94 # range check between 10 to 100



X_pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_pca)

y_pred = kmeans.labels_

score = v_measure_score(y_true, y_pred)

print('cluster={}, v_measure_score={}'.format(n_clusters, score))

virtualize_2d(X_pca, y_pred, lfw_people, plt.cm.gray, n_clusters)
from keras.layers import Input, Dense

from keras.models import Model



from sklearn.cluster import KMeans

from sklearn.metrics.cluster import v_measure_score



X = lfw_people.data

y_true = lfw_people.target

n_clusters = 7

print (X.shape)

print (y_true.shape)





# normalize

X_ae = X.astype('float32')

X_min, X_max = np.min(X_ae, 0), np.max(X_ae, 0)

X_ae = (X_ae - X_min) / (X_max - X_min)

X_ae = X_ae.reshape((len(X_ae), np.prod(X_ae.shape[1:])))



# auto-encoder

img_dim = X_ae.shape[1]

input_img = Input(shape=(img_dim,))

encoded1 = Dense(128, activation='relu')(input_img)

decoded1 = Dense(img_dim, activation='sigmoid')(encoded1)

autoencoder1 = Model(input_img, decoded1)

autoencoder1.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder1.fit(

    X_ae, X_ae,

    epochs=500,

    batch_size=256,

    shuffle=True,

    validation_split=0.2,

    verbose=False

)

print('stacked 1 times!')



encoded2 = Dense(128, activation='relu')(input_img)

encoded2 = Dense(64, activation='relu')(encoded2)

decoded2 = Dense(128, activation='relu')(encoded2)

decoded2 = Dense(img_dim, activation='sigmoid')(decoded2)

autoencoder2 = Model(input_img, decoded2)

autoencoder2.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder2.layers[1].set_weights(autoencoder1.layers[1].get_weights())

autoencoder2.layers[-1].set_weights(autoencoder1.layers[-1].get_weights())

autoencoder2.fit(

    X_ae, X_ae,

    epochs=500,

    batch_size=256,

    shuffle=True,

    validation_split=0.2,

    verbose=False

)

print('stacked 2 times!')



encoded3 = Dense(128, activation='relu')(input_img)

encoded3 = Dense(64, activation='relu')(encoded3)

encoded3 = Dense(32, activation='relu')(encoded3)

decoded3 = Dense(64, activation='relu')(encoded3)

decoded3 = Dense(128, activation='relu')(decoded3)

decoded3 = Dense(img_dim, activation='sigmoid')(decoded3)

autoencoder3 = Model(input_img, decoded3)

encoder3 = Model(input_img, encoded3)

autoencoder3.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder3.layers[1].set_weights(autoencoder2.layers[1].get_weights())

autoencoder3.layers[2].set_weights(autoencoder2.layers[2].get_weights())

autoencoder3.layers[-2].set_weights(autoencoder2.layers[-2].get_weights())

autoencoder3.layers[-1].set_weights(autoencoder2.layers[-1].get_weights())

autoencoder3.fit(

    X_ae, X_ae,

    epochs=500,

    batch_size=256,

    shuffle=True,

    validation_split=0.2,

    verbose=False

)

print('stacked 3 times!')



X_encoded = encoder3.predict(X_ae)

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_encoded)

y_pred = kmeans.labels_

score = v_measure_score(y_true, y_pred)

print('cluster={}, v_measure_score={}'.format(n_clusters, score))

virtualize_2d(X_encoded, y_pred, lfw_people, plt.cm.gray, n_clusters)
from sklearn.cluster import SpectralClustering

from sklearn.metrics.cluster import v_measure_score



X = lfw_people.data

y_true = lfw_people.target

n_clusters = 7



# normalize

X_ae = X.astype('float32')

X_min, X_max = np.min(X_ae, 0), np.max(X_ae, 0)

X_ae = (X_ae - X_min) / (X_max - X_min)

X_ae = X_ae.reshape((len(X_ae), np.prod(X_ae.shape[1:])))



# auto-encoder

img_dim = X_ae.shape[1]

input_img = Input(shape=(img_dim,))

encoded = Dense(128, activation='relu')(input_img)

encoded = Dense(64, activation='relu')(encoded)

encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(decoded)

decoded = Dense(img_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(

    X_ae, X_ae,

    epochs=500,

    batch_size=256,

    shuffle=True,

    validation_split=0.2,

    verbose=False

)



X_encoded = encoder.predict(X_ae)

sc = SpectralClustering(

    n_clusters=n_clusters, 

    eigen_solver='arpack',

    affinity="nearest_neighbors"

).fit(X_encoded)

y_pred = sc.labels_

score = v_measure_score(y_true, y_pred)

print('cluster={}, v_measure_score={}'.format(n_clusters, score))

virtualize_2d(X_encoded, y_pred, lfw_people, plt.cm.gray, n_clusters)