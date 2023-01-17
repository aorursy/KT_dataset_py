import pydicom

import os

import glob

import numpy as np

import pandas as pd

from matplotlib import cm

from matplotlib import pyplot as plt

import cv2

import seaborn as sns

from tqdm import tqdm



from keras.models import Model

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras import backend as K



import plotly.graph_objs as go

import plotly.plotly as py

import plotly.offline as pyo

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly_express as px

init_notebook_mode(connected=True)



import tensorflow as tf



from tqdm import tqdm_notebook



# ['siim-acr-pneumothorax-segmentation-data', 'siim-acr-pneumothorax-segmentation']



import sys

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation/')



from mask_functions import rle2mask

import gc
def show_dcm_info(dataset):

    print("Filename.........:", file_path)

    print("Storage type.....:", dataset.SOPClassUID)

    print()



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name......:", display_name)

    print("Patient id..........:", dataset.PatientID)

    print("Patient's Age.......:", dataset.PatientAge)

    print("Patient's Sex.......:", dataset.PatientSex)

    print("Modality............:", dataset.Modality)

    print("Body Part Examined..:", dataset.BodyPartExamined)

    print("View Position.......:", dataset.ViewPosition)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)

            

def plot_pixel_array(dataset, figsize=(10,10)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()
samplesize = 5000

train_glob = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-train/*/*/*.dcm'

test_glob = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-test/*/*/*.dcm'

train_fns = sorted(glob.glob(train_glob))[:samplesize]

test_fns = sorted(glob.glob(test_glob))[:samplesize]

df_full = pd.read_csv('../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/train-rle.csv', index_col='ImageId')
im_height = 1024

im_width = 1024

im_chan = 1

# Get train images and masks

# X_train = np.zeros((len(train_fns), im_height, im_width, im_chan), dtype=np.uint8)

Y_train = np.zeros((len(train_fns), im_height, im_width, 1), dtype=np.int16)

print('Getting train images and masks ... ')

sys.stdout.flush()

for n, _id in tqdm_notebook(enumerate(train_fns), total=len(train_fns)):

    dataset = pydicom.read_file(_id)

#     X_train[n] = np.expand_dims(dataset.pixel_array, axis=2)

    try:

        if '-1' in df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']:

            Y_train[n] = np.zeros((1024, 1024, 1))

        else:

            if type(df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']) == str:

                x = np.expand_dims(rle2mask(df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels'], 1024, 1024), axis=2)

                Y_train[n] = x

            else:

                Y_train[n] = np.zeros((1024, 1024, 1))

                for x in df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']:

                    Y_train[n] =  np.maximum(Y_train[n], np.expand_dims(rle2mask(x, 1024, 1024), axis=2))

    except KeyError:

        print(f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")

        Y_train[n] = np.zeros((1024, 1024, 1)) # Assume missing masks are empty masks.



print('Done!')
from skimage.transform import rescale



image_setori = []

for i in range(samplesize):

    count = Y_train[i].sum()

    if count > 0:

        image_setori.append(rescale(Y_train[i],1.0/4.0)) 
image_set = np.asarray(image_setori)

samplesize = len(image_set)

image_set = np.squeeze(image_set)

image_set = np.reshape(image_set, ((samplesize, image_set.shape[1] * image_set.shape[2])))
image_set = image_set * 128

# for i in range(len(image_set)):

#     print(image_set[i].max())
import sys



# These are the usual ipython objects, including this one you are creating

ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']



# Get a sorted list of the objects and their sizes

sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
del Y_train

gc.collect()
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
pca = PCA(n_components=50).fit(image_set)

#Plotting the Cumulative Summation of the Explained Variance

y=np.cumsum(pca.explained_variance_ratio_)

data = [go.Scatter(y=y)]

layout = {'title': 'PCA Explained Variance'}

iplot({'data':data,'layout':layout})
pca = PCA(n_components=20)

image_PCA = pca.fit_transform(image_set)
trace1 = go.Scatter(y=pca.explained_variance_ratio_)

trace2 = go.Scatter(y=np.cumsum(pca.explained_variance_ratio_))

fig = tools.make_subplots(rows=1,cols=2,subplot_titles=('Explained Variance','Cumulative Explained Variance'))

fig.append_trace(trace1,1,1)

fig.append_trace(trace2,1,2)

fig['layout'].update(height=600, width=1200, title="Explained Variance Ratios",showlegend=False)

iplot(fig)
Nc = range(1,20)

kmeans = [KMeans(i) for i in Nc]

score = [kmeans[i].fit(image_PCA).score(image_PCA) for i in range(len(kmeans))]
data = [go.Scatter(y=score,x=list(Nc))]

layout = {'title':'Elbow Curve for KMeans'}

iplot({'data':data,'layout':layout})
n_clusters=12

kmeans = KMeans(n_clusters=n_clusters, random_state=42)

image_kmeans = kmeans.fit_predict(image_PCA)
image_kmeans.shape
image_clusters = np.zeros((n_clusters, image_set.shape[1]), dtype=np.float64)

clustercounts = np.zeros(n_clusters,dtype=np.int)

for i in range(samplesize):

    for j in range(n_clusters):

        if image_kmeans[i] == j:

            image_clusters[j] += image_set[i]

            clustercounts[j] += 1
print(image_clusters.shape)

print(clustercounts)

print(clustercounts.sum())
for j in range(n_clusters):

    image_clusters[j] = image_clusters[j] / clustercounts[j]

image_clusters = np.reshape(image_clusters, ((n_clusters, 256, 256)))
image_clusters.shape
image_clusters_sortingval = [np.sum(image_clusters[i,:80,:80]) for i in range(n_clusters)]

cluster_ordered = range(n_clusters)

cluster_ordered = [x for _,x in sorted(zip(image_clusters_sortingval,cluster_ordered))]
fig = plt.figure(figsize=(15,5))

fig.subplots_adjust(hspace=0.05, wspace=0.05)

j = 1

for i in cluster_ordered:

    plt.subplot(2,6,j)

    plt.imshow(image_clusters[i].T, cmap=plt.cm.bone)

    plt.title('Cluster '+str(i)+'. Num Samples: '+str(clustercounts[i]))

    j += 1

plt.tight_layout()

plt.suptitle("Clusters of Pneumothorax Diagnosis based on simple PCA & K-Means",fontsize=16,y=1.05)
from sklearn.manifold import TSNE

imageTSNE = TSNE(n_components=2).fit_transform(image_PCA)
imageTSNEdf = pd.concat([pd.DataFrame(imageTSNE),pd.DataFrame(image_kmeans)],axis=1)

imageTSNEdf.columns = ['x1','x2','cluster']

px.scatter(imageTSNEdf,x='x1',y='x2',color='cluster',color_continuous_scale=px.colors.qualitative.Plotly,title="TSNE visualization of Image Clusters",width=800,height=500)
n_clusters=6

kmeans = KMeans(n_clusters=n_clusters, random_state=42)

image_kmeans = kmeans.fit_predict(image_PCA)

image_clusters = np.zeros((n_clusters, image_set.shape[1]), dtype=np.float64)

clustercounts = np.zeros(n_clusters,dtype=np.int)

for i in range(samplesize):

    for j in range(n_clusters):

        if image_kmeans[i] == j:

            image_clusters[j] += image_set[i]

            clustercounts[j] += 1

for j in range(n_clusters):

    image_clusters[j] = image_clusters[j] / clustercounts[j]

image_clusters = np.reshape(image_clusters, ((n_clusters, 256, 256)))

image_clusters_sortingval = [np.sum(image_clusters[i,:80,:80]) for i in range(n_clusters)]

cluster_ordered = range(n_clusters)

cluster_ordered = [x for _,x in sorted(zip(image_clusters_sortingval,cluster_ordered))]

fig = plt.figure(figsize=(15,5))

fig.subplots_adjust(hspace=0.05, wspace=0.05)

j = 1

for i in cluster_ordered:

    plt.subplot(1,6,j)

    plt.imshow(image_clusters[i].T, cmap=plt.cm.bone)

    plt.title('Cluster '+str(i)+'. Num Samples: '+str(clustercounts[i]))

    j += 1

plt.tight_layout()

plt.suptitle("Clusters of Pneumothorax Diagnosis based on simple PCA & K-Means",)
imageTSNEdf = pd.concat([pd.DataFrame(imageTSNE),pd.DataFrame(image_kmeans)],axis=1)

imageTSNEdf.columns = ['x1','x2','cluster']

px.scatter(imageTSNEdf,x='x1',y='x2',color='cluster',color_continuous_scale=px.colors.qualitative.Plotly,title="TSNE visualization of Image Clusters",width=800,height=500)
# Clean-up

del image_PCA

gc.collect()
# import sys



# # These are the usual ipython objects, including this one you are creating

# ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']



# # Get a sorted list of the objects and their sizes

# sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
del df_full

gc.collect()
from keras.layers import Input, Dense

from keras.models import Model

encoding_dim = 128

imgsize_flat = 256 * 256

input_img = Input(shape=(imgsize_flat,))

encoded = Dense(encoding_dim,activation='relu')(input_img)

decoded = Dense(imgsize_flat,activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)



# Encoder

encoder = Model(input_img,encoded)



# Decoder

encoded_input = Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]

decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

xtrain = image_set

autoencoder.fit(xtrain,xtrain,epochs=20,batch_size=256,shuffle=True)
fig = plt.figure(figsize=(15,10))

for i in range(5):

    plt.subplot(2,5,i+1)

    plt.imshow(xtrain[i].reshape(256,256).T, cmap=plt.cm.bone)

    autoencoded = autoencoder.predict(xtrain[i:i+1])

    plt.subplot(2,5,i+6)

    plt.imshow(autoencoded.reshape(256,256).T, cmap=plt.cm.bone)

plt.tight_layout()

plt.suptitle('Comparing original vs AE reconstruction images',fontsize=16,y=1)        
from keras.layers import Input, Dense

from keras.models import Model

encoding_dim = 64

imgsize_flat = 256 * 256

layer1_multiplier = 32

layer2_multiplier = 16



input_img = Input(shape=(imgsize_flat,))

encoded = Dense(encoding_dim*layer1_multiplier ,activation='relu')(input_img)

encoded = Dense(encoding_dim*layer2_multiplier,activation='relu')(encoded)

encodedFinal = Dense(encoding_dim,activation='relu')(encoded)

decoded = Dense(encoding_dim*layer2_multiplier,activation='relu')(encodedFinal)

decoded = Dense(encoding_dim*layer1_multiplier ,activation='relu')(decoded)

decodedFinal = Dense(imgsize_flat,activation='sigmoid')(decoded)



autoencoder = Model(input_img, decodedFinal)



# Encoder

encoder = Model(input_img,encodedFinal)



# Decoder

# encoded_input = Input(shape=(encoding_dim,))

# decoder_layer = autoencoder.layers[-1]

# decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.summary()
xtrain = image_set
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(xtrain,xtrain,epochs=20,batch_size=64,shuffle=True)
fig = plt.figure(figsize=(15,5))

for i in range(10):

    plt.subplot(2,10,i+1)

    plt.imshow(xtrain[i].reshape(256,256).T, cmap=plt.cm.bone)

    plt.title('Image ' + str(i))

    

    autoencoded = autoencoder.predict(xtrain[i:i+1])

    plt.subplot(2,10,i+11)

    plt.imshow(autoencoded.reshape(256,256).T, cmap=plt.cm.bone)

    plt.title('AE ' + str(i))

plt.tight_layout()

plt.suptitle('Comparing original vs AE reconstruction images',fontsize=16,y=1)    
image = []

for i in range(10):

    image.append(encoder.predict(xtrain[i:i+1]))

image = np.array(image)

image = np.squeeze(image)

imagedf = pd.DataFrame(image)

imagedf
fig = plt.figure(figsize=(24,8))

for i in range(8):

    series = imagedf.iloc[:,i]

    plt.subplot(4,8,i+1)

    series.hist()

    plt.title('Dim ' + str(i))

plt.suptitle('Histogram for each encoding dimension')

plt.tight_layout()
imagedf.corr().iloc[:5,:5] #just a sample
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

from keras.models import Model

from keras import backend as K



input_img=Input(shape=(256,256,1))

x = Conv2D(16,(3,3),activation='relu',padding='same')(input_img)

x = MaxPooling2D((4,4), padding='same')(x)

x = Conv2D(4,(3,3), activation='relu',padding='same')(x)

encoded = MaxPooling2D((4,4), padding='same')(x)



x = Conv2D(4,(3,3),activation='relu',padding='same')(encoded)

x = UpSampling2D((4,4))(x)

x = Conv2D(16,(3,3),activation='relu',padding='same')(x)

x = UpSampling2D((4,4))(x)

decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)



autoencoderCNN=Model(input_img,decoded)

autoencoderCNN.compile(optimizer='adam',loss='binary_crossentropy')



# Encoder

encoderCNN = Model(input_img,encoded)



# Decoder

encoded_inputCNN = Input(shape=(16,16,4,))

decoder1 = autoencoderCNN.layers[-1]

decoder2 = autoencoderCNN.layers[-2]

decoder3 = autoencoderCNN.layers[-3]

decoder4 = autoencoderCNN.layers[-4]

decoder5 = autoencoderCNN.layers[-5]



decoderCNN = Model(encoded_inputCNN,decoder1(decoder2(decoder3(decoder4(decoder5(encoded_inputCNN))))))
autoencoderCNN.layers
autoencoderCNN.summary()
xtrain = np.reshape(xtrain, (len(xtrain),256,256,1))

autoencoderCNN.fit(xtrain,xtrain,epochs=20,batch_size=64,shuffle=True)
fig = plt.figure(figsize=(15,10))

for i in range(5):

    plt.subplot(2,5,i+1)

    plt.imshow(xtrain[i:i+1].reshape(256,256).T, cmap=plt.cm.bone)

    plt.title('Image ' + str(i))

    autoencoded = autoencoderCNN.predict(xtrain[i:i+1])

    plt.subplot(2,5,i+6)

    plt.imshow(autoencoded.reshape(256,256).T, cmap=plt.cm.bone)

    plt.title('AE ' + str(i))

plt.suptitle('Comparing original vs AE reconstruction images (5 images)',fontsize=16,y=1)

plt.tight_layout()
fig = plt.figure(figsize=(15,5))

for i in range(11,20):

    plt.subplot(2,10,i-10)

    plt.imshow(xtrain[i].reshape(256,256).T, cmap=plt.cm.bone)

    plt.title('Image ' + str(i))    

    autoencoded = autoencoderCNN.predict(xtrain[i:i+1])

    plt.subplot(2,10,i-0)

    plt.imshow(autoencoded.reshape(256,256).T, cmap=plt.cm.bone)

    plt.title('AE ' + str(i))

plt.suptitle('Comparing original vs AE reconstruction images (10 images)',fontsize=16,y=1.03)

plt.tight_layout()
encoderCNN = Model(input_img,encoded)

encodedX = []

for i in range(len(xtrain)):

    encodedX.append(encoderCNN.predict(xtrain[i:i+1]))

encodedX = np.array(encodedX)

print(encodedX.shape)

encodedX = np.squeeze(encodedX)

print(encodedX.shape)

encodeddf = pd.DataFrame(encodedX.reshape(encodedX.shape[0],np.prod(encodedX.shape[1:])))

encodeddf.head(20)
pca = PCA(n_components=50)

image_PCA = pca.fit_transform(encodeddf)

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot(pca.explained_variance_ratio_)

ax[0].title.set_text("Explained variance ratio")

ax[1].plot(np.cumsum(pca.explained_variance_ratio_))

ax[1].title.set_text("Cumulative Explained variance ratio")
pca = PCA(n_components=30)

image_PCA = pca.fit_transform(encodeddf)

Nc = range(1,30)

kmeans = [KMeans(i) for i in Nc]

score = [kmeans[i].fit(image_PCA).score(image_PCA) for i in range(len(kmeans))]

plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve to evaluate number of clusters')

plt.show()
n_clusters=12

kmeans = KMeans(n_clusters=n_clusters, random_state=42)

encoding_kmeans = kmeans.fit_predict(image_PCA)
encoding_clusters = np.zeros((n_clusters, encodeddf.iloc[0,:].shape[0]), dtype=np.float64)

clustercounts = np.zeros(n_clusters,dtype=np.int)

for i in range(len(encoding_kmeans)):

    for j in range(n_clusters):

        if encoding_kmeans[i] == j:

            encoding_clusters[j] += encodeddf.iloc[i,:]

            clustercounts[j] += 1

encoding_clustersdf = pd.DataFrame(encoding_clusters)

for j in range(n_clusters):

    encoding_clustersdf[j] = encoding_clustersdf[j] / clustercounts[j]
encoding_clustersdf
from skimage.filters import gaussian

fig = plt.figure(figsize=(15,10))

for i in range(len(encoding_clustersdf)):

    decoded = decoderCNN.predict(encoding_clustersdf.iloc[i,:].values.reshape((1,16, 16, 4)))

    plt.subplot(4,5,i+1)

    imgtoshow = gaussian(decoded.reshape(256,256).T, sigma=2)

    plt.imshow(imgtoshow, cmap=plt.cm.bone)

    plt.title('Cluster '+str(i)+' Size: '+str(clustercounts[i]))

plt.tight_layout()

plt.suptitle('Images of Final Clustering Result using Encoded Features as basis of clustering',fontsize=16,y=1.08)

plt.show()