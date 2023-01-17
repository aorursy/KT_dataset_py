import os

import cv2

import time

import numpy as np

from keras.models import Model

from keras.applications import VGG16,ResNet50

from scipy.spatial import distance
# a dummy model, good old resnet50

model = ResNet50(weights='imagenet')



# removing the final classification layer 

model = Model(inputs=[model.input], outputs=[model.layers[-2].output])
index_files = []

for dirname, _, filenames in os.walk('../input/landmark-retrieval-2020/index/'):

    for filename in filenames:

        index_files.append(os.path.join(dirname, filename))



query_files = []

for dirname, _, filenames in os.walk('../input/landmark-retrieval-2020/test/'):

    for filename in filenames:

        query_files.append(os.path.join(dirname, filename))



print("We have a total of {} query images and a total of {} index images".format(len(query_files), len(index_files)))
index_embeddings = []

query_embeddings = []



# considering only the first 10,000 index images

for j,index_file in enumerate(index_files[:10000]):

    im = cv2.imread(index_file)

    im = cv2.resize(im, (224,224))

    index_embedding = model.predict(np.array([im]))[0]

    index_embeddings.append(index_embedding)



# considering only the first 500 query images

for i,query_file in enumerate(query_files[:500]):

    im = cv2.imread(query_file)

    im = cv2.resize(im, (224,224))

    query_embedding = model.predict(np.array([im]))[0]

    query_embeddings.append(query_embedding)

        
start = time.time()

distances = distance.cdist(np.array(query_embeddings), np.array(index_embeddings), 'euclidean')

predicted_positions = np.argpartition(distances, 100, axis=1)[:,:100]



print(predicted_positions.shape)

print("Time taken {} secs".format(time.time() - start))

# install

!pip install faiss



import faiss                                 # make faiss available

faiss_index = faiss.IndexFlatL2(2048)        # build the index, need to input embedding size (last layer dimension of our model)

print(faiss_index.is_trained)
# adding the index embeddings to faiss

faiss_index.add(np.array(index_embeddings))



# check how many are added

print("total embeddings added", faiss_index.ntotal) 



# now timing retrieval

start = time.time()

_, I = faiss_index.search(np.array(query_embeddings), 100)

    

print(I.shape)

print("Time taken {} secs".format(time.time() - start))

    