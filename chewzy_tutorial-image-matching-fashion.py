import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import os

import glob

import cv2

import annoy

import time



from tqdm import tqdm

from numpy.linalg import norm

from scipy.spatial import distance
train_folder_dir = '../input/fashionfull/fashion_image_resized/fashion_image_resized/train/'

test_folder_dir = '../input/fashionfull/fashion_image_resized/fashion_image_resized/test/'



df_train = pd.read_csv('../input/fashionfull/fashion.csv')

df_test = pd.read_csv('../input/fashionfull/fashion_test.csv')



train_image_paths = df_train['image_name'].values

test_image_paths = df_test['image_name'].values
def image_generator(image_paths, base_dir):

    for path in image_paths:

        img = cv2.imread(base_dir + path)

        resized = cv2.resize(img, (25,25))

        yield resized.flatten()
plt.figure(figsize=(20,12))



image1 = cv2.imread(train_folder_dir + train_image_paths[3])

resized1 = cv2.resize(image1, (25,25))



image2 = cv2.imread(train_folder_dir + train_image_paths[300])

resized2 = cv2.resize(image2, (25,25))





plt.subplot(1, 4, 1)

plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

plt.axis('off')

plt.title('Original Image 1',fontsize=16)



plt.subplot(1, 4, 2)

plt.imshow(cv2.cvtColor(resized1, cv2.COLOR_BGR2RGB))

plt.axis('off')

plt.title('Resized Image 1',fontsize=16)



plt.subplot(1, 4, 3)

plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

plt.axis('off')

plt.title('Original Image 2',fontsize=16)



plt.subplot(1, 4, 4)

plt.imshow(cv2.cvtColor(resized2, cv2.COLOR_BGR2RGB))

plt.axis('off')

plt.title('Resized Image 2',fontsize=16)



plt.tight_layout()

plt.show()
start_time = time.time()



vector_length = 25*25*3



t = annoy.AnnoyIndex(vector_length)



for i, v in enumerate(image_generator(train_image_paths, train_folder_dir)):

    t.add_item(i, v)



print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')
start_time = time.time()



Ntrees = 100

t.build(Ntrees)



print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')
def plot_neighbours(query_vector, n = 4):

    

    n_indices = t.get_nns_by_vector(query_vector, n)

    n_vectors = [np.array(query_vector, dtype=np.uint8).reshape((25,25,3))]

    n_distances = [0]

    

    for i in n_indices:

        n_vec = t.get_item_vector(i)

        n_arr = np.array(n_vec, dtype=np.uint8).reshape((25,25,3))

        n_vectors.append(n_arr)

        n_distances.append(np.abs(distance.cosine(n_vectors[0].ravel(), n_arr.ravel())))

        

    rows = n // 5 + 1

    

    plt.figure(figsize=(20, rows * 4))

    

    for i, n in enumerate(n_vectors):

        plt.subplot(rows, 5, i + 1)

        plt.imshow(cv2.cvtColor(n, cv2.COLOR_BGR2RGB))

        plt.axis('off')

        

        if i == 0:

            plt.title('Query Image', fontsize=16)

        else:

            plt.title(f'N{i}, Cosine_dist = {n_distances[i]:.2f}', fontsize=16)

        

    plt.tight_layout()

    plt.show()
for image in image_generator(test_image_paths[0:10], test_folder_dir):

    plot_neighbours(image, n=4)
t.save('fashion_100trees.ann')



# To load it next time, just run the following line with the appropriate file path

# t.load('../input/image-matching-fashion-v1/fashion_100trees.ann')