import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from skimage.util.montage import montage2d

from skimage.color import label2rgb

import os

import h5py

import seaborn as sns

sns.set_style("whitegrid", {'axes.grid': False})
%matplotlib inline

with h5py.File(os.path.join('..', 'input', 'lab_petct_vox_5.00mm.h5'), 'r') as p_data:

    fig, (c_ax1, c_ax2, c_ax3) = plt.subplots(1, 3, figsize=(10, 10), dpi = 250)

    for (p_id, ct_img), pt_img, lab_img in zip(

                                   p_data['ct_data'].items(),

                                   p_data['pet_data'].values(),

                                   p_data['label_data'].values()

                                                           ):

        c_ax1.imshow(np.sum(ct_img,1).squeeze()[::-1,:], cmap = 'bone')

        c_ax1.set_title('CT')

        c_ax1.axis('off')

        

        c_ax2.imshow(np.sqrt(np.max(pt_img,1).squeeze()[::-1,:]), cmap = 'magma')

        c_ax2.set_title('PET\n(sqrt)')

        c_ax2.axis('off')

        

        c_ax3.imshow(np.max(lab_img,1).squeeze()[::-1,:], cmap = 'gist_earth')

        c_ax3.set_title('Label')

        c_ax3.axis('off')

        cur_ct_img = np.sum(ct_img,1).squeeze()[::-1,:]

        cur_pet_img = np.sum(pt_img,1).squeeze()[::-1,:]

        cur_lab_img = np.max(lab_img,1).squeeze()[::-1,:]

        break # only load the first patient
ct_tiles = []

pet_tiles = []

lab_tiles = []

from tqdm import tqdm

for i in tqdm(range(0, cur_ct_img.shape[0], 16)):

    for j in range(0, cur_ct_img.shape[1], 16):

        for i_k in range(0, 16, 3):

            for j_k in range(0, 16, 3):

                ct_tiles += [cur_ct_img[i+i_k:(i+i_k+16),j+j_k:(j+j_k+16)]]

                pet_tiles += [cur_pet_img[i+i_k:(i+i_k+16),j+j_k:(j+j_k+16)]]

                lab_tiles += [cur_lab_img[i+i_k:(i+i_k+16),j+j_k:(j+j_k+16)]]
# we want to remove all tiles which do not have the right size (borders are a problem)

n_ct_tiles = [c_tile for c_tile in ct_tiles if c_tile.shape == (16,16)]

n_pet_tiles = [c_tile for c_tile in pet_tiles if c_tile.shape == (16,16)]

n_lab_tiles = [c_tile for c_tile in lab_tiles if c_tile.shape == (16,16)]

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

ax1.imshow(montage2d(np.stack(n_ct_tiles,0)), cmap = 'bone')

ax2.imshow(montage2d(np.stack(n_pet_tiles,0)), cmap = 'bone')

ax3.imshow(montage2d(np.stack(n_lab_tiles,0)), cmap = 'gist_earth')
lab_score = [np.mean(c_tile) for c_tile in n_lab_tiles]

ct_tile_flat = [c_tile.flatten() for c_tile in n_ct_tiles]

lab_class = [c_score > 0 for c_score in lab_score]
from sklearn.model_selection import train_test_split

train_tiles, test_tile, train_score, test_score = train_test_split(np.stack(ct_tile_flat), lab_score, 

                                                                   train_size = 0.8, stratify = lab_class, random_state = 1234)

print('Training size', train_tiles.shape, 'Testing size', test_tile.shape)

print('Train tumor tiles', np.where(train_score), 'Test tumor tiles', np.where(test_score))
from sklearn.neighbors import KNeighborsRegressor

knn1_model = KNeighborsRegressor(1, algorithm = 'brute')
knn1_model.fit(train_tiles, train_score)
test_predictions = knn1_model.predict(test_tile)

print('Predicted Results', test_predictions)
from sklearn.metrics import confusion_matrix

confusion_matrix(np.array(test_score)>0, np.array(test_predictions)>0)
fig, ax1 = plt.subplots(1,1)

ax1.plot([0,1], [0,1], 'b-', label = 'Ideal Model')

ax1.plot(test_score, test_predictions, 'ro', label = 'Current Results')

ax1.set_xlabel('Ground Truth Tumor Score')

ax1.legend()

ax1.set_ylabel('Predicted Tumor Score')

ax1.set_xlim(-0.1,1.1)
# Make a more complicated model with Neighbors = 2
knn2_model = KNeighborsRegressor(4)

knn2_model.fit(train_tiles, train_score)

test2_predictions = knn2_model.predict(test_tile)

print('Predicted Results', test2_predictions)

confusion_matrix(np.array(test_score)>0, np.array(test2_predictions)>0)
fig, ax1 = plt.subplots(1,1)

ax1.plot([0,1], [0,1], 'b-', label = 'Ideal Model')

ax1.plot(test_score, test_predictions, 'ro', label = 'KNN1 Results')

ax1.plot(test_score, test2_predictions, 'go', label = 'KNN2 Results')

ax1.set_xlabel('Ground Truth Tumor Score')

ax1.legend()

ax1.set_ylabel('Predicted Tumor Score')

ax1.set_xlim(-0.1,1.1)
petct_tile_flat = np.stack([np.hstack([c_ct_tile.flatten(), c_pet_tile.flatten()])  for c_ct_tile, c_pet_tile in zip(n_ct_tiles, n_pet_tiles)])

print('PETCT Tile Shape',petct_tile_flat.shape)
petct_train_tiles, petct_test_tile, petct_train_score, petct_test_score = train_test_split(petct_tile_flat, lab_score, 

                                                                   train_size = 0.8, stratify = lab_class, random_state = 1234)

print('Training size', petct_train_tiles.shape, 'Testing size', petct_test_tile.shape)

print('Train tumor tiles', len(np.where(petct_train_score)[0]), 

      ', Test tumor tiles', len(np.where(petct_test_score)[0]))
petct_knn1_model = KNeighborsRegressor(1, algorithm = 'brute')

petct_knn1_model.fit(petct_train_tiles, petct_train_score)
petct_test_predictions = petct_knn1_model.predict(petct_test_tile)

print('Predicted Results', np.mean(petct_test_predictions))
fig, ax1 = plt.subplots(1,1)

ax1.plot([0,1], [0,1], 'b-', label = 'Ideal Model')

ax1.plot(test_score, test_predictions, 'ro', label = 'KNN1 Results')

ax1.plot(test_score, test2_predictions, 'go', label = 'KNN2 Results')

ax1.plot(test_score, petct_test_predictions, 'mo', label = 'PETCT KNN1 Results')

ax1.set_xlabel('Ground Truth Tumor Score')

ax1.legend()

ax1.set_ylabel('Predicted Tumor Score')

ax1.set_xlim(-0.1,1.1)