import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import zipfile

from tqdm import tqdm_notebook as tqdm

import cv2
Face_Image = np.load('../input/face-images-with-marked-landmark-points/face_images.npz')['face_images']

Face_Landmark = pd.read_csv('../input/face-images-with-marked-landmark-points/facial_keypoints.csv')
Face_Landmark.columns
Face_Landmark.head()
(imHeight, imWidth, numImages) = Face_Image.shape

numKeypoints = Face_Landmark.shape[1] / 2



print('number of images = %d' %(numImages))

print('image dimentions = (%d,%d)' %(imHeight,imWidth))

print('number of facial keypoints = %d' %(numKeypoints))
num_fig_rows = 2

num_fig_cols = 2



num_plots = num_fig_rows * num_fig_cols



rand_inds_vec = np.random.choice(Face_Image.shape[2],num_plots,replace=False)

rand_inds_mat = rand_inds_vec.reshape((num_fig_rows,num_fig_cols))



plt.close('all')

fig, ax = plt.subplots(nrows=num_fig_rows,ncols=num_fig_cols,figsize=(64,64))



for i in range(num_fig_rows):

    for j in range(num_fig_cols):

        curr_ind = rand_inds_mat[i][j]

        curr_image = Face_Image[:,:,curr_ind]

    

        x_feature_coords = np.array(Face_Landmark.iloc[curr_ind,[0,2]].tolist())

        y_feature_coords = np.array(Face_Landmark.iloc[curr_ind,[1,3]].tolist())

    

        ax[i][j].imshow(curr_image, cmap='gray');

        ax[i][j].scatter(x_feature_coords,y_feature_coords,c='r',s=100)

        ax[i][j].set_axis_off()

        ax[i][j].set_title('image index = %d' %(curr_ind),fontsize=50)
%%time

OUTPUT = 'ALL-FACE-JPG-1.zip'

x_mean,x_std = [],[]



with zipfile.ZipFile(OUTPUT, 'w') as output:

    for idx in tqdm(range(Face_Image.shape[2])):

        image = Face_Image[:,:,idx]

        x_mean.append(np.array(image, float).mean())

        x_std.append((np.array(image, float)**2).mean())

        image = cv2.imencode('.jpg', np.array(image))[1]

        output.writestr(str(idx)+'.jpg', image)

        

Face_Landmark.to_csv('ALL-FACE-1.csv', index=False)
np.array(x_mean).mean(), np.sqrt(np.array(x_std).mean() - np.array(x_mean).mean()**2)
Face_Image = Face_Image[:,:,Face_Landmark.loc[:,'left_eye_center_x':'right_eye_outer_corner_y'].isnull().sum(axis=1)==0]

Face_Landmark = Face_Landmark.loc[Face_Landmark.loc[:,'left_eye_center_x':'right_eye_outer_corner_y'].isnull().sum(axis=1)==0,:].reset_index(drop=True)



(imHeight, imWidth, numImages) = Face_Image.shape

numKeypoints = Face_Landmark.shape[1] / 2



print('number of images = %d' %(numImages))

print('image dimentions = (%d,%d)' %(imHeight,imWidth))

print('number of facial keypoints = %d' %(numKeypoints))
num_fig_rows = 2

num_fig_cols = 2



num_plots = num_fig_rows * num_fig_cols



rand_inds_vec = np.random.choice(Face_Image.shape[2],num_plots,replace=False)

rand_inds_mat = rand_inds_vec.reshape((num_fig_rows,num_fig_cols))



plt.close('all')

fig, ax = plt.subplots(nrows=num_fig_rows,ncols=num_fig_cols,figsize=(64,64))



for i in range(num_fig_rows):

    for j in range(num_fig_cols):

        curr_ind = rand_inds_mat[i][j]

        curr_image = Face_Image[:,:,curr_ind]

    

        x_feature_coords = np.array(Face_Landmark.iloc[curr_ind,0:12:2].tolist())

        y_feature_coords = np.array(Face_Landmark.iloc[curr_ind,1:12:2].tolist())

    

        ax[i][j].imshow(curr_image, cmap='gray');

        ax[i][j].scatter(x_feature_coords,y_feature_coords,c='r',s=100)

        ax[i][j].set_axis_off()

        ax[i][j].set_title('image index = %d' %(curr_ind),fontsize=50)
%%time

OUTPUT = 'FRONT_FACE-JPG-1.zip'

x_mean,x_std = [],[]



with zipfile.ZipFile(OUTPUT, 'w') as output:

    for idx in tqdm(range(Face_Image.shape[2])):

        image = Face_Image[:,:,idx]

        x_mean.append(np.array(image, float).mean())

        x_std.append((np.array(image, float)**2).mean())

        image = cv2.imencode('.jpg', np.array(image))[1]

        output.writestr(str(idx)+'.jpg', image)

        

Face_Landmark.to_csv('FRONT-FACE-1.csv', index=False)
np.array(x_mean).mean(), np.sqrt(np.array(x_std).mean() - np.array(x_mean).mean()**2)