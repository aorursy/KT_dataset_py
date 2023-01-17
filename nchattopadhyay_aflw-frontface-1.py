import os

import glob

import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import zipfile

from tqdm import tqdm_notebook as tqdm
Label_List = glob.glob("/kaggle/input/aflworiginal/annotation/annotation/*/*.pts")

Image_List = glob.glob("/kaggle/input/aflworiginal/aflw-Original/aflw-Original/*/*.jpg")

len(Label_List), len(Image_List)
%%time



Landmarks_X = pd.DataFrame(columns=range(0,22))

Landmarks_Y = pd.DataFrame(columns=range(0,22))



row=0

for i in tqdm(Label_List):

    Landmarks = np.loadtxt(i, comments=("True", "False")).T

    Temp_X = pd.DataFrame(Landmarks[[1]], columns=Landmarks[0])

    Temp_Y = pd.DataFrame(Landmarks[[2]], columns=Landmarks[0])

    Landmarks_X = Landmarks_X.append(Temp_X)

    Landmarks_Y = Landmarks_Y.append(Temp_Y)

    Landmarks_X.iloc[row,0] = i[49:61].replace('.pts', '')

    Landmarks_Y.iloc[row,0] = i[49:61].replace('.pts', '')

    row+=1
Landmarks_X = Landmarks_X[Landmarks_X.iloc[:, [8,11]].isnull().sum(axis=1)==0]

Landmarks_Y = Landmarks_Y[Landmarks_Y.iloc[:, [8,11]].isnull().sum(axis=1)==0]



Landmarks_X.reset_index(inplace=True, drop=True)

Landmarks_Y.reset_index(inplace=True, drop=True)



Landmarks_X.shape, Landmarks_Y.shape
Landmarks_X
num_fig_rows = 10

num_fig_cols = 1



num_plots = num_fig_rows * num_fig_cols



rand_inds_vec = np.random.choice(Landmarks_X.shape[0],num_plots,replace=False)

rand_inds_mat = rand_inds_vec.reshape((num_fig_rows,num_fig_cols))



plt.close('all')

fig, ax = plt.subplots(nrows=num_fig_rows,ncols=num_fig_cols,figsize=(128,128))



for i in range(num_fig_rows):

    for j in range(num_fig_cols):

        curr_ind = rand_inds_mat[i][j]

        curr_image = cv2.imread('/kaggle/input/aflworiginal/aflw-Original/aflw-Original/'+Landmarks_X.iloc[curr_ind, 0]+'.jpg', cv2.IMREAD_COLOR)

    

        x_feature_coords = np.array(Landmarks_X.iloc[curr_ind,[8,11]].tolist())

        y_feature_coords = np.array(Landmarks_Y.iloc[curr_ind,[8,11]].tolist())

    

        ax[i].imshow(curr_image[:,:,[2,1,0]]);

        ax[i].scatter(x_feature_coords,y_feature_coords,c='r',s=20)

        ax[i].set_axis_off()

        ax[i].set_title('image index = '+(Landmarks_X.iloc[curr_ind, 0]),fontsize=10)
curr_ind=7448



plt.close('all')

plt.figure(figsize = (96, 96))



curr_image = cv2.imread('/kaggle/input/aflworiginal/aflw-Original/aflw-Original/'+Landmarks_X.iloc[curr_ind, 0]+'.jpg', cv2.IMREAD_COLOR)

    

x_feature_coords = np.array(Landmarks_X.iloc[curr_ind,[8,11]].tolist())

y_feature_coords = np.array(Landmarks_Y.iloc[curr_ind,[8,11]].tolist())

    

plt.imshow(curr_image[:,:,[2,1,0]]);

plt.scatter(x_feature_coords,y_feature_coords,c='r',s=500)

print(Landmarks_X.iloc[curr_ind, 0])
i=7448



plt.close('all')

plt.figure(figsize = (64, 64))



image = cv2.imread('/kaggle/input/aflworiginal/aflw-Original/aflw-Original/'+Landmarks_X.iloc[i, 0]+'.jpg', cv2.IMREAD_COLOR)



left = np.int(np.floor(Landmarks_X.iloc[i,1:].min()))

right = np.int(np.ceil(Landmarks_X.iloc[i,1:].max()))

top = np.int(np.floor(Landmarks_Y.iloc[i,1:].min()))

bottom = np.int(np.ceil(Landmarks_Y.iloc[i,1:].max()))



size = max(right-left, bottom-top)



if right-left < size:

    extra = (size-right+left)//2

    left = max(left-extra, 0)

    right = min(right+extra, image.shape[1])

    

else:

    extra = (size-bottom+top)//2

    top = max(top-extra, 0)

    bottom = min(bottom+extra, image.shape[0])



x_feature_coords = np.array(Landmarks_X.iloc[i,[8,11]].tolist())

y_feature_coords = np.array(Landmarks_Y.iloc[i,[8,11]].tolist())



image = image[top:bottom, left:right, [2,1,0]]

plt.imshow(image)

plt.scatter(x_feature_coords-left, y_feature_coords-top, c='r', s=500)
i=9651



plt.close('all')

plt.figure(figsize = (64, 64))



image = cv2.imread('/kaggle/input/aflworiginal/aflw-Original/aflw-Original/'+Landmarks_X.iloc[i, 0]+'.jpg', cv2.IMREAD_COLOR)



left = np.int(np.floor(Landmarks_X.iloc[i,1:].min()))

right = np.int(np.ceil(Landmarks_X.iloc[i,1:].max()))

top = np.int(np.floor(Landmarks_Y.iloc[i,1:].min()))

bottom = np.int(np.ceil(Landmarks_Y.iloc[i,1:].max()))



size = max(right-left, bottom-top)



if right-left < size:

    extra = (size-right+left)//2

    left = max(left-extra, 0)

    right = min(right+extra, image.shape[1])

    

else:

    extra = (size-bottom+top)//2

    top = max(top-extra, 0)

    bottom = min(bottom+extra, image.shape[0])



x_feature_coords = np.array(Landmarks_X.iloc[i,[8,11]].tolist())

y_feature_coords = np.array(Landmarks_Y.iloc[i,[8,11]].tolist())



image = image[top:bottom, left:right, [2,1,0]]

plt.imshow(image)

plt.scatter(x_feature_coords-left, y_feature_coords-top, c='r', s=500)
%%time



OUTPUT = 'FRONT_FACE-AFLW-JPG-1.zip'

x_mean,x_std = [],[]



with zipfile.ZipFile(OUTPUT, 'w') as output:

    

    for i in tqdm(range(Landmarks_X.shape[0])):



        image = cv2.imread('/kaggle/input/aflworiginal/aflw-Original/aflw-Original/'+Landmarks_X.iloc[i, 0]+'.jpg', cv2.IMREAD_COLOR)



        left = np.int(np.floor(Landmarks_X.iloc[i,1:22].min()))

        right = np.int(np.ceil(Landmarks_X.iloc[i,1:22].max()))

        top = np.int(np.floor(Landmarks_Y.iloc[i,1:22].min()))

        bottom = np.int(np.ceil(Landmarks_Y.iloc[i,1:22].max()))



        size = max(right-left, bottom-top)



        if right-left < size:

            extra = (size-right+left)//2

            left = max(left-extra, 0)

            right = min(right+extra, image.shape[1])



        else:

            extra = (size-bottom+top)//2

            top = max(top-extra, 0)

            bottom = min(bottom+extra, image.shape[0])



        try:

            image_1 = image[top:bottom, left:right, :]

            image_2 = cv2.imencode('.jpg', np.array(image_1))[1]

            output.writestr('Image_'+str(i)+'.jpg', image_2)

            

            x_mean.append(np.array(image_1, float).mean())

            x_std.append((np.array(image_1, float)**2).mean())

            

            Landmarks_X.loc[i, 'Image'] = 'Image_'+str(i)+'.jpg'

            Landmarks_Y.loc[i, 'Image'] = 'Image_'+str(i)+'.jpg'

            

            Landmarks_X.loc[i, 'Left_Eye_Center_X'] = Landmarks_X.iloc[i,8]-left

            Landmarks_Y.loc[i, 'Left_Eye_Center_Y'] = Landmarks_Y.iloc[i,8]-top

            

            Landmarks_X.loc[i, 'Right_Eye_Center_X'] = Landmarks_X.iloc[i,11]-left

            Landmarks_Y.loc[i, 'Right_Eye_Center_Y'] = Landmarks_Y.iloc[i,11]-top

            

        except:

            print(Landmarks_X.iloc[i, 0])

        

Landmarks_X.to_csv('FRONT-FACE-AFLW-X-1.csv', index=False)

Landmarks_Y.to_csv('FRONT-FACE-AFLW-Y-1.csv', index=False)
np.array(x_mean).mean(), np.sqrt(np.array(x_std).mean() - np.array(x_mean).mean()**2)
Landmarks_X[Landmarks_X[0] == '0/image10200']