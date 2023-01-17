import numpy as np

import pandas as pd

from os import listdir

from glob import glob



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
train_data = pd.read_csv('../input/landmark-recognition-2020/train.csv')

submission = pd.read_csv("../input/landmark-recognition-2020/sample_submission.csv")

print('Training dataframe shape: ', train_data.shape)

print('Total number of training images', train_data.shape[0])

print('Total number of test images', len(glob('../input/landmark-recognition-2020/test/*/*/*/*.jpg')))

train_data.head()
# Let's have a look at the sample submission file. 

submission.head()
train_data['landmark_id'].value_counts().hist()
# missing data in training data 

total = train_data.isnull().sum().sort_values(ascending = False)

print(total)

percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)

missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head()
most_frequent = 10

# Occurance of landmark_id in decreasing order(Top categories)

temp = pd.DataFrame(train_data.landmark_id.value_counts().head(most_frequent)).copy()

temp.reset_index(inplace=True)

temp.columns = ['landmark_id','count']

temp.sort_values(by='count', ascending=False, inplace = True)

temp.set_index('landmark_id', inplace = True)

temp.plot.bar()

most_ids = temp.index.copy()
least_frequent = 10

# Occurance of landmark_id in decreasing order(Top categories)

temp = pd.DataFrame(train_data.landmark_id.value_counts().tail(least_frequent)).copy()

temp.reset_index(inplace=True)

temp.columns = ['landmark_id','count']

temp.sort_values(by='count', ascending=False, inplace = True)

temp.set_index('landmark_id', inplace = True)

print(temp)

temp.plot.bar()

least_ids = temp.index.copy()
train_data[train_data['landmark_id'] == 110417].reset_index()['id']


def view_images(ID=110417):

    import PIL

    from PIL import Image

#     ID = 110417

#     Generating the filepaths 

    image_file_names = train_data[train_data['landmark_id'] == ID].reset_index()['id']

#     print(image_file_names)

    image_paths = []

    for image_name in image_file_names:

        sub_folder = image_name[0] + '/'+ image_name[1] + '/' +  image_name[2] + '/'

        image_paths.append('../input/landmark-recognition-2020/train/' + sub_folder + image_name + '.jpg')

#     print(image_paths)







    grid_size = 4



    rows = grid_size

    cols = grid_size

    fig = plt.figure(figsize=(grid_size*3+2, grid_size*3))

    for i in range(1, rows*cols+1):

        if(i>len(image_paths)):

            break

        fig.add_subplot(rows, cols, i)

        plt.imshow(Image.open(image_paths[i-1]))

        plt.title('ID =' + str(ID), fontsize=16)

        plt.axis(False)

        fig.add_subplot

    plt.show()
from ipywidgets import interact

interact(view_images, ID = most_ids)

interact(view_images, ID = least_ids)
view_images(least_ids[0])
view_images(least_ids[1])

view_images(least_ids[2])

view_images(least_ids[3])

view_images(least_ids[0])

view_images(least_ids[4])

view_images(least_ids[5])

view_images(most_ids[0])
view_images(most_ids[1])
view_images(most_ids[2])
view_images(most_ids[3])
view_images(most_ids[4])