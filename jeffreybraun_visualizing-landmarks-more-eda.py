import numpy as np
import pandas as pd
import seaborn as sns
import plotly as py
import plotly_express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import folium
from folium import plugins
from plotly.offline import init_notebook_mode, iplot
import os
init_notebook_mode()

print("First 5 entries in train.csv: ")
df_train = pd.read_csv('/kaggle/input/landmark-recognition-2020/train.csv')
print(df_train.head())
print('\n')

print("Number of photos in train: ")
print(df_train.shape[0])
print('\n')

print("Number of unique landmarks: ")
print(df_train.landmark_id.nunique())
print('\n')

print("Number of photos in test: ")
test_count = sum(len(files) for _, _, files in os.walk(r'/kaggle/input/landmark-recognition-2020/test'))
print(test_count)
df_landmark = pd.DataFrame(df_train.landmark_id.value_counts()).reset_index()
df_landmark.rename(columns = {'index':'landmark_id', 'landmark_id':'num_photos'}, inplace=True)
df_landmark['landmark_id'] = df_landmark['landmark_id'].astype(str)

fig = px.bar(df_landmark[0:10], x = 'landmark_id', y = 'num_photos', hover_name = 'landmark_id', color = 'num_photos', title = 'Top 10 most frequent landmarks')
fig.update_layout(xaxis_type='category')
fig.show()


fig = px.bar(df_landmark[0:100], x = 'landmark_id', y = 'num_photos', hover_name = 'landmark_id', color = 'num_photos', title = 'Top 100 most frequent landmarks')
fig.update_layout(xaxis_type='category')
fig.show()

fig = px.violin(df_landmark, x = 'num_photos', title='Violin plot for landmark frequency')
fig.show()

fig = px.histogram(df_landmark, x = 'num_photos', title='Histogram for landmark frequency (truncated)')
fig.update_layout(yaxis_title='number of landmarks', xaxis=dict(range=[0,100]))
fig.show()


df_landmark.describe(percentiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95,0.99, 0.999])
from PIL import Image 

def get_file_path(photo_id, ttype):
    file_path = '/kaggle/input/landmark-recognition-2020/' + ttype
    file_path = file_path + '/' + photo_id[0] + '/' + photo_id[1] + '/' + photo_id[2] + '/' + photo_id + '.jpg'
    return file_path


def get_landmarks(landmark_ids, df_train, ttype, ninstances):
    df = pd.DataFrame([], columns=['id', 'landmark_id'])
    if ttype not in ['train', 'test']:
        print('Please enter a valid ttype: train or test')
        return df
    photos_url = "/kaggle/input/landmark-recognition-2020/" + ttype
    for lmark in landmark_ids:
        df_lmark = df_train[df_train.landmark_id == lmark]
        if len(df_lmark) < ninstances:
            print('Too many instances for landmark_id ' + str(lmark) + ' (there are only ' + str(len(df_lmark)) + ')')
            return df
        else:
            df_lmark = df_lmark[0:ninstances]
            df = pd.concat([df, df_lmark])
    df['file_path'] = df['id'].apply(lambda x: get_file_path(x, ttype))
    return df.reset_index()
    
    
def show_landmark(landmark_file, name):
    img_array = np.array(Image.open(landmark_file))
    plt.title(name)
    plt.imshow(img_array)
    plt.show()
    

def show_grid(landmark_id, rows, cols, df_train, ttype):
    landmark_ids = [landmark_id]
    instances = len(df_train[df_train.landmark_id == landmark_id])
    df_show_landmarks = get_landmarks(landmark_ids, df_train, ttype, rows*cols)
    fig, axs = plt.subplots(rows, cols, figsize = (rows*4,cols*4))
    count = 0
    for i in range(0,rows):
        for j in range(0,cols):
            landmark_file = df_show_landmarks.loc[count, 'file_path']
            img_array = np.array(Image.open(landmark_file))
            axs[i,j].imshow(img_array)
            count += 1
    plt.suptitle('Instances of landmark_id: '+ str(landmark_id) + ', which has ' + str(instances) + ' instances in the data.')
    plt.show()
    

landmark_ids = list(df_landmark[0:10]['landmark_id'].astype(int))
for landmark_id in landmark_ids:
    show_grid(landmark_id, 5, 5, df_train, 'train')
    

landmark_ids = list(df_landmark[15452:15462]['landmark_id'].astype(int))
for landmark_id in landmark_ids:
    show_grid(landmark_id, 5, 5, df_train, 'train')
    

landmark_ids = list(df_landmark[64016:64026]['landmark_id'].astype(int))
for landmark_id in landmark_ids:
    show_grid(landmark_id, 2, 2, df_train, 'train')
    
from tqdm._tqdm_notebook import tqdm_notebook
import os
import struct
import io
import warnings
warnings.filterwarnings('ignore')

def get_image_size(file_path):
    """
    :Name:        get_image_size
    :Purpose:     extract image dimensions given a file path
    :Author:      Paulo Scardine (based on code from Emmanuel VAÏSSE)
    :Created:     26/09/2013
    :Copyright:   (c) Paulo Scardine 2013
    :Licence:     MIT
    
    """
    size = os.path.getsize(file_path)
    with io.open(file_path, "rb") as input:
        height = -1
        width = -1
        data = input.read(26)
        input.seek(0)
        input.read(2)
        b = input.read(1)
        while (b and ord(b) != 0xDA):
            while (ord(b) != 0xFF):
                b = input.read(1)
            while (ord(b) == 0xFF):
                b = input.read(1)
            if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                input.read(3)
                h, w = struct.unpack(">HH", input.read(4))
                break
            else:
                input.read(int(struct.unpack(">H", input.read(2))[0]) - 2)
            b = input.read(1)
        width = int(w)
        height = int(h)
        return [width, height]
    

def get_width(dim):
    return dim[0]


def get_height(dim):
    return dim[1]

NUM_SUBSET = 100000

df_train_subset = df_train[0:NUM_SUBSET]
df_train_subset['file_path'] = df_train_subset['id'].apply(lambda x: get_file_path(x, 'train'))
df_train_subset['dimension'] = df_train_subset['file_path'].apply(lambda x: get_image_size(x))
df_train_subset['width'] = df_train_subset['dimension'].apply(lambda x: get_width(x))
df_train_subset['height'] = df_train_subset['dimension'].apply(lambda x: get_height(x))
df_train_subset['total_pixels'] = df_train_subset['width'] * df_train_subset['height']
df_train_subset['ratio'] = df_train_subset['width'] / df_train_subset['height']
df_train_subset['ratio'] = df_train_subset['width'] / df_train_subset['height']
fig = px.scatter(df_train_subset, x = 'width', y = 'height', hover_name='dimension', color='total_pixels', title='Image dimension (first 100,000 images)')
fig.show()

fig = px.histogram(df_train_subset, x = 'total_pixels', title='Total Pixels per Image (first 100,000 images)')
fig.show()

fig = px.scatter(df_train_subset, x = 'width', y = 'height', hover_name='dimension', color='ratio', range_color=(0.5,1.5), title='Image dimension (first 100,000 images)')
fig.show()

fig = px.histogram(df_train_subset, x = 'ratio', title='Image Ratio (first 100,000 images)')
fig.show()