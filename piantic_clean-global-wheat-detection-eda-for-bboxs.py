import cv2

import math

import numpy as np

import scipy as sp

import pandas as pd



import glob 

from PIL import Image, ImageDraw

import matplotlib.pyplot as plt



import plotly.graph_objects as go

import plotly.express as px

import plotly.figure_factory as ff
train_df = pd.read_csv('../input/gwd-clean-train/original_train.csv')

clean_df = pd.read_csv('../input/gwd-clean-train/new_train_0805.csv')
train_image_path = "../input/global-wheat-detection/train/"

test_image_path = "../input/global-wheat-detection/test/"
train_df.head()
clean_df.head()
train_df['image_id'].nunique()
clean_df['image_id'].nunique()
def show_box(df, image_id, color='red'):

    df = df.where(df['image_id']== image_id)

    df = df.dropna(axis='rows')

    arr = df["bbox"].to_numpy()



    image = cv2.imread(f'{train_image_path}/{image_id}.jpg')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box in arr:

        box=box[1:-1]

        xmin,ymin,width,height= box.split(",")

 

        xmin = int(float(xmin))

        ymin= int(float(ymin)) 

        width = int(float(width))

        height= int(float(height))



        xmax = xmin + width

        ymax = ymin + height

        

        color_tuple = (255,0,0)

        if color == 'blue':

            color_tuple = (0,0,255)

            

        image = cv2.rectangle(image,(xmin,ymin), (xmax,ymax),color_tuple,3)

        

        img = Image.fromarray(image)

    return img
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    

ax.imshow(show_box(train_df, '41c0123cc'))
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    

ax.imshow(show_box(clean_df, '41c0123cc', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '41c0123cc'))

ax[1].imshow(show_box(clean_df, '41c0123cc', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '2cc75e9f5'))

ax[1].imshow(show_box(clean_df, '2cc75e9f5', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'c01a58fdb'))

ax[1].imshow(show_box(clean_df, 'c01a58fdb', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'c631c7fdb'))

ax[1].imshow(show_box(clean_df, 'c631c7fdb', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '61ff5cdc2'))

ax[1].imshow(show_box(clean_df, '61ff5cdc2', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'c6d94be4c'))

ax[1].imshow(show_box(clean_df, 'c6d94be4c', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '8bad19780'))

ax[1].imshow(show_box(clean_df, '8bad19780', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '4cbb2b7bd'))

ax[1].imshow(show_box(clean_df, '4cbb2b7bd', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'fade0e053'))

ax[1].imshow(show_box(clean_df, 'fade0e053', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '9a30dd802'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'b53afdf5c'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'd96205316'))

ax[1].imshow(show_box(clean_df, 'd96205316', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'dc7c60052'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '6106eefbc'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '83842ec14'))

ax[1].imshow(show_box(clean_df, '83842ec14', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '42e6efaaa'))

ax[1].imshow(show_box(clean_df, '42e6efaaa', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'fc6860020'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '9780d64f5'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '49c3e4f6e'))

ax[1].imshow(show_box(clean_df, '49c3e4f6e', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '409a8490c'))

ax[1].imshow(show_box(clean_df, '409a8490c', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'd7a02151d'))

ax[1].imshow(show_box(clean_df, 'd7a02151d', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'b53afdf5c'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'a1321ca95'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'a36608629'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '5ec31deb1'))

ax[1].imshow(show_box(clean_df, '5ec31deb1', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '834690a35'))

ax[1].imshow(show_box(clean_df, '834690a35', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'f5f5a9d30'))

ax[1].imshow(show_box(clean_df, 'f5f5a9d30', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'd8cae4d1b'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'd067ac2b1'))

ax[1].imshow(show_box(clean_df, 'd067ac2b1', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '69e509038'))

ax[1].imshow(show_box(clean_df, '69e509038', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'f24698e88'))

ax[1].imshow(show_box(clean_df, 'f24698e88', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '4b4f6de9b'))

ax[1].imshow(show_box(clean_df, '4b4f6de9b', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '91c7fb84e'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '7d5af5b74'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '7c659d49a'))

ax[1].imshow(show_box(clean_df, '7c659d49a', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, 'd60e832a5'))

ax[1].imshow(show_box(clean_df, 'd60e832a5', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '595709e55'))

ax[1].imshow(show_box(clean_df, '595709e55', 'blue'))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    

ax[0].imshow(show_box(train_df, '893938464'))

ax[1].imshow(show_box(clean_df, '893938464', 'blue'))