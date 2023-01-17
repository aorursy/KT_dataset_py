# let us install gdcm library 

!conda install -c conda-forge gdcm -y
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

import pydicom as dcm

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

import glob

import gdcm

from matplotlib import animation, rc



import matplotlib

%matplotlib inline

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import matplotlib.animation as animation

TRAIN_DIR = "../input/rsna-str-pulmonary-embolism-detection/train/"

files = glob.glob('../input/rsna-str-pulmonary-embolism-detection/train/*/*/*.dcm')



rc('animation', html='jshtml')

train = pd.read_csv('/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv')

test = pd.read_csv('/kaggle/input/rsna-str-pulmonary-embolism-detection/test.csv')
train.head()
test.head()

def bar_plot(column_name):

    ds = train[column_name].value_counts().reset_index()

    ds.columns = ['Values', 'Total Number']

    fig = px.bar(

        ds, 

        y='Values', 

        x="Total Number", 

        orientation='h', 

        title='Bar plot of: ' + column_name,

        width=600,

        height=400

    )

    fig.show()
col = train.columns

col
col[0+3]
len(col)-3
for i in range(len(col)-3):

    bar_plot(col[i+3])
# drop the first column ('sig_id'), and 

df = train.drop(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

df.head()


df.columns = ['column', 'nonzero_records']

fig = px.bar(

    df, 

    y='nonzero_records', 

    x='column', 

    orientation='v', 

    title='Columns and non zero samples', 

    height=500, 

    width=1000

)

fig.show()



# drop the first column ('sig_id') and count the 0s in 

df1 = train.drop(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

df1.columns = ['column', 'zero_records']

df1['zero_records'] = len(train) -  df1['zero_records']

# plot the bar 



fig = px.bar(

    df1.head(50), 

    y='zero_records', 

    x='column', 

    orientation='v', 

    title='Columns with the zero samples ', 

    height=500, 

    width=1000

)

fig.show()
corr = train.corr()

corr.style.background_gradient(cmap='coolwarm')
scans = glob.glob('/kaggle/input/rsna-str-pulmonary-embolism-detection/train/*/*/')

def read_scan(path):

    fragments = glob.glob(path + '/*')

    

    slices = []

    for f in fragments:

        img = dcm.dcmread(f)

        img_data = img.pixel_array

        length = int(img.InstanceNumber)

        slices.append((length, img_data))

    slices.sort()

    return [s[1] for s in slices]



def animate(ims):

    fig = plt.figure(figsize=(11,11))

    plt.axis('off')

    im = plt.imshow(ims[0], cmap='gray')



    def animate_func(i):

        im.set_array(ims[i])

        return [im]



    anim = animation.FuncAnimation(fig, animate_func, frames = len(ims), interval = 1000//24)

    

    return anim
movie = animate(read_scan(scans[1]))
movie
movie.save('Test.gif', dpi=80, writer='imagemagick')