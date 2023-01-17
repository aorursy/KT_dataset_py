# let us install gdcm library 

!conda install -c conda-forge gdcm -y
# For example, here's several helpful packages to load

import pandas as pd 

import numpy as np 

#plotly

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

import matplotlib.pyplot as plt

#color

from colorama import Fore, Back, Style



import seaborn as sns

sns.set(style="whitegrid")



#pydicom

import pydicom



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



# Settings for pretty nice plots

plt.style.use('fivethirtyeight')

plt.show()





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import pydicom as dcm

import plotly.express as px



import seaborn as sns

import glob

import gdcm

from matplotlib import animation, rc

%matplotlib inline

import matplotlib

%matplotlib inline

matplotlib.use("Agg")



import matplotlib.animation as animation



TRAIN_DIR = "../input/osic-pulmonary-fibrosis-progression/train/"

files = glob.glob('../input/osic-pulmonary-fibrosis-progression/train/*/*/*.dcm')



rc('animation', html='jshtml')
import plotly

plotly.offline.init_notebook_mode (connected = True)
train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')
train.head()
def bar_plot(column_name):

    ds = train[column_name].value_counts().reset_index()

    ds.columns = ['Values', 'Total Number']

    fig = px.bar(

        ds, 

        x='Values', 

        y="Total Number", 

        orientation='v',

        title='Bar plot of: ' + column_name,

        width=700,

        height=400

    )

    fig.show()
col = train.columns

col
bar_plot('Weeks')

bar_plot('Sex')

bar_plot('SmokingStatus')

bar_plot('Age')

fig = px.scatter(train, x="FVC", y="Age", color='Sex')

fig.show()
fig = px.scatter(train, x="FVC", y="Percent", color='Age')

fig.show()
fig = px.scatter(train, x="FVC", y="Weeks", color='SmokingStatus')

fig.show()
fig = px.scatter(train, x="Weeks", y="Age", color='Sex')

fig.show()


fig = px.violin(train, y='Percent', x='SmokingStatus', box=True, color='Sex', points="all",

          hover_data=train.columns)

fig.show()
fig = px.box(train, x="Sex", y="Age", points="all")

fig.show()
corr = train.corr()

corr.style.background_gradient(cmap='coolwarm')
scans = glob.glob('/kaggle/input/osic-pulmonary-fibrosis-progression/train/*/')
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
movie = animate(read_scan(scans[12]))
movie
import pydicom

# https://www.kaggle.com/yeayates21/osic-simple-image-eda



imdir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"

print("total images for patient ID00123637202217151272140: ", len(os.listdir(imdir)))



# view first (columns*rows) images in order

fig=plt.figure(figsize=(12, 12))

columns = 4

rows = 5

imglist = os.listdir(imdir)

for i in range(1, columns*rows +1):

    filename = imdir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array)

plt.show()