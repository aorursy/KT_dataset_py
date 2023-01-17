import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os 

from scipy import ndimage

from subprocess import check_output



import cv2





pal = sns.color_palette()



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



print('# File sizes')

for f in os.listdir('../input'):

    if not os.path.isdir('../input/' + f):

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')

    else:

        sizes = [os.path.getsize('../input/'+f+'/'+x)/1000000 for x in os.listdir('../input/' + f)]

        print(f.ljust(30) + str(round(sum(sizes), 2)) + 'MB' + ' ({} files)'.format(len(sizes)))
df_train_labels = pd.read_csv('../input/train_labels.csv')

df_train_labels.head()
df_train_labels.describe()

print ("Invasive")

print (df_train_labels[df_train_labels.invasive == 1].describe())

print ()

print ("Non-Invasive")

print (df_train_labels[df_train_labels.invasive == 0].describe())
import cv2



new_style = {'grid': False}

plt.rc('axes', **new_style)

_, ax = plt.subplots(4, 4, sharex='col', sharey='row', figsize=(20, 20))

i = 0

for f, l in df_train_labels[:16].values:

    img = cv2.imread('../input/train/{}.jpg'.format(f))

    ax[i // 4, i % 4].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[i // 4, i % 4].set_title('{} - {}'.format(f, l))

    

    i += 1

    

plt.show()
#So the pictures that have a flower are '1' and are invasive.

#The pictures with no flowers are '0' and are not of interest to us.
# Plots # to add more analysis later
#img_rows, img_cols= 866,1154

#im_array = cv2.imread('../input/train/3.jpg',0)

#template = np.zeros([ img_rows, img_cols], dtype='uint8') # initialisation of the template

#template[:, :] = im_array[0:866,0:1154] # 



#plt.subplots(figsize=(10, 7))

#plt.subplot(121),plt.imshow(template, cmap='gray') 

#plt.subplot(122), plt.imshow(im_array, cmap='gray')