# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

from skimage import img_as_float

from skimage.morphology import reconstruction

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_json('../input/train.json')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load the training dataset.

train = pd.read_json('../input/train.json')
# Isolation function.

def iso(arr):

    image = img_as_float(np.reshape(np.array(arr), [75,75]))

    image = gaussian_filter(image,2.5)

    seed = np.copy(image)

    seed[1:-1, 1:-1] = image.min()

    mask = image 

    dilated = reconstruction(seed, mask, method='dilation')

    return image-dilated

# Plotting to compare

arr = train.band_1[12]

dilated = iso(arr)

fig, (ax0, ax1) = plt.subplots(nrows=1,

                                    ncols=2,

                                    figsize=(16, 5),

                                    sharex=True,

                                    sharey=True)



ax0.imshow(np.reshape(np.array(arr), [75,75]))

ax0.set_title('original image')

ax0.axis('off')

ax0.set_adjustable('box-forced')



ax1.imshow(dilated, cmap='gray')

ax1.set_title('dilated')

ax1.axis('off')

ax1.set_adjustable('box-forced')

# Plotting to compare

arr = train.band_1[8]

dilated = iso(arr)

fig, (ax0, ax1) = plt.subplots(nrows=1,

                                    ncols=2,

                                    figsize=(16, 5),

                                    sharex=True,

                                    sharey=True)



ax0.imshow(np.reshape(np.array(arr), [75,75]))

ax0.set_title('original image')

ax0.axis('off')

ax0.set_adjustable('box-forced')



ax1.imshow(dilated, cmap='gray')

ax1.set_title('dilated')

ax1.axis('off')

ax1.set_adjustable('box-forced')
# Feature engineering iso1 and iso2.

train['iso1'] = train.iloc[:, 0].apply(iso)

train['iso2'] = train.iloc[:, 1].apply(iso)
import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

img = train.iso1[12]+train.iso2[12]

data = [

    go.Surface(

        z=img

    )

]

layout = go.Layout(

    title='Iceberg to 3D Surface',

    autosize=False,

    width=500,

    height=500,

    margin=dict(

        l=65,

        r=50,

        b=65,

        t=90

    )

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
img = train.iso1[8] + train.iso2[8]

data = [

    go.Surface(

        z= img

    )

]

layout = go.Layout(

    title='Ship to 3D Surface',

    autosize=False,

    width=500,

    height=500,

    margin=dict(

        l=65,

        r=50,

        b=65,

        t=90

    )

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
#Adding up every pixel value is equivalent to the volumen under the surface. 

def volume(arr):

    return np.sum(arr)
train['vol'] = (train.iloc[:, 0] + train.iloc[:, 1]).apply(volume)
# Additional features from the morphological analysis and how is working on discrimination.

train[train.is_iceberg==1]['iso1'].apply(np.max).plot(alpha=0.4)

train[train.is_iceberg==0]['iso1'].apply(np.max).plot(alpha=0.4)
# Additional features from the morphological analysis and how is working on discrimination.

train[train.is_iceberg==1]['iso2'].apply(np.max).plot(alpha=0.4)

train[train.is_iceberg==0]['iso2'].apply(np.max).plot(alpha=0.4)
# Volume. Additional features from the morphological analysis and how is working on discrimination.

train[train.is_iceberg==1]['vol'].plot(alpha=0.4)

train[train.is_iceberg==0]['vol'].plot(alpha=0.4)
