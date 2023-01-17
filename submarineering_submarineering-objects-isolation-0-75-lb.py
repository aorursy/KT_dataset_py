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

    image = gaussian_filter(image,2)

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
# Indexes for ships or icebergs.

index_ship=np.where(train['is_iceberg']==0)

index_ice=np.where(train['is_iceberg']==1)
# For ploting

def plots(band,index,title):

    plt.figure(figsize=(12,10))

    for i in range(12):

        plt.subplot(3,4,i+1)

        plt.xticks(())

        plt.yticks(())

        plt.xlabel((title))

        plt.imshow(np.reshape(train[band][index[0][i]], (75,75)),cmap='gist_heat')

    plt.show()  
plots('band_1',index_ship,'band1 ship')
plots('band_1',index_ice,'band1 iceberg')
plots('iso1',index_ship,'iso1 ship')
plots('iso1',index_ice,'iso1 iceberg')
# Additional features from the morphological analysis and how is working on discrimination.

train[train.is_iceberg==1]['iso1'].apply(np.max).plot(alpha=0.4)

train[train.is_iceberg==0]['iso1'].apply(np.max).plot(alpha=0.4)
# Additional features from the morphological analysis and how is working on discrimination.

train[train.is_iceberg==1]['iso2'].apply(np.max).plot(alpha=0.4)

train[train.is_iceberg==0]['iso2'].apply(np.max).plot(alpha=0.4)