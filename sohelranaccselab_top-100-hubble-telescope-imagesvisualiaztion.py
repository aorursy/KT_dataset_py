# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import numpy as np

from spectral import get_rgb, ndvi

from skimage import io

from sklearn.preprocessing import MinMaxScaler

from PIL import Image

from glob import glob

from tqdm import tqdm





def read_tif_to_jpg(img_path):

    img = io.imread(img_path)

    img2 = get_rgb(img, [2, 1, 0]) # RGB



    # rescaling to 0-255 range - uint8 for display

    rescaleIMG = np.reshape(img2, (-1, 1))

    scaler = MinMaxScaler(feature_range=(0, 255))

    rescaleIMG = scaler.fit_transform(rescaleIMG) # .astype(np.float32)

    img2_scaled = (np.reshape(rescaleIMG, img2.shape)).astype(np.uint8)

    

    return img2_scaled

img2 = read_tif_to_jpg('/kaggle/input/top-100-hubble-telescope-images/heic0822b.tif')

plt.imshow(img2)
from PIL import Image
def display_Image(path, save):

    img = Image.open(path)

    display(img)

    if save == True:

        img.save('Hubble__.jpg')
display_Image('/kaggle/input/top-100-hubble-telescope-images/heic0814a.tif', 0)
display_Image('/kaggle/input/top-100-hubble-telescope-images/opo0006a.tif',0)
display_Image('/kaggle/input/top-100-hubble-telescope-images/heic0822b.tif',True)
display_Image('/kaggle/input/top-100-hubble-telescope-images/heic1406a.tif',0)
display_Image('/kaggle/input/top-100-hubble-telescope-images/heic1410a.tif',0)