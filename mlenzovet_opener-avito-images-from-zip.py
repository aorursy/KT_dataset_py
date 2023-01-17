import zipfile

from PIL import Image

import matplotlib.pyplot as plt



from keras.preprocessing import image





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('../input/avito-demand-prediction/train.csv', usecols=['price','image'], nrows=100)

data
%%time

imgzip = zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip')

inflist = imgzip.infolist()
def im_from_zip(template, imgzip, name):

    f = template.format(name)

    print(f)

    ifile = imgzip.open(f)

    img = Image.open(ifile)

    print(img)

    plt.imshow(img)

    plt.show()

    return img





template = 'data/competition_files/train_jpg/{}.jpg'

name = data.image[0]

im_from_zip(template, imgzip, name)
