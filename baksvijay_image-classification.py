# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input, decode_predictions

import PIL

from PIL import Image

from urllib.request import urlopen
model = ResNet50(weights='imagenet')
img_path = Image.open(urlopen('https://www.marylandzoo.org/wp-content/uploads/2018/04/lemuranimaheader3.jpg'))

img_path
basewidth = 224

hsize = 224

img = img_path.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
img_path = Image.open(urlopen('https://www.thescottishsun.co.uk/wp-content/uploads/sites/2/2019/04/NINTCHDBPICT000316151681.jpg'))

img_path
basewidth = 224

hsize = 224

img = img_path.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)
preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])
from keras.models import save_model
model.save('classification_model.h5')