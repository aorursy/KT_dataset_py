# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 3. Import training dataset into a dataframe -D
import zipfile

with zipfile.ZipFile('../input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/Train')
filenames = os.listdir("/kaggle/working/Train/train")
filenames[0]
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
df.head()
# Data Augmenation using Python and Keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=40, width_shift_range= 0.2,
                             height_shift_range= 0.2, shear_range= 0.2, 
                             zoom_range= 0.2, horizontal_flip= True,
                             fill_mode= 'nearest')
img = load_img('/kaggle/working/Train/train/dog.2937.jpg')
img
x = img_to_array(img); # Numpy array with shape - (3,150,150)
x = x.reshape((1,) +x.shape) # Numpy array with shape - (1, 3, 150, 150) -in 4 dim
# .flow() command generates batches of randomly transformed images and saves the results to 'preview/'command

i=0
for batch in datagen.flow(x, batch_size=1, save_to_dir='/kaggle/working/Train', save_prefix='dogt',save_format='.jpeg'):
    i+=1
    if i>20: # After creating 20 images, it will break
        break; 
