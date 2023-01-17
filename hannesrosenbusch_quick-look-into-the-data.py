#imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from PIL import Image
from matplotlib.pyplot import imshow
#read in data
labels = pd.read_csv('../input/names_and_strengths.csv', header = 'infer')
data = np.load('../input/poke_image_data.npy')
#How do the labels look like? Example:
labels.iloc[42]
#How do the images look like? Example:
example_pic = Image.fromarray(data[42], 'RGB')
imshow(example_pic)
#How many pokemon are there in the data?
print('There are {} different pokemon in the dataset' .format(len(set(labels.name))))
#How does the distribution of battle strength look like?
labels.strength.hist()