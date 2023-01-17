import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from os import listdir

import matplotlib.pyplot as plt

from PIL import Image

from tqdm import tqdm 
data_dir = '../input/uci-shoulder-implant-xray-manufacturer/data/'

image_paths = [data_dir + f for f in listdir(data_dir)]

image_paths
def get_class_title(i):

    if 'Depuy' in image_paths[i]:

        return 'Depuy'

    elif 'Zimmer' in image_paths[i]:

        return 'Zimmer'

    elif 'Tornier' in image_paths[i]:

        return 'Tornier'

    else:

        return 'Cofield'
train_df = pd.DataFrame(listdir(data_dir), columns = ['Image Files'])

train_df['Label'] = 'NA'



for i in range(train_df.shape[0]):

    train_df.iloc[i, 1] = get_class_title(i)

train_df.head()



train_df['Label'].value_counts().plot.bar()

plt.grid(True)

plt.title('Relative Class Frequency')

plt.show()
fig=plt.figure(figsize=(20, 15))

columns = 10; rows = 7

for i in tqdm(range(1, columns*rows +1)):

    img = Image.open(image_paths[i])

    fig.add_subplot(rows, columns, i)

    plt.imshow(img, cmap = 'gray')

    plt.title(get_class_title(i) + str(img.size))

    plt.axis(False)

    fig.add_subplot