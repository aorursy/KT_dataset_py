# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
WIDTH=137

HEIGHT=236

BATCHSIZE = 36

VALIDATION_SIZE = .2
train = pd.read_csv("/kaggle/input/bengaliai-cv19/train.csv")

p1 = pd.read_parquet("/kaggle/input/bengaliai-cv19/test_image_data_0.parquet")
parquet_file = "/kaggle/input/bengaliai-cv19/train_image_data_1.parquet"



# def process_parquet(parquet_file):

parquet = pd.read_parquet(parquet_file).iloc[:,:]

# print(parquet.shape)

labels = pd.DataFrame(parquet.iloc[:,0]).merge(train, on = 'image_id', how = 'left')

img = parquet.iloc[:,1:].values.reshape(-1, WIDTH, HEIGHT)#.astype(float)



print(img.shape)

print(labels.shape)
rows = 3

columns = 3

n_plots = rows*columns



f, ax = plt.subplots(rows,columns, figsize = [25,15])

for i, img_i in enumerate(np.random.choice(img.shape[0], n_plots)):

    ax = plt.subplot(rows, columns, i+1)

    ax.imshow(img[img_i,:,:], cmap='gray', vmin=0, vmax=255)

    ax.set_title(" ".join([f"{x1}: {x2}\n" for x1,x2 in labels.iloc[img_i,:].to_dict().items()]))

plt.tight_layout(pad=0)

plt.show()