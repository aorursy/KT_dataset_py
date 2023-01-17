# The usual

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
# Read in the data

data = pd.read_csv("../input/dataset.csv")



# There are some bad images that aren't characters. Let's remove them

bad_chars = np.where(data.iloc[:, 1024].values == 1024)

data = data.drop(data.index[bad_chars[0]])
# Let's also only select out digits (according to charlist.csv)

digits = np.where(data.iloc[:, 1024].values < 36)

data = data.drop(data.index[digits[0]])
# Let's sort and make sure indices are right

data = data.sample(frac=1).reset_index(drop=True)

data.sort_values(by = ['1024'], ascending = True, inplace = True)

data = data.reset_index(drop = True)



# Since we selected digits, sorting means we should see all the digits 0 to 9, in that order
# Extract the pixel and label info

pixels = data.iloc[: , : -1]

labels = data.iloc[: , -1]

pixels = pixels.values.reshape(pixels.shape[0], 32, 32)
# Plot 8 images from every group of 1000

for j in range(1, 20000, 1000):

    for i in range(j+1, j+9):    

        plt.subplot(240+i-j)

        plt.axis('off')

        plt.imshow(pixels[i-1], cmap=plt.get_cmap('gray'))

        plt.title(labels[i-1]);

    plt.show()
