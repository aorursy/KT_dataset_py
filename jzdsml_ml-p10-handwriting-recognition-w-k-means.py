# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

from matplotlib import pyplot as plt



from sklearn import datasets



digits = datasets. load_digits()

print(digits)



print(digits.DESCR)

#The digit images are 8 x 8. And the dataset is from Bogazici University (Istanbul, Turkey).



print(digits.data)



#Each list contains 64 values which respent the pixel colors of an image (0-16). 0 is white; 16 is black.



print(digits.target)

#Printed result shows us that the first data point in the set was tagged as a 0 and the last one was tagged as an 8.



#Lets visualize the image at index 100

plt.gray() 

plt.matshow(digits.images[100])

plt.show()



#Is it a 4? Lets print out the target label at index 100 to find out!

print(digits.target[100])





#Take a look at 64 sample images.



# Figure size (width, height)



fig = plt.figure(figsize=(6, 6))



# Adjust the subplots 



fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



# For each of the 64 images



for i in range(64):



    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position



    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])



    # Display an image at the i-th position



    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')



    # Label the image with the target value



    ax.text(0, 7, str(digits.target[i]))



plt.show()







from sklearn.cluster import KMeans

model = KMeans(n_clusters=10, random_state=42)

model.fit(digits.data)





#Visualize all the centroids

fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):



  # Initialize subplots in a grid of 2X5, at i+1th position

  ax = fig.add_subplot(2, 5, 1 + i)



  # Display images

  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()





new_samples = np.array([

[0.00,0.00,0.03,0.40,0.63,0.18,0.00,0.00,0.00,1.00,2.09,2.13,2.13,1.89,0.00,0.00,0.00,1.57,2.11,1.64,2.05,2.14,0.00,0.00,0.00,0.16,0.32,0.00,1.91,2.14,0.00,0.00,0.00,0.00,0.00,0.69,2.14,2.06,0.00,0.00,0.00,0.00,0.31,2.07,2.14,1.24,0.43,0.14,0.00,0.00,1.36,2.13,2.13,2.14,2.18,1.56,0.00,0.00,0.87,1.71,1.61,1.34,1.30,0.69],

[0.00,0.00,0.85,1.02,0.85,0.54,0.00,0.00,0.00,0.86,2.15,2.19,2.18,2.16,1.29,0.00,0.00,1.44,2.15,1.16,1.05,2.00,2.15,0.62,0.00,1.48,2.15,0.42,0.00,1.13,2.15,0.85,0.00,1.48,2.15,0.42,0.02,1.77,2.16,0.66,0.00,1.44,2.14,1.06,0.87,2.16,1.82,0.04,0.00,0.66,2.14,2.14,2.13,2.14,0.69,0.00,0.00,0.00,0.61,1.39,1.48,1.00,0.00,0.00],

[0.00,0.02,0.88,2.07,1.60,0.00,0.00,0.00,0.00,0.75,2.15,2.14,2.16,0.20,0.00,0.00,0.00,0.14,1.04,1.94,2.16,0.21,0.00,0.00,0.00,0.00,0.00,1.71,2.15,0.21,0.00,0.00,0.00,0.00,0.00,1.71,2.15,0.20,0.00,0.00,0.00,0.00,0.00,1.71,2.15,0.20,0.00,0.00,0.00,0.00,0.00,1.74,2.18,0.20,0.00,0.00,0.00,0.00,0.00,1.05,1.53,0.02,0.00,0.00],

[0.00,0.94,2.08,2.14,2.02,0.31,0.00,0.00,1.05,2.17,2.13,1.80,2.16,1.39,0.00,0.00,1.88,2.15,0.54,0.56,2.16,1.70,0.00,0.00,1.87,2.14,1.18,1.50,2.16,1.71,0.00,0.00,0.63,2.11,2.16,2.16,2.15,1.71,0.00,0.00,0.00,0.46,0.86,1.18,2.16,1.67,0.00,0.00,0.00,0.47,0.80,1.60,2.14,1.04,0.00,0.00,0.00,1.89,2.18,2.17,2.05,0.13,0.00,0.00]

])



new_labels = model.predict(new_samples)



#Wait, because this is a clustering algorithm, we don’t know which label is which.



#By looking at the cluster centers, let’s map out each of the labels with the digits we think it represents.



print(new_labels)



for i in range(len(new_labels)):

  if new_labels[i] == 0:

    print(0, end='')

  elif new_labels[i] == 1:

    print(9, end='')

  elif new_labels[i] == 2:

    print(2, end='')

  elif new_labels[i] == 3:

    print(1, end='')

  elif new_labels[i] == 4:

    print(6, end='')

  elif new_labels[i] == 5:

    print(8, end='')

  elif new_labels[i] == 6:

    print(4, end='')

  elif new_labels[i] == 7:

    print(5, end='')

  elif new_labels[i] == 8:

    print(7, end='')

  elif new_labels[i] == 9:

    print(3, end='')

    


