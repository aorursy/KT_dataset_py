# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# loading the training dataset

df = pd.read_csv('../input/train.csv')

df.shape
df.head()
# Converting the DataFrame to a numpy array

M = df.as_matrix()

type(M)
# let's try to visualize the first row as an image using matplotlib library

# Let's discard the 'label' (first column)

first_image = M[0,1:]
# checking the dimension

first_image.shape
# reshaping and converting the row in a 28x28 matrix

first_image = first_image.reshape(28,28)

first_image.shape
import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.imshow(255 - first_image, cmap='gray')

plt.grid()

plt.show()
df.head(1).label
M[0,0]
fourth_image = M[3,1:]



fourth_image = fourth_image.reshape(28, 28)

plt.imshow(255 - fourth_image, cmap='gray')

plt.grid()

plt.show()
M[3,0]