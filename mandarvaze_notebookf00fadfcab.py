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
import pandas as pd

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

%matplotlib inline

labeled_images = pd.read_csv('../input/train.csv')

#labeled_images.head()

a=labeled_images.iloc[3,1:].values

a=a.reshape(28,28).astype('uint8')

#plt.imshow(a)

df_x=data.iloc[:,1:]

df_y=data.iloc[:,0]

df_x.head()

#images = labeled_images.iloc[0:5000,1:]

#labels = labeled_images.iloc[0:5000,:1]

#train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

#i=1

#img=train_images.iloc[i].as_matrix()

#img=img.reshape((28,28))

#plt.imshow(img,cmap='gray')

#plt.title(train_labels.iloc[i,0])