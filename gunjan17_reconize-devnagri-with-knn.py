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
#importing the dataset

data = pd.read_csv("../input/dataset.csv")
#importing the liberary

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

import seaborn as sns

from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline
#setig the no of column display

pd.set_option('display.max_columns',5000)
data.head()
#seting the data X,y

X =data.iloc[:,:-1]

Y = data.iloc[:,-1]
#removing the unwanted values

data.iloc[:,1024].value_counts()

rows_to_remove = np.where(data.iloc[:,1024].values==1024)

data = data.drop(data.index[rows_to_remove[0]])

char = pd.read_csv("../input/charlist.csv")
#lets see some images

i=1

a= Y[i]

Char_value = np.where(char.iloc[:,0].values==1)

img=X.iloc[i].as_matrix()

img=img.reshape((32,32))

plt.imshow(img,cmap='gray')

plt.title(Y[i])




