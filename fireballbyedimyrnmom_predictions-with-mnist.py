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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
from keras.models import Sequential

from keras.layers import Dense #This is a linear operation where every input is connected to every output by a weight.

from keras.layers import Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K
from keras.preprocessing import image

gen = image.ImageDataGenerator()

#same as: from keras.preprocessing.image import ImageDataGenerator
#test-data set excludes the label column

train.head(2)
test.head(2)
train['label'].unique()
train['label'].value_counts()
#check the size of both data sources

train.shape, test.shape #2-dimensions
#converts labels to integers and pixels into floats

X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # Labels, column 0, target 

X_test = test.values.astype('float32')
#Reshape by adding dimension for color channel 

X_train_4D = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test_4D = X_test.reshape(X_test.shape[0], 28, 28,1)

X_train_4D.shape, X_test_4D.shape
X_trainA = X_train_4D.reshape(X_train_4D.shape[0], 28, 28) ##This is important for images to show properly

# put labels into y_train variable
# visualize number of digits classes

import seaborn as sns

plt.figure(figsize=(15,7))

g = sns.countplot(y_train, palette="icefire")

plt.title("Number of classes")
#prints a different range (beware: range over 15 fails)

for i in range(10, 14):

    plt.subplot(330 + (i+1))

    plt.imshow(X_trainA[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i]);
#prints the digits in positions 0 to 5 (not the digit image)

for i in range(0, 5):

    plt.subplot(330 + (i+1))

    plt.imshow(X_trainA[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i]);
# shows the pixel values of the image

plt.figure()

plt.imshow(X_trainA[0])

plt.colorbar()

plt.grid(False)

plt.show()

# The label data as individual dataframe

Labls=train[['label']]

Labls.shape
Labls.head(2)
Labls['label'].unique()
#label as array

# The label data as individual set

arrayLbl=train['label']

arrayLbl.shape
#split the data 

from sklearn.model_selection import train_test_split

train_img, test_img, train_lbl, test_lbl = train_test_split(

    train, arrayLbl, test_size=28000, random_state=0)

from sklearn.linear_model import LogisticRegression

Lmodel = LogisticRegression(solver = 'lbfgs') #  =Limited-memory Broyden–Fletcher–Goldfarb–Shanno 

#solver = seeks parameter weights that minimize a cost function

#lbfgs solver= approximates the second derivative matrix updates with gradient evaluations 

# and stores only the last few updates to save memory 

#Source: https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451
#fit the model

Lmodel.fit(test_img, test_lbl) 
# Make predictions on entire test data

predictions = Lmodel.predict(test_img)

print(predictions)
predictions2 = Lmodel.predict(train_img)

print(predictions2)

    
Acc = Lmodel.score(test_img, test_lbl)

print(Acc)
df = pd.DataFrame(predictions, columns=['ImageId'])

df.head(2)
df.shape
S=pd.concat([df, Labls], axis=1)

S.head(2)
S.info()
a=S.iloc[0 : 28000] #from 0 to 27999

a.head(2)
a.tail(2) #verify the end rows of the table
a=a.astype(int)

a.head(3)
a=a.rename(columns={'label': "Label"})
a.info()
sorted_by_img = a.sort_values('ImageId')

sorted_by_img=sorted_by_lbl.astype(int)

sorted_by_img.head(3)
#Submit dataframe/table a containing:   

#ImageId,Label   

#1,0   

#2,0   

a.to_csv('Subms.csv',index=False)