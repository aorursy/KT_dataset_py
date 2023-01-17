# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt



from IPython.display import clear_output

from time import sleep

import os



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_dir = "../input/training/training.csv"

test_dir = "../input/test/test.csv"

look_dir = "../input/IdLookupTable.csv"

train_data = pd.read_csv(train_dir)

test_data = pd.read_csv(test_dir)

lookup_data = pd.read_csv(look_dir)

lookup_data.head()
train_data.head().T
train_data.isnull().any().value_counts()
train_data.fillna(method = "ffill",inplace = True)
train_data.isnull().any().value_counts()
imag = []

for i in range(7049):

    img = train_data["Image"][i].split(' ')

    img = ['0' if x == ' ' else x for x in img]

    imag.append(img)
#reshape it into float

image_list = np.array(imag, dtype = 'float')

X_train = image_list.reshape(-1,96,96)
plt.imshow(X_train[0],cmap = 'gray')

plt.show()
#now lets separate labels

training = train_data.drop('Image',axis = 1)

y_train = []

for i in range(7049):

    y = training.iloc[i,:]

    y_train.append(y)

y_train = np.array(y_train,dtype = 'float') 
from keras.models import Sequential

from keras.layers import Dense, Activation,Flatten,Dropout



model = Sequential([Flatten(input_shape=(96,96)),

                         Dense(128, activation="relu"),

                         Dropout(0.1),

                         Dense(64, activation="relu"),

                         Dense(30)

                         ])

model.compile(optimizer = 'adam',loss = 'mse',metrics=['mae','accuracy'])




model.fit(X_train,y_train,epochs = 500,batch_size = 128,validation_split = 0.2)



#now lets prepare out test data

timage  = []

for j in range(1783):

    imgi = test_data['Image'][j].split(' ')

    imgi = ['0' if x == '' else x for x in imgi]

    timage.append(imgi)

timage = np.array(timage,dtype = 'float')
X_test = timage.reshape(-1,96,96)

plt.imshow(X_test[0])

plt.show()
#lets Predict our result

y_pred = model.predict(X_test)
df = pd.DataFrame(y_pred)

df.columns = train_data.columns[0:30]

df = df.T

df.head()




sub = lookup_data



for i in range(sub.shape[0]):

    row = sub.loc[i,'FeatureName']

    col = sub.loc[i,'ImageId'] - 1

    sub.loc[i,'Location'] = df.loc[row, col]

sub = sub.drop(['ImageId', 'FeatureName'],axis=1)

sub.head()





sub.to_csv('facial_2.csv',index=False)