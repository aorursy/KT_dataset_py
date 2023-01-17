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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os
wd = "/kaggle/input/jamp-hackathon-drive-1/train_set/"
images_dir=os.listdir(wd)
data=[]

for i in images_dir:

    path=os.path.join(wd,i)

    path_list=os.listdir(path)

    for j in path_list:

        data.append(os.path.join(path,j))
df=pd.DataFrame()

df['Images']=data

classes=df["Images"].str.split("/", n = 6, expand = True)[5]

df['Label']=classes

df = df.sample(frac=1).reset_index(drop=True)
df.head()
df['Label'].value_counts()
## train test split

from sklearn.model_selection import train_test_split



train, test = train_test_split(df, test_size=0.2,stratify=df['Label'])
sample = plt.imread(train['Images'].iloc[0])

plt.imshow(sample)
sample.shape
# importing required libraries



from keras.models import Sequential

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

import keras

from keras.layers import Dense



from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import decode_predictions

from tqdm import tqdm

import pickle
train_img=[]

for i in tqdm(df['Images']):



    temp_img=image.load_img(i,target_size=(224,224))



    temp_img=image.img_to_array(temp_img)



    train_img.append(temp_img)

    

#converting train images to array and applying mean subtraction processing



train_img=np.array(train_img) 

train_img=preprocess_input(train_img)

    
test_wd = "/kaggle/input/jamp-hackathon-drive-1/test_set/"

test_dir=os.listdir(test_wd)

test_data=[]

for i in test_dir:

    test_data.append(os.path.join(test_wd,i))

    

test_df=pd.DataFrame()

test_df['Images']=test_data

test_df.head()
test_img=[]

for i in tqdm(test_df['Images']):



    temp_img=image.load_img(i,target_size=(224,224))



    temp_img=image.img_to_array(temp_img)



    test_img.append(temp_img)

    

test_img=np.array(test_img) 

test_img=preprocess_input(test_img)
model = VGG16(weights='imagenet', include_top=False)
train_img.shape,test_img.shape
features_train=model.predict(train_img)
features_test=model.predict(test_img)
features_train.shape,features_test.shape
train_x=features_train.reshape(1027,-1)
test_x=features_test.reshape(256,-1)
# converting target variable to array



train_y=np.asarray(df['Label'])

# performing one-hot encoding for the target variable



train_y=pd.get_dummies(train_y)

train_y=np.array(train_y)

# creating training and validation set



from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid=train_test_split(train_x,train_y,test_size=0.3, random_state=42)

test_y=np.asarray(test['Label'])



test_y=pd.get_dummies(test_y)

test_y=np.array(test_y)
train_x.shape,train_y.shape,test_x.shape,test_y.shape
# creating a mlp model

from keras.layers import Dense, Activation

model=Sequential()



model.add(Dense(1000, input_dim=25088, activation='relu',kernel_initializer='uniform'))

keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)



model.add(Dense(500,input_dim=1000,activation='sigmoid'))

keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)



model.add(Dense(150,input_dim=500,activation='sigmoid'))

keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)



model.add(Dense(units=2))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])



model.fit(X_train, Y_train, epochs=20, batch_size=128,validation_data=(X_valid,Y_valid))
model.predict(X_valid)
model.evaluate(X_valid,Y_valid)
scores=model.evaluate(test_x,test_y)
print(f"Accuracy is {scores[1]*100} %")
output=np.argmax(model.predict(test_x),axis=1)
submission=pd.DataFrame()

submission['name']=test_df['Images'].str.split("/", n = 6, expand = True)[5].str.split(".", n = 3, expand = True)[0]

submission['class']=output
submission['class'] = submission['class'].replace({0:1, 1:0})
submission.head()
submission.to_csv("submission.csv", index=False)