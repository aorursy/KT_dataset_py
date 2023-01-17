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
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,Flatten
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
num_classes=10

img_row=img_col=28
def data_prep(raw_data):

    raw=raw_data.to_numpy()

    y=raw[:,0]

    out_y=keras.utils.to_categorical(y,num_classes)

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_row, img_col, 1)

    out_x = out_x / 255

    return out_x, out_y
x,y=data_prep(train)
my_new_model=Sequential()

my_new_model.add(Conv2D(28,kernel_size=(3,3),activation='relu'))



my_new_model.add(Conv2D(20,kernel_size=(3,3),activation='relu'))

my_new_model.add(Conv2D(18,kernel_size=(3,3),activation='relu'))

my_new_model.add(Conv2D(15,kernel_size=(3,3),activation='relu'))



my_new_model.add(Flatten())

my_new_model.add(Dense(100,activation='relu'))

my_new_model.add(Dense(10,activation='softmax'))
my_new_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
my_new_model.fit(x,y,batch_size=100,epochs=40,validation_split=0.2)
def data_prep_test(raaw):

    rr=raaw.to_numpy()

    imgnn=rr.shape[0]

    out_x=rr.reshape(imgnn,img_row,img_col,1)

    out_x=out_x/255

    return out_x
pred=my_new_model.predict_classes(data_prep_test(test))
output=pd.DataFrame(range(1,28001),columns=['ImageId'])

output['Label']=pd.DataFrame(pred)
output.to_csv('submission.csv',index=False)
output