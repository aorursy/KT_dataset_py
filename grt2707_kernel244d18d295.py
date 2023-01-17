# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as pd

import pandas as pd

from tensorflow import keras

from sklearn.model_selection import train_test_split



img_rows , img_cols = 28 , 28 

num_classes = 10 



def prep_data(raw):

    y = raw[: , 0 ]

    out_y = keras.utils.to_categorical(y , num_classes)

    x = raw[: , 1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images , img_rows , img_cols , 1)

    out_x = out_x / 255

    return out_x , out_y



mnsit_file = "../input/digit-recognizer/train.csv"

mnsit_data = np.loadtxt(mnsit_file ,skiprows = 1 , delimiter = ',')

x , y = prep_data(mnsit_data)





from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense , Flatten , Conv2D



mnsit_model = Sequential()

mnsit_model.add(Conv2D(12 , 

                      activation = 'relu' ,

                      kernel_size = 4 , 

                      input_shape = (img_rows ,img_cols , 1)))

mnsit_model.add(Conv2D(20 , activation = 'relu' , kernel_size = 4))

mnsit_model.add(Conv2D(20 , activation = 'relu' , kernel_size = 4))

mnsit_model.add(Flatten())

mnsit_model.add(Dense(100 , activation = 'relu'))

mnsit_model.add(Dense(10 , activation = 'softmax'))



mnsit_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam',

                      metrics=['accuracy'])

mnsit_model.fit(x , y , batch_size = 100 , epochs = 4 , validation_split = 0.2)





#---------------------------------------------------------------------------------------------------# 

#--------------Trial 1--------------#

#  Total size of Conv = 4

#  total dense  = 2 , ( 100 , 10)

#  conv1 = 20 , kernel size = 3

# Epoch 1/4 =  loss: 0.2212 - accuracy: 0.9345 - val_loss: 0.0669 - val_accuracy: 0.9792

# Epoch 2/4 = loss: 0.0580 - accuracy: 0.9825 - val_loss: 0.0589 - val_accuracy: 0.9826

# Epoch 3/4 =  loss: 0.0357 - accuracy: 0.9891 - val_loss: 0.0417 - val_accuracy: 0.9867

# Epoch 4/4 = loss: 0.0232 - accuracy: 0.9927 - val_loss: 0.0552 - val_accuracy: 0.9830

#---------------------------------------------------------------------------------------------------#

#----------------Trail 2------------#

#  Total size of Conv = 3

#  total dense  = 2 , ( 100 , 10)

#  conv1 = 20 , kernel size = 3

# Epoch 1/4 =  loss: 0.2260 - accuracy: 0.9328 - val_loss: 0.0731 - val_accuracy: 0.9768

# Epoch 2/4 = loss: 0.0603 - accuracy: 0.9808 - val_loss: 0.0593 - val_accuracy: 0.9802

# Epoch 3/4 =  loss: 0.0374 - accuracy: 0.9878 - val_loss: 0.0529 - val_accuracy: 0.9843

# Epoch 4/4 = loss: 0.0244 - accuracy: 0.9919 - val_loss: 0.0537 - val_accuracy: 0.9849

#----------------------------------------------------------------------------------------------------#

#  Total size of Conv = 3

#  total dense  = 2 , ( 100 , 10)

#  conv1 = 20 , kernel size = 4 ( shown improvement)

# Epoch 1/4 = loss: 0.2163 - accuracy: 0.9344 - val_loss: 0.0767 - val_accuracy: 0.9738

# Epoch 2/4 = loss: 0.0638 - accuracy: 0.9799 - val_loss: 0.0571 - val_accuracy: 0.9818

# Epoch 3/4 = loss: 0.0410 - accuracy: 0.9869 - val_loss: 0.0561 - val_accuracy: 0.9832

# Epoch 4/4 = loss: 0.0284 - accuracy: 0.9907 - val_loss: 0.0508 - val_accuracy: 0.9851

#-----------------------------------------------------------------------------------------------------#