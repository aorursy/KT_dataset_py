# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/intel-image-classification/seg_train/seg_train/"))

mountain_paths=os.listdir("../input/intel-image-classification/seg_train/seg_train/mountain")

street_paths=os.listdir("../input/intel-image-classification/seg_train/seg_train/street")

glacier_paths=os.listdir("../input/intel-image-classification/seg_train/seg_train/glacier")

buildings_paths=os.listdir("../input/intel-image-classification/seg_train/seg_train/buildings")

sea_paths=os.listdir("../input/intel-image-classification/seg_train/seg_train/sea")

forest_paths=os.listdir("../input/intel-image-classification/seg_train/seg_train/forest")

# Any results you write to the current directory are saved as output.
import numpy as np

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



image_size = 150



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = preprocess_input(img_array)

    return(output)


buildings_paths=[os.path.join("../input/intel-image-classification/seg_train/seg_train/buildings",filename_) for filename_ in buildings_paths ]

mountain_paths=[os.path.join("../input/intel-image-classification/seg_train/seg_train/mountain",filename_1) for filename_1 in mountain_paths ]

forest_paths=[os.path.join("../input/intel-image-classification/seg_train/seg_train/forest",filename_0) for filename_0 in forest_paths ]

sea_paths=[os.path.join("../input/intel-image-classification/seg_train/seg_train/sea",filename_2) for filename_2 in sea_paths ]

glacier_paths=[os.path.join("../input/intel-image-classification/seg_train/seg_train/glacier",filename_3) for filename_3 in glacier_paths ]

street_paths=[os.path.join("../input/intel-image-classification/seg_train/seg_train/street",filename_4) for filename_4 in street_paths ]
len(forest_paths)
import keras

y_buildings = keras.utils.to_categorical([0 for i in range(2100)], 6)

y_forest = keras.utils.to_categorical([1 for i in range(2100)], 6)

y_glacier = keras.utils.to_categorical([2 for i in range(2100)], 6)

y_mountain = keras.utils.to_categorical([3 for i in range(2100)],6 )

y_sea = keras.utils.to_categorical([4 for i in range(2100)], 6)

y_street = keras.utils.to_categorical([5 for i in range(2100)],6)


y_buildings[0]

y_forest[0]
y_glacier[0]
y_mountain[0]
y_sea[0]
y_street[0]
X=[]

Y=np.array([[1.0,0.,0.,0.,0.,0.]])

c=500



for i in range(500):

    X.append(buildings_paths[i])

    if(i!=0):Y=np.append(Y,[y_buildings[i]],axis=0)

for i in range(500):

    X.append(forest_paths[i])

    Y=np.append(Y,[y_forest[i]],axis=0)

for i in range(500):

    X.append(glacier_paths[i])

    Y=np.append(Y,[y_glacier[i]],axis=0)

for i in range(500):

    X.append(mountain_paths[i])

    Y=np.append(Y,[y_mountain[i]],axis=0)

for i in range(500):

    X.append(sea_paths[i])

    Y=np.append(Y,[y_sea[i]],axis=0)

for i in range(500):

    X.append(street_paths[i])

    Y=np.append(Y,[y_street[i]],axis=0) 



for i in range(500,1000):

    X.append(buildings_paths[i])

    if(i!=0):Y=np.append(Y,[y_buildings[i]],axis=0)

for i in range(500,1000):

    X.append(forest_paths[i])

    Y=np.append(Y,[y_forest[i]],axis=0)

for i in range(500,1000):

    X.append(glacier_paths[i])

    Y=np.append(Y,[y_glacier[i]],axis=0)

for i in range(500,1000):

    X.append(mountain_paths[i])

    Y=np.append(Y,[y_mountain[i]],axis=0)

for i in range(500,1000):

    X.append(sea_paths[i])

    Y=np.append(Y,[y_sea[i]],axis=0)

for i in range(500,1000):

    X.append(street_paths[i])

    Y=np.append(Y,[y_street[i]],axis=0) 

    

    

for i in range(1000,1500):

    X.append(buildings_paths[i])

    if(i!=0):Y=np.append(Y,[y_buildings[i]],axis=0)

for i in range(1000,1500):

    X.append(forest_paths[i])

    Y=np.append(Y,[y_forest[i]],axis=0)

for i in range(1000,1500):

    X.append(glacier_paths[i])

    Y=np.append(Y,[y_glacier[i]],axis=0)

for i in range(1000,1500):

    X.append(mountain_paths[i])

    Y=np.append(Y,[y_mountain[i]],axis=0)

for i in range(1000,1500):

    X.append(sea_paths[i])

    Y=np.append(Y,[y_sea[i]],axis=0)

for i in range(1000,1500):

    X.append(street_paths[i])

    Y=np.append(Y,[y_street[i]],axis=0)   
Y

X=read_and_prep_images(X, 150, 150)
X.shape
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPooling2D



model = Sequential()

model.add(Conv2D(200, kernel_size=(2,2),activation='relu' ,input_shape = (150, 150, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(200, kernel_size=(2,2),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(150, kernel_size=(3, 3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(100, kernel_size=(3, 3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(250, kernel_size=(3, 3),activation='relu'))

model.add(Conv2D(200, kernel_size=(3, 3),activation='relu'))

model.add(Conv2D(250, kernel_size=(3, 3),activation='relu'))

#model.add(Conv2D(200, kernel_size=(3, 3),activation='relu'))

#model.add(Conv2D(250,kernel_size= (3,3),activation='relu'))

#model.add(Conv2D(200, kernel_size=(3, 3),activation='relu'))

#model.add(Conv2D(250, kernel_size=(3,3),activation='relu'))

#model.add(Conv2D(100, kernel_size=(3, 3),activation='relu'))

model.add(Flatten())

model.add(Dense(100,activation='relu'))

model.add(Dense(50,activation='relu'))

model.add(Dense(6, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


model.fit(X,Y,batch_size=10,epochs=40,validation_split=0.3)







#model_street.fit(output_street,y_street,batch_size=10,epochs=1,validation_split=0.2)
#model_sea.fit(output_sea,y_sea,batch_size=10,epochs=1,validation_split=0.2)
#model_glacier.fit(output_glacier,y_glacier,batch_size=10,epochs=1,validation_split=0.2)
#model_buildings.fit(output_buildings,y_buildings,batch_size=10,epochs=1,validation_split=0.2)
mountain_test=os.listdir('../input/intel-image-classification/seg_test/seg_test/mountain')

mountain_paths=[os.path.join("../input/intel-image-classification/seg_test/seg_test/mountain",filename) for filename in mountain_test ]

test_mountain=read_and_prep_images(mountain_paths, 150, 150)
forest_test=os.listdir('../input/intel-image-classification/seg_test/seg_test/forest')

forest_paths=[os.path.join("../input/intel-image-classification/seg_test/seg_test/forest",filename) for filename in forest_test ]

forest_mountain=read_and_prep_images(forest_paths, 150, 150)
model.predict(forest_mountain)
model.predict(test_mountain)