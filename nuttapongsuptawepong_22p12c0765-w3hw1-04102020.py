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
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, layers, optimizers
image_size = 224
df_train_label = pd.read_csv ( "/kaggle/input/super-ai-image-classification/train/train/train.csv" )  
# สร้าง Function สำหรับแปลง list เป็น string

def list_to_string ( list_data ) :  
    str_data = " "
    return str_data.join ( map ( str, list_data ) )
# เตรียมข้อมูล x_train และ y_train โดยจะไม่นำรูปภาพที่ซ้ำกันมาใช้

x_train = [ ]
y_train = [ ]
list_imagestat = [ ]
list_duplicate = [ ]

image_rootpath = "/kaggle/input/super-ai-image-classification/train/train/images"
for image_filename in os.listdir ( image_rootpath ) :
    image_fullpath = os.path.join ( image_rootpath, image_filename )
    pil_image = PIL.Image.open ( image_fullpath )
    pil_image_stat = list_to_string ( PIL.ImageStat.Stat ( pil_image ).mean )
    if pil_image_stat not in list_imagestat :
        list_imagestat.append ( pil_image_stat )
        x_train.append ( [ np.asarray ( pil_image.convert ( "L" ).resize ( ( image_size, image_size ) ) ) ] )
        y_train.append ( [ df_train_label [ df_train_label [ "id" ] == image_filename ] [ "category" ].values [ 0 ] ] )
    else :
        list_duplicate.append ( pil_image_stat )
x_train = np.concatenate ( x_train, axis = 0 )
x_train.shape
y_train = np.concatenate ( y_train, axis = 0 )
y_train.shape
plt.imshow ( x_train [ 0 ], cmap = "gray" )
plt.show ( )
x_train = x_train.astype ( np.float32 ) / 255.0
plt.imshow ( x_train [ 0 ], cmap = "gray" )
plt.show ( )
list_imageduplicate = [ ]

image_rootpath = "/kaggle/input/super-ai-image-classification/train/train/images"
for image_filename in os.listdir ( image_rootpath ) :
    image_fullpath = os.path.join ( image_rootpath, image_filename )
    pil_image = PIL.Image.open ( image_fullpath )
    pil_image_stat = list_to_string ( PIL.ImageStat.Stat ( pil_image ).mean )
    if ( pil_image_stat == list_duplicate [ 0 ] ) :
        list_imageduplicate.append ( image_filename )
# แสดงตัวอย่างรูปภาพที่ซ้ำกัน

list_imageduplicate
print ( list_imageduplicate [ 0 ] )

temp_image = np.asarray ( PIL.Image.open ( image_rootpath + "/" + list_imageduplicate [ 0 ] ).convert ( "L" ).resize ( ( image_size, image_size ) ) )

plt.imshow ( temp_image, cmap = "gray" )
plt.show ( )
print ( list_imageduplicate [ 1 ] )

temp_image = np.asarray ( PIL.Image.open ( image_rootpath + "/" + list_imageduplicate [ 1 ] ).convert ( "L" ).resize ( ( image_size, image_size ) ) )

plt.imshow ( temp_image, cmap = "gray" )
plt.show ( )
print ( list_imageduplicate [ 2 ] )

temp_image = np.asarray ( PIL.Image.open ( image_rootpath + "/" + list_imageduplicate [ 2 ] ).convert ( "L" ).resize ( ( image_size, image_size ) ) )

plt.imshow ( temp_image, cmap = "gray" )
plt.show ( )
print ( list_imageduplicate [ 3 ] )

temp_image = np.asarray ( PIL.Image.open ( image_rootpath + "/" + list_imageduplicate [ 3 ] ).convert ( "L" ).resize ( ( image_size, image_size ) ) )

plt.imshow ( temp_image, cmap = "gray" )
plt.show ( )
x_test = [ ]
x_test_filename = [ ]

image_rootpath = "/kaggle/input/super-ai-image-classification/val/val/images"
for image_filename in os.listdir ( image_rootpath ) :
    image_fullpath = os.path.join ( image_rootpath, image_filename )
    x_test.append ( [ np.asarray ( PIL.Image.open ( image_fullpath ).convert ( "L" ).resize ( ( image_size, image_size ) ) ) ] )
    x_test_filename.append ( image_filename )
x_test = np.concatenate ( x_test, axis = 0 )
x_test.shape
plt.imshow ( x_test [ 0 ], cmap = "gray" )
plt.show ( )
x_test = x_test.astype ( np.float32 ) / 255.0
plt.imshow ( x_test [ 0 ], cmap = "gray" )
plt.show ( )
from tensorflow.keras.applications.densenet import DenseNet201
def trainmodel ( np_x, np_y ) :

    app = DenseNet201 ( include_top = False, weights = "imagenet" )

    x_in = layers.Input ( shape = ( image_size, image_size, 1 ) )
    x = layers.Conv2D ( 3, 1 ) ( x_in )
    x = app ( x )

    x = layers.Flatten ( ) ( x )
    x = layers.Dense ( 4096, activation = "relu" ) ( x )
    x = layers.Dense ( 4096, activation = "relu" ) ( x )
    x = layers.Dense ( 2, activation = "softmax" ) ( x )
            
    model = Model ( x_in, x )
    model.summary ( )
    
    x_train = np.delete ( np_x, np.s_ [ 0 : : 5 ], axis = 0 )
    x_test = np_x [ 0 : : 5 ]
    y_train = np.delete ( np_y, np.s_ [ 0 : : 5 ], axis = 0 )
    y_test = np_y [ 0 : : 5 ]
    
    optimizer_adam = optimizers.Adam ( learning_rate = 0.000001 )
    model.compile ( loss = "sparse_categorical_crossentropy", optimizer = optimizer_adam, metrics = [ "accuracy" ] )
    model.fit ( x_train, y_train, epochs = 50, batch_size = 16, verbose = 1, validation_data = ( x_test, y_test ) )
        
    return model
model = trainmodel ( x_train, y_train )
y_test_predict = model.predict ( x_test )
y_test_predict = y_test_predict.argmax ( axis = 1 )
y_test_predict
df = pd.DataFrame ( list ( zip ( x_test_filename, y_test_predict ) ), columns = [ "id", "category" ] ) 
df.to_csv ( "22p12c0765-DenseNet-Image224-Epocs50.csv", index = False )
model.save ( "22p12c0765-DenseNet-Image224-Epocs50.h5" )