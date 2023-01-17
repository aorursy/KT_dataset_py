

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        ... #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# rastgele bir resme erişim

import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def show_image(img_path):
    img = mpimg.imread(img_path)
    plt.figure()
    plt.imshow(img)
    
show_image("/kaggle/input/landmark-places-of-turkey/images_test/images_test/Japon Bahfesi//1_japon-bahcesi-09.jpg")

# test 
len(os.listdir("/kaggle/input/landmark-places-of-turkey/images_test/images_test/"))
# train

os.listdir("/kaggle/input/landmark-places-of-turkey/images_train/images_train")



# bastan sona
os.listdir("/kaggle/input/")
os.listdir("/kaggle/input/landmark-places-of-turkey")
os.listdir("/kaggle/input/landmark-places-of-turkey/images_test/images_test/")
# test dosya sayısı

len(os.listdir("/kaggle/input/landmark-places-of-turkey/images_test/images_test/"))
# traing dosya isimleri 
os.listdir("/kaggle/input/landmark-places-of-turkey/images_train/images_train/")
# train
len(os.listdir("/kaggle/input/landmark-places-of-turkey/images_train/images_train/"))
# train
os.listdir("/kaggle/input/landmark-places-of-turkey/images_train/images_train/kordon/")


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# Toplu isim Değisikliği

import os
#files = os.listdir(path)


#for index, file in enumerate(files):
    #os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpg'])))
    

path = "/kaggle/input/landmark-places-of-turkey"
# print(type(path))
# str donuyor

for i in range(len(os.listdir(path))):
    #print(i) # imagetest + image_train
    if not os.listdir(path)[i].endswith("zip"):
        #print(os.listdir(path)[i])
        
        # 2 kere ıc içe
        path2 = path +"/" + os.listdir(path)[i] + "/" + os.listdir(path)[i]
        #print(path2)
        # train ve test içinde ayrı ayrı gez
        
        
        for i in range(len(os.listdir(path2))):
            if i == 1:
                break
            #print(os.listdir(path2)[i])
            
            path3 = path2 +"/"+ os.listdir(path2)[i]
            
            
            print(path3)
            
            # isim degısıklıgı ıcın sadece
            #print(path3.split("/")[-1]) # "buyuk_yenı_han"
            #print(path3.split("/")[-2][7:]) #  "test" ve train kısmı ıcın
            
            for i in range(len(os.listdir(path3))):
                print(os.listdir(path3)[i])


import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
import random

# test 

plt.figure(figsize=(20,20))
test_folder=r'/kaggle/input/landmark-places-of-turkey/images_test/images_test/B_y_k Yeni Han'
for i in range(5):
    file = random.choice(os.listdir(test_folder))
    image_path= os.path.join(test_folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.imshow(img)
# train 

plt.figure(figsize=(20,20))
test_folder=r'/kaggle/input/landmark-places-of-turkey/images_train/images_train/B_y_k Yeni Han'
for i in range(5):
    file = random.choice(os.listdir(test_folder))
    image_path= os.path.join(test_folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.imshow(img)
IMG_WIDTH=200
IMG_HEIGHT=200
img_folder=r'/kaggle/input/landmark-places-of-turkey/images_train/images_train'
def create_dataset(img_folder):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        print(dir1)
        for file in os.listdir(os.path.join(img_folder, dir1)):
            print(file)
            image_path= os.path.join(img_folder, dir1,  file)
            print(image_path)
 
            try:
                image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            
                #print(image)
                image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            
            except Exception as e:
                print(str(e))
    
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            del image
            # extract the image array and class name
            class_name.append(dir1)
    return img_data_array, class_name



img_data, class_name = create_dataset(r'/kaggle/input/landmark-places-of-turkey/images_train/images_train')
# test


def create_dataset(img_folder):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        print(dir1)
        for file in os.listdir(os.path.join(img_folder, dir1)):
            print(file)
            image_path= os.path.join(img_folder, dir1,  file)
            print(image_path)
 
            try:
                image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            
                #print(image)
                image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            
            except Exception as e:
                print(str(e))
    
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            del image
                # extract the image array and class name
            class_name.append(dir1)
    return img_data_array, class_name



img_data_test, class_name_test = create_dataset(r'/kaggle/input/landmark-places-of-turkey/images_test/images_test')

# train 
target_dict={k: v for v, k in enumerate(np.unique(class_name))}
target_dict
# train
target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
target_val

# test
target_dict_test ={k: v for v, k in enumerate(np.unique(class_name))}
target_dict_test
# test
target_val_test =  [target_dict[class_name[i]] for i in range(len(class_name))]
target_val_test

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

hist = model.fit(
 img_data, 
 epochs=100,
 verbose=1,
 validation_data= img_data_test,
).history
plt.plot(model.history['accuracy'], label='accuracy')
plt.plot(model.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


print(test_acc)

