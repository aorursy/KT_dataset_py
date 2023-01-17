import pandas as pd
from shutil import copy
import os
import pathlib
import PIL
import numpy as np
import sys
import matplotlib.pyplot as plt
%matplotlib inline
# os.environ['TF_KERAS'] = '1'
import keras
!ls ../input/

from keras.models import Model ,Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,Input,AveragePooling2D ,GlobalAveragePooling2D ,Dropout


data_dir = '../input/hackathon/train_data'
# df = pd.read_csv('dataset/train.csv')
# df.head()
def data_loader(dataset,df):

    all_images = os.listdir("dataset/train")

    co = 0

    for idx, row in df.iterrows():
        image_name = row["Image"]
        labels = row["target"]
        src_path = "./dataset/train/" + image_name
        des_path = os.path.join("train_data/" + str(labels))

        try:
            copy(src_path, des_path)
            print("Copied",co)
            co += 1
            

        except IOError as e:
            print("Unable to copy file {} to {}".format(src_path, des_path))

        except:
            print(
                "When try copy file {} to {}, unexpected error: {}".format(
                    src_path, des_path, sys.exc_info()
                )
            )
            
         

    


# if __name__ == "__main__":
#     train = pd.read_csv("dataset/train.csv")
#     test = pd.read_csv("./test.csv")
#     data_loader('train',train)
#     data_loader('test',test)
# data_dir = pathlib.Path(data_dir)
# total_train = len(list(data_dir.glob('./*/*jpg')))
mohiniyattam = list(data_dir.glob('./mohiniyattam/*'))

PIL.Image.open(str(mohiniyattam[1]))

# print(img.show())

batch_size = 32
img_height = 150
img_width = 150
epochs=40
data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                        width_shift_range=.15,
                                                        height_shift_range=.15,
                                                        horizontal_flip=True,
                                                        zoom_range=0.5,
                                                        validation_split=0.15)

train_data_gen = data_gen.flow_from_directory(batch_size=batch_size,
                                                           directory=data_dir,
                                                           target_size=(img_height, img_width),
                                                           subset='training',
                                                   )

valid_data_gen = data_gen.flow_from_directory(batch_size=batch_size,
                                                           directory=data_dir,
                                                           subset='validation',
                                                           target_size=(img_height, img_width)
                                                   )
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
for image,label in train_data_gen:
    plotImages(image)
    print(image.shape)
    print(label.shape)
    break
for image , label in train_data_gen:
    feature = base(image)
    print(feature.shape)
# #     globalAverage =  GlobalAveragePooling2D()(feature)
# #     print(globalAverage.shape)
    
    break
base = keras.applications.VGG16(input_shape=(img_height,img_width,3),
                                               include_top=False,
                                               weights='imagenet',
                               pooling='max')
base.summary()
#Freeze  VGG layers upto 17
base.trainable = True
for layer in base.layers[:17]:
    layer.trainable = False

model_t = Sequential([
    base,
#     globalAverage,
    Dense(units=1024,activation='relu',kernel_initializer='uniform'),
    Dropout(0.5),
#     Dense(units=512,activation='relu'),
#     Dropout(0.5),
    Dense(units=8,activation='softmax')
])
model_t.summary()
model_t.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2,  
                                            factor=0.5, 
                                            min_lr=0.00001)

history = model_t.fit(
    train_data_gen,
    steps_per_epoch=(train_data_gen.samples) // batch_size,
    epochs=epochs,
    validation_data=valid_data_gen,
    validation_steps=valid_data_gen.samples//batch_size
)
plt.figure(figsize=(15,7))
ax1 = plt.subplot(1,2,1)
ax1.plot(history.history['loss'], color='b', label='Training Loss') 
ax1.plot(history.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)
legend = ax1.legend(loc='best', shadow=True)
ax2 = plt.subplot(1,2,2)
ax2.plot(history.history['accuracy'], color='b', label='Training Accuracy') 
ax2.plot(history.history['val_accuracy'], color='r', label = 'Validation Accuracy')
legend = ax2.legend(loc='best', shadow=True)
val = []
for file in valid_data_gen.filenames:
    print(file)
val = []
for file in valid_data_gen.filenames:
    val.append(file)
    

pred = model_t.predict(valid_data_gen)
pred_clases = np.argmax(pred,axis=-1)
    
   


    
prediction = pd.DataFrame({'imagList':val , 'pred_class':pred_clases})

class_indices = {value : key for (key, value) in valid_data_gen.class_indices.items()}
class_indices
prediction['target'] = prediction['pred_class'].map(class_indices)
prediction