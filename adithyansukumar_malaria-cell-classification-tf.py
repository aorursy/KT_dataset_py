import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
import os
my_data="../input/files1/Malaria Cells"
os.listdir(my_data)
train_path=my_data+"/training_set"
test_path=my_data+"/testing_set"
os.listdir(train_path)
os.listdir(test_path)
os.listdir(train_path+'/Parasitized')[0]
para_cell=train_path+"/Parasitized"+"/C59P20thinF_IMG_20150803_112802_cell_196.png"
para_image=imread(para_cell)
plt.imshow(para_image)
para_image.shape
os.listdir(train_path+'/Uninfected')[0]
normal_cell=train_path+"/Uninfected"+"/C130P91ThinF_IMG_20151004_142951_cell_89.png"
normal_img=imread(normal_cell)
plt.imshow(normal_img)
normal_img.shape
d1=[]
d2=[]
for image_filename in os.listdir(test_path+"/Uninfected"):
    img=imread(test_path+"/Uninfected"+"/"+image_filename)
    w,h,colors=img.shape
    d1.append(w)
    d2.append(h)
sns.jointplot(d1,d2)
np.mean(d1)
np.mean(d2)
image_shape=(131,131,3)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen=ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.10,
                             height_shift_range=0.10,
                             rescale=1/255,
                             shear_range=0.10,
                             zoom_range=0.10,
                             horizontal_flip=True,
                             fill_mode='nearest'
                              )
plt.imshow(image_gen.random_transform(para_image))
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dropout,Flatten,Dense,Conv2D,MaxPooling2D
model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='val_loss')
train_image_gen=image_gen.flow_from_directory(train_path,target_size=(131,131),color_mode='rgb',batch_size=16,class_mode='binary',shuffle=True)
test_image_gen=image_gen.flow_from_directory(test_path,target_size=(131,131),color_mode='rgb',batch_size=16,class_mode='binary',shuffle=True)
train_image_gen.class_indices
results=model.fit_generator(train_image_gen,epochs=20,validation_data=test_image_gen,callbacks=[early_stopping])
losses=pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.evaluate_generator(test_image_gen)
from tensorflow.keras.preprocessing import image
pred_probabilities = model.predict_generator(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
predictions
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(test_image_gen.classes,predictions))
len(os.listdir(train_path+'/Parasitized'))
len(os.listdir(train_path+'/Uninfected'))
my_image = image.load_img(normal_cell,target_size=image_shape)
my_image = image.img_to_array(my_image)
my_image = np.expand_dims(my_image, axis=0)
model.predict(my_image)
normal_img=imread(normal_cell)
plt.imshow(normal_img)
my_image = image.load_img(para_cell,target_size=image_shape)
my_image = image.img_to_array(my_image)
my_image = np.expand_dims(my_image, axis=0)
model.predict(my_image)
para_img=imread(para_cell)
plt.imshow(para_img)
new_img="../input/malaria-test/31-researchersm.jpg"

test= image.load_img(new_img,target_size=image_shape)
my_image = image.img_to_array(test)
my_image = np.expand_dims(my_image, axis=0)
model.predict(my_image)
plt.imshow(test_img)
model.save('malaria_classifier.h5')
import tensorflow as tf
new_model =  tf.keras.models.load_model('malaria_classifier.h5')
new_img2="../input/healthy-cell2/healthy_cell2.jpg"
test2=image.load_img(new_img2,target_size=image_shape)
new_image3 = image.img_to_array(test2)
new_img3= np.expand_dims(new_image3, axis=0)
new_model.predict(new_img3)
