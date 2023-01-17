import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
# Technically not necessary in newest versions of jupyter
%matplotlib inline
pwd
my_data_dir = '../input/chest-xray-pneumonia/chest_xray/'


os.listdir(my_data_dir)
train_path = my_data_dir + 'train'
test_path =  my_data_dir + "test"
os.listdir(test_path)
os.listdir(test_path+"/PNEUMONIA")[0]
example = test_path + "/PNEUMONIA" + "/person100_bacteria_475.jpeg"
example_img = imread(example)
plt.imshow(example_img)
example_img.shape
# Other options: https://stackoverflow.com/questions/1507084/how-to-check-dimensions-of-all-images-in-a-directory-using-python
dim1 = []
dim2 = []
c = []
for image_filename in os.listdir(test_path+'/PNEUMONIA/'):
    
    img = imread(test_path+'/PNEUMONIA'+'/'+image_filename)
    d1,d2 = img.shape
    dim1.append(d1)
    dim2.append(d2)
    
np.mean(dim1)
np.mean(dim2)
image_shape = (128,128,3)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
batch_size = 16
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
train_image_gen.class_indices
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten
# model = Sequential()

# model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=image_shape,activation='relu'))
# model.add(MaxPool2D(4,4))

# model.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))
# model.add(MaxPool2D(4,4))


# model.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))
# model.add(MaxPool2D(4,4))


# model.add(Flatten())

# model.add(Dense(256,activation='relu'))

# model.add(Dropout(0.4))

# model.add(Dense(1,activation='sigmoid'))

# model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

# model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=5)
# results = model.fit_generator(train_image_gen,epochs=200,
#                               validation_data=test_image_gen,
#                              callbacks=[early_stop])
# model.evaluate(test_image_gen)
# from tensorflow.keras.models import load_model
# model.save('72%.h5')
# model = Sequential()

# model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=image_shape,activation='relu'))
# model.add(MaxPool2D(4,4))
# model.add(Dropout(0.2))


# model.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))
# model.add(MaxPool2D(4,4))



# model.add(Flatten())

# model.add(Dense(256,activation='relu'))

# model.add(Dropout(0.4))

# model.add(Dense(1,activation='sigmoid'))

# model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

# results = model.fit_generator(train_image_gen,epochs=200,
#                               validation_data=test_image_gen,
#                              callbacks=[early_stop])
# model.evaluate(test_image_gen)
# losses= pd.DataFrame(model.history.history)
# losses.plot()
# losses[['val_loss','loss']].plot()
# losses[['val_accuracy','accuracy']].plot()
# model.save("74%.h5")
# #good model 84% at 2 epoch
# model = Sequential()

# model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=image_shape,activation='relu'))
# model.add(MaxPool2D(4,4))



# model.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))
# model.add(MaxPool2D(4,4))



# model.add(Flatten())

# model.add(Dense(128,activation='relu'))

# model.add(Dropout(0.35))

# model.add(Dense(1,activation='sigmoid'))

# model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

# results = model.fit_generator(train_image_gen,epochs=200,
#                               validation_data=test_image_gen,
#                              callbacks=[early_stop])
# losses[['val_accuracy','accuracy']].plot()
# losses[['val_loss','loss']].plot()
# model.evaluate(test_image_gen)
# #good model 84% at 2 epoch
# model = Sequential()

# model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=image_shape,activation='relu'))
# model.add(MaxPool2D(4,4))



# model.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))
# model.add(MaxPool2D(4,4))



# model.add(Flatten())

# model.add(Dense(64,activation='relu'))

# model.add(Dropout(0.35))

# model.add(Dense(1,activation='sigmoid'))

# model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

# results = model.fit_generator(train_image_gen,epochs=200,
#                               validation_data=test_image_gen,
#                              callbacks=[early_stop])
# model.evaluate(test_image_gen)
# # model.save("good_model.h5")
# from tensorflow.keras import regularizers

# #good model 84% at 2 epoch
# model = Sequential()

# model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=image_shape,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#     bias_regularizer=regularizers.l2(1e-4),
#     activity_regularizer=regularizers.l2(1e-5)))
# model.add(MaxPool2D(4,4))



# model.add(Conv2D(filters=32,kernel_size=(4,4),activation='relu'))
# model.add(MaxPool2D(4,4))



# model.add(Flatten())

# model.add(Dense(128,activation='relu'))

# model.add(Dropout(0.35))

# model.add(Dense(1,activation='sigmoid'))

# model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

# results = model.fit_generator(train_image_gen,epochs=200,
#                               validation_data=test_image_gen,
#                              callbacks=[early_stop])
#  model.evaluate(test_image_gen)
#good model 84% at 2 epoch
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=image_shape,activation='relu'))
model.add(MaxPool2D(4,4))



model.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))
model.add(MaxPool2D(4,4))



model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.35))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

results = model.fit_generator(train_image_gen,epochs=2,
                              validation_data=test_image_gen,
                             callbacks=[early_stop])
model.evaluate(test_image_gen)
# model.save("best_model.h5")
losses=pd.DataFrame(model.history.history)
# losses[['val_loss','loss']].plot()
predictions = model.predict(test_image_gen)
from sklearn.metrics import confusion_matrix,classification_report
predictions = predictions > 0.5
confusion_matrix(test_image_gen.classes,predictions)
print(classification_report(test_image_gen.classes,predictions))
test_image_gen.class_indices
