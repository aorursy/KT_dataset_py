# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
filenames = os.listdir("../input/train/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)
        
        
df = pd.DataFrame({'filename':filenames,
                  'category':categories})

df.head()
filenames = os.listdir("../input/train/train")
sample = random.choice(filenames)
image = load_img("../input/train/train/"+sample)
plt.imshow(image)
df['category'].value_counts().plot.bar()
train_df, validation_df = train_test_split(df,test_size = 0.20, random_state = 42)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validation_df.shape[0]
batch_size=15
from keras.models import Sequential,model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
image_width = 50
image_height = 50
image_size = (image_width, image_height)
image_channel = 3 # RGB color
classifier1 = Sequential()
classifier1.add(Conv2D(32,(3,3),input_shape = (image_width,image_height,image_channel),activation = 'relu'))
classifier1.add(BatchNormalization())
classifier1.add(MaxPooling2D(pool_size = (2,2)))
                
classifier1.add(Conv2D(64,(3,3),activation = 'relu'))
classifier1.add(BatchNormalization())
classifier1.add(MaxPooling2D(pool_size = (2,2)))
                
classifier1.add(Flatten())
classifier1.add(Dense(256,activation = 'relu'))
classifier1.add(Dense(units = 1, activation = 'sigmoid'))
classifier1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier1.summary()
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)


train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../input/train/train/", 
    x_col='filename',
    y_col='category',
    target_size=image_size,
    class_mode='binary',
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validation_df, 
    "../input/train/train/", 
    x_col='filename',
    y_col='category',
    target_size=image_size,
    class_mode='binary',
    batch_size=batch_size
)
x , y  = train_generator.next()
for i in range(0,1):
    random_image = x[i]
    plt.imshow(random_image)
    plt.show()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]
history = classifier1.fit_generator(
    train_generator, 
    epochs= 25,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)
model_json = classifier1.to_json()
!mkdir Saved_models
with open("Saved_models/cnn_base_model.json","w") as json_file:
    json_file.write(model_json)
    
classifier1.save_weights("Saved_models/cnn_base_model.h5")
print("Saved model to disk")
json_file = open('Saved_models/cnn_base_model.json', 'r')

loaded_classifier_json = json_file.read()

json_file.close()

loaded_classifier = model_from_json(loaded_classifier_json)

loaded_classifier.load_weights("Saved_models/cnn_base_model.h5")
print("Loaded model from disk")

loaded_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, 25, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, 25, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


test_image = load_img('../input/test1/test1/1.jpg', target_size = (50, 50))
plt.imshow(test_image)
plt.show()
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'This is a dog'
else:
    prediction = 'This is a cat'

print (prediction)
test_filenames = os.listdir("../input/test1/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../input/test1/test1/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=image_size,
    batch_size=batch_size,
    shuffle=False
)
predict = loaded_classifier.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
threshold = 0.5
test_df['probability'] = predict
test_df['category'] = np.where(test_df['probability'] > threshold, 1,0)
test_df['category'].value_counts().plot.bar()
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    probability = row['probability']
    img = load_img("../input/test1/test1/"+filename, target_size=image_size)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' '(' + "{}".format(round(probability, 2)) + ')')
plt.tight_layout()
plt.show()

