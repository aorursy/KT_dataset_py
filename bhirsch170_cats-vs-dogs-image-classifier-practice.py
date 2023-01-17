import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
import matplotlib.pyplot as plt
from PIL import Image
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.optimizers import rmsprop

print('Imports done')
print(os.listdir("../input"))


train_dir = os.listdir("../input/train/train")

categories = []
for filename in train_dir:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)
        

LR=1e-4
earlystop = EarlyStopping(monitor='val_loss', patience=2)
callbacks=[earlystop]

rms=rmsprop(lr=LR, rho=0.9, epsilon=None, decay=0.0)
#Create test dataframe        
test_dir=os.listdir("../input/test1/test1/")
test_data=pd.DataFrame(test_dir,columns=['filename'])

#Create training dataframe
train_data = pd.DataFrame({
    'filename': train_dir,
    'category': categories
})

train_data.head()
sample = train_data['filename'][0]
image = load_img("../input/train/train/"+sample)
plt.imshow(image)


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.3)

train_generator = datagen.flow_from_dataframe(
    train_data, 
    "../input/train/train/", 
    x_col='filename',
    y_col='category',
    target_size=(128,128),
    class_mode='binary',
    batch_size=15
)

validation_generator=datagen.flow_from_dataframe(
    train_data, 
    "../input/train/train/", 
    x_col='filename',
    y_col='category',
    target_size=(128,128),
    class_mode='binary',
    subset='validation',
batch_size=15)
plt.figure(figsize=(10, 10))

x,y=validation_generator[0]

for i in range(0,3):
    plt.subplot(3,3,i+1)
    image = x[i]
    if y[i]==0:
        plt.title('It\'s a cat!')
    else:
        plt.title('It\'s a dog!')
    plt.imshow(image)
    
plt.show()
#Build CNN
model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(128,128,3), activation='relu'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
    
model.compile(loss='binary_crossentropy',
            optimizer=rms,
            metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_generator, 
    epochs=20,
    validation_data=validation_generator,
callbacks=callbacks)
fig, ax = plt.subplots(figsize=(12, 12))

ax.plot(history.history['acc'], color='b', label="Training accuracy")
ax.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax.set_xticks(np.arange(1, 10, 1))

legend = plt.legend(loc='best',shadow=False)
plt.show()
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_data, 
    '../input/test1/test1/', 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(128,128),
    batch_size=50
)
predictions=model.predict_generator(test_generator)
print('done!')
print(predictions)

threshold = 0.5
test_data['probability'] = predictions
test_data['category'] = np.where(test_data['probability'] > threshold, 1,0)


test_data.head()
plt.figure(figsize=(30, 30))

directory="../input/test1/test1/"

for i in range(0,9):
    plt.subplot(3,3,i+1)
    image = load_img(directory+test_data['filename'][i])
    if test_data['category'][i]==0:
        plt.title('I am sure this is a cat!')
    else:
        plt.title('I am sure this is a dog!')
    plt.imshow(image)
plt.show()


submission_df = test_data.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['probability']
submission_df.drop(['filename', 'probability'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)

print('done')