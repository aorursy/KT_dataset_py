import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir('../input/dogs-vs-cats/'))
os.listdir("../input/dogs-vs-cats/")[0]
from zipfile import ZipFile
zf = ZipFile('../input/dogs-vs-cats/train.zip', 'r')
zf.extractall('../kaggle/working/Temp')
zf.close()
print(os.listdir("../kaggle/working/Temp/train"))
#Commented to reduce display...
print(os.listdir("../kaggle/working/Temp/train")[0].split('.'))
filenames = os.listdir("../kaggle/working/Temp/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append('dog')
        
    elif category == 'cat':
        categories.append('cat')

df = pd.DataFrame({
    'filename': filenames,
    'category':categories
})
df.shape
df.head()
df['category'].value_counts()
sns.countplot(x='category', data=df)
filenames[0]
from keras.preprocessing import image
img = image.load_img("../kaggle/working/Temp/train/"+filenames[0])
print(img)
plt.imshow(img)
test_image = image.load_img("../kaggle/working/Temp/train/"+filenames[0], target_size=(128, 128))
image.img_to_array(test_image).shape
plt.imshow(test_image)
from sklearn.model_selection import train_test_split

train_data, val_data  = train_test_split(df, test_size=0.2, random_state=42)
train_data.reset_index(drop=True, inplace=True)
val_data.reset_index(drop=True, inplace=True)
train_data.head()
val_data.head()
train_data['category'].value_counts()
sns.countplot(x='category', data=train_data)
val_data['category'].value_counts()
sns.countplot(x='category', data=val_data)
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, BatchNormalization
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
classifier = Sequential([Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                       input_shape=(128,128,3),padding='valid', activation='relu'),
                         
                         BatchNormalization(),
                         MaxPooling2D(pool_size=(2, 2)),
                         Dropout(0.2),
                         
                         Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                        padding='valid', activation='relu'),
                         BatchNormalization(),
                         MaxPooling2D(pool_size=(2, 2)),
                         Dropout(0.2),
                         
                         Flatten(),
                         Dense(512, activation='relu'),
                         BatchNormalization(),
                         Dropout(0.25),
                         Dense(2, activation='softmax')])

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.summary()
train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

train_generator = train_datagen.flow_from_dataframe(
                                train_data, "../kaggle/working/Temp/train/", 
                                x_col='filename', y_col='category', 
                                target_size=(128, 128),
                                batch_size=32, class_mode='categorical')
print(train_generator)
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_dataframe(
        val_data,
        "../kaggle/working/Temp/train/",
        x_col='filename',
        y_col='category',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
mc = ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history=classifier.fit_generator(train_generator,
                                steps_per_epoch=625,
                                epochs=50,
                                validation_data=val_generator,
                                validation_steps=200,
                                callbacks=[es, mc])
history = classifier.fit_generator(train_generator,
                                 steps_per_epoch=625, 
                                 epochs=50,
                                 validation_data=val_generator,
                                 validation_steps=200,
                                 callbacks=[es, mc])
history.history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], '')
plt.xlabel("Epochs")
plt.ylabel('Accuracy')
plt.title('Change of Accuracy over Epochs')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], '')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.title('Change of Loss over Epochs')
plt.legend(['loss', 'val_loss'])
plt.show()
train_generator.class_indices
from zipfile import ZipFile
zf = ZipFile('../input/dogs-vs-cats/test1.zip', 'r')
zf.extractall('../kaggle/working/Temp')
zf.close()
#Commented to reduce display...
#print(os.listdir("../kaggle/working/Temp/test1"))
filenames = os.listdir("../kaggle/working/Temp/test1")

test_data = pd.DataFrame({
    'filename': filenames
})
from keras.models import load_model

saved_model = load_model('best_model.h5')
img = image.load_img("../kaggle/working/Temp/test1/"+filenames[29])
                            
test_image = image.load_img("../kaggle/working/Temp/test1/"+filenames[29], 
                            target_size=(128, 128))
test_image = image.img_to_array(test_image)
plt.imshow(img)
test_image = np.expand_dims(test_image, axis=0)
result = saved_model.predict(test_image)
print(np.argmax(result, axis=1))
img = image.load_img("../kaggle/working/Temp/test1/"+filenames[39])
                            
test_image = image.load_img("../kaggle/working/Temp/test1/"+filenames[39], 
                            target_size=(128, 128))
test_image = image.img_to_array(test_image)
plt.imshow(img)
test_image = np.expand_dims(test_image, axis=0)
result = saved_model.predict(test_image)
print(np.argmax(result, axis=1))
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
        test_data,
        "../kaggle/working/Temp/test1/",
        x_col='filename',
        y_col=None,
        target_size=(128, 128),
        batch_size=32,
        class_mode=None)
predict = saved_model.predict_generator(test_generator)
final_prediction = np.argmax(predict, axis=1)
predict_df = pd.DataFrame(final_prediction, columns=['label'])
submission_df = test_data.copy()
submission_df['id'] = (submission_df['filename'].str.split('.').str[0]).astype(int)
submission_df = pd.concat([submission_df, predict_df], axis=1)
submission_df = submission_df.drop(['filename'], axis=1)
submission_df = submission_df.sort_values(by=['id'])
submission_df = submission_df.reset_index(drop=True)
submission_df.to_csv('submission.csv', index=False)