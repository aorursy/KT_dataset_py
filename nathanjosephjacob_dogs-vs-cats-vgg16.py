import numpy as np
import pandas as pd
import zipfile, os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip:
    zip.extractall()    
    zip.close()
sample_sub = pd.read_csv('/kaggle/input/dogs-vs-cats/sampleSubmission.csv')
print(sample_sub.head())

sample_img = load_img('/kaggle/working/train/cat.6562.jpg') # cute pic :)
plt.imshow(sample_img)
filenames = os.listdir('/kaggle/working/train')

labels = []
for filename in filenames:
    label = filename.split('.')[0] # splits on the first dot
    if label == 'cat':
        labels.append('0')
    else:
        labels.append('1')
        
df = pd.DataFrame({'id': filenames, 'label':labels })
print(df.shape)
df.head()
        
        
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(200,200,3))
# include_top refers to including the Dense layer on top of the network (1000 classes, in this case)

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# freezing the convolutional base so that its weights aren't updated:
#conv_base.trainable = False
# only the weights of the Dense layers will be updated

# we're gonna do some fine-tuning by training a part of the convolutional base
# it's basically freezing all the layers except the most abstract ones
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])

model.summary()
train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
train_df, validation_df = train_test_split(df, test_size=0.1)

train_size = train_df.shape[0]
validation_size = validation_df.shape[0]
batch_size = 20

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    '/kaggle/working/train/',
                                                    x_col='id',
                                                    y_col='label',
                                                    class_mode='binary',
                                                   target_size=(200,200),
                                                   batch_size=batch_size)

validation_generator = test_datagen.flow_from_dataframe(validation_df,
                                                       '/kaggle/working/train/',
                                                       x_col='id',
                                                       y_col='label',
                                                       class_mode='binary',
                                                       target_size=(200,200),
                                                       batch_size=batch_size)
history = model.fit_generator(train_generator,
                             steps_per_epoch=train_size//batch_size,
                             epochs=5,
                             validation_data=validation_generator,
                             validation_steps=validation_size//batch_size)

model.save('catsvsdogs_vgg16.h5')
plt.style.use('ggplot')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='training acc')
plt.plot(epochs, val_acc, 'r', label='validation acc')
plt.title('accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'b', label='training loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.title('loss')
plt.legend()

plt.show()

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as zip:
    zip.extractall()    
    zip.close()
    
filenames = os.listdir('/kaggle/working/test1')
test_df = pd.DataFrame({'id': filenames})
test_size = test_df.shape[0]

test_generator = test_datagen.flow_from_dataframe(test_df,
                                                 '/kaggle/working/test1/',
                                                 x_col='id',
                                                 y_col=None,
                                                 class_mode=None,
                                                 batch_size=batch_size,
                                                 target_size=(200,200))
prediction = model.predict_generator(test_generator, steps=test_size//batch_size)
threshold = 0.5
test_df['label'] = np.where(prediction > threshold, 1, 0) # if ...: 1; else: 0

submission = test_df.copy()
submission['id'] = submission['id'].str.split('.').str[0]
submission.to_csv('submission_mt.csv', index=False)
