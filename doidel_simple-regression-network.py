import os
#print(os.listdir("../lib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
from PIL import Image
import math

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Lambda, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K


sns.set(style='white', context='notebook', palette='deep')
def create_simple_image_set(phase):
    df = pd.read_csv('../input/sdobenchmark_full/' + phase + '/meta_data.csv', sep=",", parse_dates=["start", "end"], index_col="id")
    new_df = {'id': [], 'label': [], 'img': []}
    for row in df.iterrows():
        ar_nr, p = row[0].split("_", 1)
        img_path = os.path.join('../input/sdobenchmark_full/', phase, ar_nr, p)
        
        if not os.path.isdir(img_path):
            print(img_path + ' does not exist!')
            continue
        
        for img_name in os.listdir(img_path):
            if img_name.endswith('_magnetogram.jpg'):
                new_df['id'].append(row[0] + '-' + img_name.split('__')[0])
                new_df['label'].append(row[1]['peak_flux'])
                
                # load the image and preprocess it
                im = Image.open(os.path.join(img_path, img_name))
                im = im.crop((44, 44, 212, 212))
                im = im.resize((28,28), Image.ANTIALIAS)
                im = np.array(im) / 255.0
                im = im.reshape(28,28,1)
                new_df['img'].append(im)
    
    return pd.DataFrame(data=new_df)

train = create_simple_image_set('training')
test = create_simple_image_set('test')
print('Checking for NaNs')
print(train.isnull().any())
print(test.isnull().any())

Y_train = train["label"]

# Drop 'label' column
X_train = np.asanyarray(list(train['img']))

# free some space
#del train

# and do the same for validation data
# Here we could also split the training data into test and validation. But we'd have to make sure to split by Active Region numbers (top-level folder)
Y_val = test["label"]
X_val = np.asanyarray(list(test['img']))
#del test
g = plt.imshow(X_train[0][:,:,0])
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation='relu', kernel_initializer='ones'))

# Define a custom exponential layer
# It allows the network to map from a linear space (images) to an exponential space (emission 'peak_flux' output neuron).
# The clipping is there to avoid exploding gradients
model.add(Lambda(lambda val: 10. ** (K.clip(val, min_value=0., max_value=1.)*7.-9.)))

model.add(Dense(1, activation='linear'))

# Define the optimizer
optimizer = Adam()

# Compile the model
model.compile(optimizer = optimizer , loss = 'mean_absolute_error')

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=4,
                                            min_delta=1e-8,
                                            verbose=1, 
                                            factor=0.5,
                                            min_lr=0.00001)

batch_size = 128

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(X_train)
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = 100,
                              validation_data = (X_val,Y_val),
                              verbose = 2,
                              steps_per_epoch=X_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction])
# Plot the loss and accuracy curves for training and validation
plt.yscale('log')
plt.plot(history.history['loss'], color='b', label="Training loss")
plt.plot(history.history['val_loss'], color='r', label="validation loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
# Predict the values from the validation dataset
Y_pred = model.predict(X_val).reshape((3476,))
# evaluate
print(f'Mean absolute error:  {np.mean(np.abs(Y_val-Y_pred))}')
# copied from https://github.com/i4Ds/SDOBenchmark/blob/master/notebooks/utils/statistics.py

goes_classes = ['quiet','A','B','C','M','X']

def class_to_flux(c: str):
    'Inverse of flux_to_class \
    Maps a flare class (e.g. B6, M, X9) to a GOES flux value'
    if c == 'quiet':
        return 1e-9
    decade = goes_classes.index(c[0])-9
    sub = float(c[1:]) if len(c) > 1 else 1
    return round(10 ** decade * sub, 10)

def true_skill_statistic(y_true, y_pred, threshold='M'):
    'Calculates the True Skill Statistic (TSS) on binarized predictions\
    It is not sensitive to the balance of the samples\
    This statistic is often used in weather forecasts (including solar weather)\
    1 = perfect prediction, 0 = random prediction, -1 = worst prediction'
    separator = class_to_flux(threshold)
    y_true = [1 if yt >= separator else 0 for yt in y_true]
    y_pred = [1 if yp >= separator else 0 for yp in y_pred]
    print(f'Predicted {np.sum(np.array(y_pred))} M+, {len(y_pred)-np.sum(np.array(y_pred))} < M')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) - fp / (fp + tn)

print('TSS: ' + str(true_skill_statistic(Y_val, Y_pred)))
