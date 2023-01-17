import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
%matplotlib inline

import hashlib
from imageio import imread
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import os


from google.colab import drive
drive.mount('/content/drive')
# !unzip '/content/drive/My Drive/super-ai-image-classification.zip'

# make a seperate folder, which the duplicate will be removed later
from distutils.dir_util import copy_tree
fromDirectory = '/content/train/train/images'
toDirectory = "/content/image_no_duplicate"
copy_tree(fromDirectory, toDirectory)
img_folder = r'/content/image_no_duplicate'
original_cwd = os.getcwd()
original_cwd
# change the working directory to the new folder
os.chdir(img_folder)
os.getcwd()
file_list = os.listdir()
print(len(file_list))
# listing the index of the duplicate picture
duplicates = []
hash_keys = dict()
for index, filename in enumerate(os.listdir('.')):
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys:
            hash_keys[filehash] = index
        else:
            duplicates.append((index,hash_keys[filehash]))


# remove the duplicate picture accoring to the index
for index in duplicates:
    os.remove(file_list[index[0]])
file_list = os.listdir()
print(len(file_list))
# remove the deleted (duplicate) name from datafrmae
csv_file = pd.read_csv(r"/content/train/train/train.csv")
i = 0
while len(csv_file) != len(file_list):
    if csv_file.loc[i][0] not in os.listdir():
        csv_file = csv_file.drop(i)
    i+=1

# reaarange the dataframe, make sure that the dataframe's category is in order of 0,1,0,1,0,1,... 
# We do this to abuse the validate_split by Keras, since we cannot use stratified sampling directly
def zigzag_dataframe(df):
    adict = {}
    even = 0
    odd = 1
    for i in range(len(df)):
        if df.iloc[i,1] == 0:
            adict[even] = df.iloc[i]
            even+=2
        elif df.iloc[i,1] == 1:
            adict[odd] = df.iloc[i]
            odd+=2
        else:
            print(0)
    newdf = pd.DataFrame.from_dict(adict,orient = 'index')
    newdf = newdf.sort_index()
    return newdf
# change the category column to be string, to put in the data_generator
csv_file = zigzag_dataframe(csv_file)
csv_file['category'] = csv_file['category'].astype('str')
csv_file
# change the working directory back to the original one
os.chdir(original_cwd)
os.getcwd()
# use the pre-trained weights DenseNet169 architecture
import tensorflow.keras as K
input_t = K.Input(shape = (224,224,3))
import_model = K.applications.DenseNet169(include_top = False,
                                   weights = 'imagenet',
                                   input_tensor = input_t)

# freeze every CNN blocks except the last one
for layer in import_model.layers[:369]:
    layer.trainable = False
for i, layer in enumerate(import_model.layers):
    print(i, layer.name, '-', layer.trainable)

from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.3

# Initialize ImageDataGenerator for training set, data augmentation applied
datagen_train = ImageDataGenerator(rescale = 1./255.,
                                    validation_split=VALIDATION_SPLIT,
                                    rotation_range=30,
                                    width_shift_range=0.3,
                                    height_shift_range=0.3,
                                    shear_range=0.3,
                                    zoom_range=0.3,
                                    horizontal_flip=True)

# Initialize ImageDataGenerator for validation set
datagen_val = ImageDataGenerator(rescale=1./255, 
                                validation_split=VALIDATION_SPLIT) 

train_generator = datagen_train.flow_from_dataframe(dataframe = csv_file,
                                             directory = img_folder,
                                             x_col = 'id',
                                             y_col = 'category',
                                             batch_size = BATCH_SIZE,
                                             subset="training",
                                             class_mode = 'binary',
                                             target_size = (224,224),
                                             shuffle=True,
                                             seed=150)
val_generator = datagen_val.flow_from_dataframe(dataframe = csv_file,
                                             directory = img_folder,
                                             x_col = 'id',
                                             y_col = 'category',
                                             batch_size = BATCH_SIZE,
                                             subset="validation",
                                             class_mode = 'binary',
                                             target_size = (224,224),
                                             shuffle=True,
                                             seed=150)



# make sure that there is no training file in the validation set
for file in train_generator.filenames:
    if file in val_generator.filenames:
        print('FILE LEAKED!')
# Since the data is slightly unbalanced, we given differnet weight for each class
from sklearn.utils import class_weight

Y_train = np.array(train_generator.labels)
class_weight = class_weight.compute_class_weight('balanced'
                                               ,np.unique(Y_train)
                                               ,Y_train)
class_weight
class_weight = {0:class_weight[0],1:class_weight[1]}
class_weight
# Check the balance of training set
pd.Series(train_generator.labels).value_counts()
# Check the balance of validation set
pd.Series(val_generator.labels).value_counts()

# define f1 metric function


def recall_m(y_true, y_pred):
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    from keras import backend as K
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# build model
import tensorflow.keras as K
from keras.regularizers import l2

model = K.models.Sequential()
model.add(import_model)
model.add(K.layers.Flatten())
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(32,activation = 'relu',kernel_regularizer=l2(0.9)))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(32,activation = 'relu',kernel_regularizer=l2(0.9)))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(1,activation = 'sigmoid'))

# make a checkpoint for weight with best validation score
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
check_point = K.callbacks.ModelCheckpoint(filepath=filepath,
                                          monitor='val_accuracy',
                                          verbose=1, 
                                          save_best_only=True, 
                                          mode='max')

# compile the model
model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
              optimizer = tf.keras.optimizers.SGD(lr=0.001, 
                                                  decay=1e-6, 
                                                  momentum=0.9, 
                                                  nesterov=True),
                                                  metrics = ['accuracy',f1_m])
# train the model
history = model.fit_generator(train_generator,
                    epochs = 100,
                    verbose = 1,
                   validation_data=val_generator,
                    class_weight = class_weight,
                    callbacks = [check_point])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.summary()
model.save('model.h5')

# make a prediction 
test_folder = '/content/val/val'
test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_directory(directory = test_folder,
                                             target_size=(224, 224),
                                             shuffle = False,
                                             class_mode='categorical',
                                             batch_size=1)

# build the csv prediction file
filenames = test_generator.filenames
nb_samples = len(filenames)


predict = model.predict_generator(test_generator,steps = nb_samples)
predict = predict.flatten()
predict = np.where(predict> 0.5,1,0)

submission = pd.DataFrame({'id':filenames,'category':predict})
submission['id'] = submission['id'].apply(lambda x: x.split('/')[1]) # remove unneccessary suffix
submission.to_csv(r'submission.csv',index = False)
