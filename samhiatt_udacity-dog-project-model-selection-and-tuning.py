from sklearn.datasets import load_files       

from keras.utils import np_utils

import numpy as np

from glob import glob



np.random.seed(37)



# define function to load train, test, and validation datasets

def load_dataset(path):

    data = load_files(path)

    dog_files = np.array(data['filenames'])

    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)

    return dog_files, dog_targets



# load train, test, and validation datasets

train_files, train_targets = load_dataset('../input/udacitydogproject/dogimages/dogImages/train')

valid_files, valid_targets = load_dataset('../input/udacitydogproject/dogimages/dogImages/valid')

test_files, test_targets = load_dataset('../input/udacitydogproject/dogimages/dogImages/test')



# load list of dog names

dog_names = [item[20:-1] for item in sorted(glob("../input/udacitydogproject/dogimages/dogImages/train/*/"))]



# print statistics about the dataset

print('There are %d total dog categories.' % len(dog_names))

print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))

print('There are %d training dog images.' % len(train_files))

print('There are %d validation dog images.' % len(valid_files))

print('There are %d test dog images.'% len(test_files))
from keras.preprocessing import image                  

from tqdm import tqdm



def path_to_tensor(img_path):

    # loads RGB image as PIL.Image.Image type

    img = image.load_img(img_path, target_size=(224, 224))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)

    x = image.img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor

    return np.expand_dims(x, axis=0)



def paths_to_tensor(img_paths):

    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]

    return np.vstack(list_of_tensors)
from PIL import ImageFile                            

ImageFile.LOAD_TRUNCATED_IMAGES = True                 



# pre-process the data for Keras

train_tensors = paths_to_tensor(train_files).astype('float32')/255

valid_tensors = paths_to_tensor(valid_files).astype('float32')/255

# test_tensors = paths_to_tensor(test_files).astype('float32')/255
train_tensors.shape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Dense

from keras.models import Sequential



from keras.wrappers.scikit_learn import KerasClassifier



def model_cnn_dropout(optimizer='rmsprop', filter_size=None,

                      filter_size_0=3, filter_size_1=3, filter_size_2=3, filter_size_3=3,

                      n_filters_0=64, n_filters_1=64, n_filters_2=64, n_filters_3=None,

                      dropout_rate_0=0, dropout_rate_1=0, dropout_rate_2=.25, dropout_rate_3=None,

                      pool_size=None,

                      pool_size_0=3, pool_size_1=3, pool_size_2=3, pool_size_3=None,

                     ):

    model = Sequential()

    model.add(Conv2D(n_filters_0,filter_size or filter_size_0,input_shape=(224, 224, 3),padding='valid',activation="relu"))

    model.add(MaxPooling2D(pool_size=pool_size or pool_size_0))

    model.add(Dropout(rate=dropout_rate_0))

    model.add(Conv2D(n_filters_1,filter_size or filter_size_1,padding='valid',activation="relu"))

    model.add(MaxPooling2D(pool_size=pool_size or pool_size_1))

    model.add(Dropout(rate=dropout_rate_1))

    model.add(Conv2D(n_filters_2,filter_size or filter_size_2,padding='valid',activation="relu"))

    model.add(MaxPooling2D(pool_size=pool_size or pool_size_2))

    model.add(Dropout(rate=dropout_rate_2))

    if n_filters_3:

        model.add(Conv2D(n_filters_3,filter_size or filter_size_3,padding='valid',activation="relu"))

        if pool_size_3:

            model.add(MaxPooling2D(pool_size=pool_size_3))

        if dropout_rate_3: 

            model.add(Dropout(rate=dropout_rate_3))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(133, activation="softmax"))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
# model = model_cnn_dropout(filter_size=3,n_filters_3=64,dropout_rate_3=.25)

# model.summary()
from sklearn.model_selection import GridSearchCV

from keras.callbacks import EarlyStopping, Callback



model = KerasClassifier(build_fn=model_cnn_dropout, epochs=50, batch_size=20, verbose=1)



hyperparams = {

    'n_filters_0':[32,64],

    'n_filters_3':[32,64],

    'filter_size':[3,4],

    'dropout_rate_3':[None,.25],

}

grid = GridSearchCV(estimator=model, param_grid=hyperparams, n_jobs=1, verbose=2, cv=3, return_train_score=True)

grid_result = grid.fit(train_tensors, train_targets)
grid_result.cv_results_
grid_result.best_params_
model = model_cnn_dropout(**grid_result.best_params_)

model.summary()
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint  



train_datagen = ImageDataGenerator(

        height_shift_range=20,  width_shift_range=20, 

        shear_range=20, 

        rotation_range=20,

        horizontal_flip=True,

    ).flow(

        train_tensors, y=train_targets, seed=37, batch_size=20,

    )



valid_datagen = ImageDataGenerator().flow(

        valid_tensors, y=valid_targets, batch_size=20,

    )

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', 

                               verbose=1, save_best_only=True)



learning_curve = model.fit_generator(

    train_datagen, 

    validation_data=valid_datagen,

    epochs=200, 

    steps_per_epoch=np.ceil(len(train_datagen.y)/train_datagen.batch_size),

    validation_steps=np.ceil(len(valid_datagen.y)/valid_datagen.batch_size),

    callbacks=[checkpointer], 

#     workers=4,

#     use_multiprocessing=True,

    verbose=1)
import matplotlib.pyplot as plt

def vis_learning_curve(learning):

    train_loss = learning.history['loss']

    train_acc = learning.history['acc']

    val_loss = learning.history['val_loss']

    val_acc = learning.history['val_acc']



    fig, axes = plt.subplots(1, 2, figsize=(20,4), subplot_kw={'xlabel':'epoch'} )

    axes[0].set_title("Accuracy")

    axes[0].plot(train_acc)

    axes[0].plot(val_acc)

    axes[0].legend(['training','validation'])

    axes[1].set_title("Loss")

    axes[1].plot(train_loss)

    axes[1].plot(val_loss)

    axes[1].legend(['training','validation'])



    best_training_epoc = val_loss.index(np.min(val_loss))

#     axes[0].axvline(x=best_training_epoc, color='red')

#     axes[1].axvline(x=best_training_epoc, color='red')

    acc=(best_training_epoc,val_acc[best_training_epoc])

    axes[0].annotate("val accuracy at best loss: %.1f%%"%(100*acc[1]),

                     xy=acc,

                     xytext=(len(train_loss)/2,.05),

                     arrowprops=dict(facecolor='black'))

    axes[1].annotate("val loss at epoch %i: %.1f"%(best_training_epoc,(val_loss[best_training_epoc])),

                     xy=(best_training_epoc,val_loss[best_training_epoc]),

                     xytext=(len(train_loss)/2,4.5),

                     arrowprops=dict(facecolor='black'))
vis_learning_curve(learning_curve)
test_tensors = paths_to_tensor(test_files).astype('float32')/255

dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]



test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)

print('Test accuracy: %.4f%%' % test_accuracy)