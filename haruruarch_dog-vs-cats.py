# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install optuna

!pip install pillow
#get image filenames

cats = os.listdir("../input/cat-and-dog/training_set/training_set/cats")

dogs = os.listdir("../input/cat-and-dog/training_set/training_set/dogs")



from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from keras.utils import to_categorical

image_size = 112

num_classes = 2



#preparing dataset

labels = []

images = []





for cat in cats:

    if cat.startswith("c"):

        labels.append(0)

        cat_path = "../input/cat-and-dog/training_set/training_set/cats/" + cat

        image = load_img(cat_path, target_size = (image_size, image_size))

        image = img_to_array(image)

        images.append(image)

for dog in dogs:

    if dog.startswith("d"):

        labels.append(1)

        dog_path = "../input/cat-and-dog/training_set/training_set/dogs/" + dog

        image = load_img(dog_path, target_size = (image_size, image_size))

        image = img_to_array(image)

        images.append(image)

    

images = np.array(images)

labels = np.array(labels)

images = images.astype("float32") / 255.0

print("image shape: {}, label shape: {}".format(images.shape, labels.shape))

from keras.applications import VGG16,  MobileNetV2, InceptionResNetV2

from keras.layers import GlobalAveragePooling2D, Dense

from sklearn.model_selection import StratifiedKFold, train_test_split

from keras.optimizers import SGD, RMSprop, Adam

from keras.models import Model

from keras.callbacks import History



X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=0)



input_shape = (image_size, image_size, 3)

epochs = 30

batch_size = 8

fold_num = 5

seed = 0

log_scores = []



kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)



datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)



def create_model(model_name):

    assert model_name in ["vgg", "mobile", "inception"]

    if model_name == "vgg":

        model = VGG16(include_top = False, weights = "imagenet", input_shape = input_shape)

    elif model_name == "mobile":

        model = MobileNetV2(include_top = False, weights = "imagenet", input_shape = input_shape)

    elif model_name == "inception":

        model = InceptionResNetV2(include_top = False, weights = "imagenet", input_shape = input_shape)

    for layer in model.layers[:-5]:

        layer.trainable = False

    x = GlobalAveragePooling2D()(model.layers[-1].output)

    x = Dense(2, activation = "softmax")(x)

    return Model(inputs = model.inputs, outputs = x)



#train

def train(model, optimizer, learning_rate, trial):

    for train, valid in kfold.split(X_train, y_train):

        if optimizer == "sgd":

            model.compile(loss = 'categorical_crossentropy', optimizer = SGD())

        elif optimizer == "rmsprop":

            model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop())

        elif optimizer == "adam":

            model.compile(loss = 'categorical_crossentropy', optimizer = Adam())

        history = History()

        model.fit_generator(datagen.flow(X_train[train], to_categorical(y_train[train], num_classes), batch_size = batch_size),

                    steps_per_epoch = len(X_train[train]) / 32, epochs = epochs, callbacks = [history], verbose=1)

        scores = model.evaluate(X_train[valid], to_categorical(y_train[valid], num_classes), verbose=0)

        log_scores.append(scores)

        

        return history

        

        

#Hyperparameter optimization using optuna

def objective(trial):

    optimizer = trial.suggest_categorical("optimizer", ["sgd", "rmsprop", "adam"])

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e0)

    model_type = trial.suggest_categorical('model_type', ['vgg', 'mobile', 'inception'])

    model = create_model(model_type)

    hist = train(model, optimizer, learning_rate, trial)

    

    return np.min(hist.history["loss"])





#test
import optuna

study = optuna.create_study()

study.optimize(objective, n_trials = 5)