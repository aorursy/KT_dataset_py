from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras.utils import to_categorical

from keras.preprocessing.image import  ImageDataGenerator, img_to_array, image, load_img

from keras import backend as K

from keras.optimizers import Adam, SGD

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.models import load_model



import os

import numpy as np

import pandas as pd

import csv

from sklearn.model_selection import StratifiedKFold



from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 16



#Variable defining

SAMPLE_PER_CATEGORY = 200

SEED = 42

WIDTH = 64

HEIGHT = 64

DEPTH = 3

INPUT_SHAPE = (WIDTH, HEIGHT, DEPTH)



data_dir = '../input/plant-seedlings-classification/'

train_dir = os.path.join(data_dir, 'train')

test_dir = os.path.join(data_dir, 'test')

sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
!ls ../input/plant-seedlings-classification
CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

NUM_CATEGORIES = len(CATEGORIES)

NUM_CATEGORIES
for category in CATEGORIES:

    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))
def read_img(filepath, size):

    img = image.load_img(os.path.join(data_dir, filepath), target_size=size) ## https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img

    img = image.img_to_array(img)

    return img
train = []

for category_id, category in enumerate(CATEGORIES):

    for file in os.listdir(os.path.join(train_dir, category)):

        train.append(['train/{}/{}'.format(category, file), category_id, category])

train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])

train.shape
train.head(2)
train = pd.concat([train[train['category'] == c][:SAMPLE_PER_CATEGORY] for c in CATEGORIES])

train = train.sample(frac=1)

train.index = np.arange(len(train))

train.shape
train
test = []

for file in os.listdir(test_dir):

    test.append(['test/{}'.format(file), file])

test = pd.DataFrame(test, columns=['filepath', 'file'])

test.shape
test.head(2)
fig = plt.figure(1, figsize=(NUM_CATEGORIES, NUM_CATEGORIES))

grid = ImageGrid(fig, 111, nrows_ncols=(NUM_CATEGORIES, NUM_CATEGORIES), axes_pad=0.05)

i = 0

for category_id, category in enumerate(CATEGORIES):

    for filepath in train[train['category'] == category]['file'].values[:NUM_CATEGORIES]:

        ax = grid[i]

        img = read_img(filepath, (WIDTH, HEIGHT))

        ax.imshow(img / 255.)

        ax.axis('off')

        if i % NUM_CATEGORIES == NUM_CATEGORIES - 1:

            ax.text(250, 112, filepath.split('/')[1], verticalalignment='center')

        i += 1

plt.show();
np.random.seed(seed=SEED)
#create model from scratch

def createModel(number_of_hidden_layers, activation, optimizer, learning_rate, epochs):

    print("Create Model")



    model = Sequential()



    model.add(Conv2D(WIDTH, (3, 3), padding="same", input_shape=INPUT_SHAPE))

    model.add(Activation(activation))

    model.add(Conv2D(WIDTH, (3, 3)))

    model.add(Activation(activation))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(2*WIDTH, (3, 3), padding="same"))

    model.add(Activation(activation))

    model.add(Conv2D(2*WIDTH, (3, 3)))

    model.add(Activation(activation))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    

    for i in range(0,number_of_hidden_layers):

        model.add(Dense(512))

        model.add(Activation(activation))

        model.add(Dropout(0.5))



    model.add(Dense(12, activation='softmax'))



    if optimizer == 'SGD':

        opt = SGD(lr=learning_rate, decay=learning_rate / epochs)

    elif optimizer == 'Adam':

        opt = Adam(lr=learning_rate, decay=learning_rate / epochs)



    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model
def printHistory(history, title, epochs):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    t = f.suptitle(title, fontsize=12)

    f.subplots_adjust(top=0.85, wspace=0.3)



    epoch_list = list(range(1,epochs+1))

    ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')

    ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')

    ax1.set_xticks(np.arange(0, epochs+1, 5))

    ax1.set_ylabel('Accuracy Value')

    ax1.set_xlabel('Epoch')

    ax1.set_title('Accuracy')

    l1 = ax1.legend(loc="best")



    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')

    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')

    ax2.set_xticks(np.arange(0, epochs+1, 5))

    ax2.set_ylabel('Loss Value')

    ax2.set_xlabel('Epoch')

    ax2.set_title('Loss')

    l2 = ax2.legend(loc="best")
#callbacks for keras modal

def get_callbacks(patience):

    print("Get Callbacks")



    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=1e-5, patience=patience, verbose=1)

    #msave = ModelCheckpoint(filepath, save_best_only=True)

    return [lr_reduce, EarlyStopping()]
def evaluateModelDFViaCrossValidation(images, epochs, batch_size, learning_rate, cross_validation_folds, activation, number_of_hidden_layers, optimizer):

    print("Train Model")

     

    datagen_train = ImageDataGenerator(rescale=1./255)



    datagen_valid = ImageDataGenerator(rescale=1./255)

        

    print("Cross validation")

    kfold = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True)

    cvscores = []

    iteration = 1

    

    t = images.category_id

    

    for train_index, test_index in kfold.split(np.zeros(len(t)), t):



        print("======================================")

        print("Iteration = ", iteration)



        iteration = iteration + 1



        train = images.loc[train_index]

        test = images.loc[test_index]



        print("======================================")

        

        model = createModel(number_of_hidden_layers, activation, optimizer, learning_rate, epochs)



        print("======================================")

        

        train_generator = datagen_train.flow_from_dataframe(dataframe=train,

                                                  directory="/kaggle/input/plant-seedlings-classification/",

                                                  x_col="file",

                                                  y_col="category",

                                                  batch_size=batch_size,

                                                  seed=SEED,

                                                  shuffle=True,

                                                  class_mode="categorical",

                                                  target_size=(HEIGHT, WIDTH));

        valid_generator=datagen_valid.flow_from_dataframe(dataframe=test,

                                                  directory="/kaggle/input/plant-seedlings-classification/",

                                                  x_col="file",

                                                  y_col="category",

                                                  batch_size=batch_size,

                                                  seed=SEED,

                                                  shuffle=False,

                                                  class_mode="categorical",

                                                  target_size=(HEIGHT, WIDTH));

        

        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

        STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size



        #Trains the model on data generated batch-by-batch by a Python generator

        history = model.fit_generator(generator=train_generator,\

                            validation_data = valid_generator, \

                            steps_per_epoch=STEP_SIZE_TRAIN, \

                            validation_steps=STEP_SIZE_VALID, \

                            epochs=epochs, \

                            verbose=1)#, \

#                             callbacks = get_callbacks(patience=2))

        

        scores = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID, pickle_safe=True)

        print("Accuarcy %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        cvscores.append(scores[1] * 100)

        

        printHistory(history, "Basic CNN performance", epochs)



    accuracy = np.mean(cvscores);

    std = np.std(cvscores);

    print("Accuracy: %.2f%% (+/- %.2f%%)" % (accuracy, std))

    return accuracy, std
# Use different combinations to find the best params.

# Also change the CreateModel function to change the network architecture

evaluateModelDFViaCrossValidation(

    train,

    batch_size =32,

    cross_validation_folds = 5,

    learning_rate = 0.001,

    activation = 'relu',

    number_of_hidden_layers = 1,

    optimizer = 'Adam',

    epochs = 48

)
def trainFinalModel(images, epochs, batch_size, learning_rate, activation, number_of_hidden_layers, optimizer):

    print("Train Model")

     

    datagen_train = ImageDataGenerator(rescale=1./255)

    

    print("======================================")    

    model = createModel(number_of_hidden_layers, activation, optimizer, learning_rate, epochs)

    print("======================================")

    

    train_generator = datagen_train.flow_from_dataframe(dataframe=images,

                                                        directory="/kaggle/input/plant-seedlings-classification/",

                                                        x_col="file",

                                                        y_col="category",

                                                        batch_size=batch_size,

                                                        seed=SEED,

                                                        shuffle=True,

                                                        class_mode="categorical",

                                                        target_size=(HEIGHT, WIDTH));

        

    print (train_generator.class_indices)

    

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

    

    #Trains the model on data generated batch-by-batch by a Python generator

    model.fit_generator(generator=train_generator,\

                            steps_per_epoch=STEP_SIZE_TRAIN, \

                            epochs=epochs, \

                            verbose=1)#, \

#                             callbacks = get_callbacks(patience=2))

        

    model.save("/kaggle/working/best_model")

    

    return train_generator.class_indices
#predict values 

def predict_createSubmission(class_indices):

    print("Predicting......")

    

    datagen_test = ImageDataGenerator(rescale=1./255)

    

    test_generator = datagen_test.flow_from_dataframe(dataframe=test,

                                                        directory="/kaggle/input/plant-seedlings-classification/test/",

                                                        x_col="file",

                                                        y_col=None,

                                                        batch_size=1,

                                                        seed=SEED,

                                                        shuffle=False,

                                                        class_mode=None,

                                                        target_size=(HEIGHT, WIDTH));

        

    model = load_model('/kaggle/working/best_model')

    filenames = test_generator.filenames

    nb_samples = len(filenames)



    predictions = model.predict_generator(test_generator,steps = nb_samples) # return prob of each class per image (softmax)

    

    predicted_class_indices=np.argmax(predictions,axis=1)

       

    labels = dict((v,k) for k,v in class_indices.items())

    predicted_labels = [labels[k] for k in predicted_class_indices]

    

    results=pd.DataFrame({"file":filenames,

                          "species":predicted_labels})



    print (results)

    

    results.to_csv("submission.csv",index=False)



    print("Prediction Completed")
# Following model parameters were identified as best by evaluating various combinations above.

class_indices = trainFinalModel(

    train,

    batch_size =32,

    learning_rate = 0.001,

    activation = 'relu',

    number_of_hidden_layers = 1,

    optimizer = 'Adam',

    epochs = 48

)
predict_createSubmission(class_indices)