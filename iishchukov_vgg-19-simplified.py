%matplotlib inline

import os

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from keras import optimizers

from keras.models import Sequential

from keras.layers import (

    Dense,

    Dropout,

    Flatten,

    ZeroPadding2D,

    Conv2D,

    MaxPool2D,

    Activation,

)

from keras.preprocessing.image import ImageDataGenerator



print("import is ready")
# set consistent random seed

random_seed = 2018

np.random.seed(random_seed)  

tf.set_random_seed(random_seed)
print(os.listdir(".."))

print(os.listdir("."))
print(os.listdir("../input"))
# flow_from_dataframe

# https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

traindf=pd.read_csv("../input/train.csv",dtype=str)

# remove new whales from input

traindf = traindf[traindf.Id != "new_whale"]

# remove single whales values

traindf = traindf.groupby('Id').filter(lambda x: len(x) > 1)

# plot Id frequencies

traindf['Id'].value_counts()[1:16].plot(kind='bar')



testdf=pd.read_csv("../input/sample_submission.csv",dtype=str)



# datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

datagen = ImageDataGenerator(

        rotation_range=20,

        width_shift_range=0.1,

        height_shift_range=0.1,

        shear_range=0.5,

        zoom_range=(0.9, 1.1),

        horizontal_flip=False,

        vertical_flip=False,

        fill_mode='constant',

        cval=0,

        rescale=1./255.,

        validation_split=0.25    

)
traindf.shape

# Calculate number of unique classes (whales)

number_of_classes = traindf['Id'].nunique()
testdf.head(1)
# Pass the dataframes to 2 different flow_from_dataframe functions

# https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

train_generator=datagen.flow_from_dataframe(

dataframe=traindf,

directory="../input/train/",

x_col="Image",

y_col="Id",

subset="training",

batch_size=32,

seed=42,

shuffle=True,

class_mode="categorical",

target_size=(80,80))



valid_generator=datagen.flow_from_dataframe(

dataframe=traindf,

directory="../input/train/",

x_col="Image",

y_col="Id",

subset="validation",

batch_size=32,

seed=42,

shuffle=True,

class_mode="categorical",

target_size=(80,80))



test_datagen=ImageDataGenerator(rescale=1./255.)



test_generator=test_datagen.flow_from_dataframe(

dataframe=testdf,

directory="../input/test/",

x_col="Image",

y_col=None,

batch_size=32,

seed=42,

shuffle=False,

class_mode=None,

target_size=(80,80))

# Model @frommedium

# https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',

                 input_shape=(80,80,3)))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# VGG 19 start

# https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d

# part 2

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(128, (3, 3)))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(128, (3, 3)))

model.add(MaxPool2D(pool_size=(2, 2)))



# part1

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(256, (3, 3)))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(256, (3, 3)))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(256, (3, 3)))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(256, (3, 3)))

model.add(MaxPool2D(pool_size=(2, 2)))

# VGG 19 end



model.add(Flatten())

# model.add(Dense(512))

model.add(Dense(8192))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(number_of_classes, activation="softmax"))

model.compile(

    optimizers.rmsprop(lr=0.0001, decay=1e-6),

    loss="categorical_crossentropy",

    metrics=["accuracy"],

)



print("The model is ready")
# Fit the model @frommedium

# https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

# Class weights balancing

history = model.fit_generator(

    generator=train_generator,

    steps_per_epoch=STEP_SIZE_TRAIN,

    validation_data=valid_generator,

    validation_steps=STEP_SIZE_VALID,

    class_weight="auto",

    epochs=30,

)

# history plots

plt.plot(history.history['acc'])

plt.title('Model accuracy')

plt.ylabel('accuracy')

plt.xlabel('Epoch')

plt.show()



# Plot the loss curve for training

plt.plot(history.history['loss'], color='r', label="Train Loss")

plt.title("Train Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()

# Evaluate model

# https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

model.evaluate_generator(generator=valid_generator, steps=1)
print(os.listdir("."))
# Predict the output

# https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c



test_generator.reset()

pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST + 1, verbose=1)



predicted_class_indices = np.argmax(pred, axis=1)



labels = train_generator.class_indices

labels = dict((v, k) for k, v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



filenames = test_generator.filenames



print("filenames were prepared")
# multiple classes output

# https://www.kaggle.com/hexadd5/simple-resnet50-with-keras

kth = 5

classes = np.array([c for c, v in train_generator.class_indices.items()])



if True:

    classify_index = np.argpartition(-pred, kth)[:, :kth]

    classify_value = pred[np.arange(pred.shape[0])[:, None], classify_index]

    best_5_pred = np.zeros((len(classify_index), 5))

    best_5_class = np.zeros((len(classify_index), 5), dtype='int32')

    for i, p in enumerate(classify_value):

        sort_index = np.argsort(p)[::-1]

        best_5_pred[i] = (p[sort_index])

        best_5_class[i] = (classify_index[i][sort_index])

        

    # create output

    submit = pd.DataFrame(columns=['Image', 'Id'])

    for i, p in enumerate(best_5_pred):

        submit_classes = []

        if p[0] < 0.55:

            submit_classes.append('new_whale')

            submit_classes.extend(classes[best_5_class[i]][0:4])

        elif p[1] < 0.4 :

            submit_classes.extend(classes[best_5_class[i]][0:1])

            submit_classes.append('new_whale')

            submit_classes.extend(classes[best_5_class[i]][1:4])

        elif p[2] < 0.1 :

            submit_classes.extend(classes[best_5_class[i]][0:2])

            submit_classes.append('new_whale')

            submit_classes.extend(classes[best_5_class[i]][2:4])

        elif p[3] < 0.05 :

            submit_classes.extend(classes[best_5_class[i]][0:3])

            submit_classes.append('new_whale')

            submit_classes.extend(classes[best_5_class[i]][3:4])

        else:

            submit_classes.extend(classes[best_5_class[i]])

        classes_text = ' '.join(submit_classes)

        submit = submit.append(pd.Series(np.array([test_generator.filenames[i], classes_text]), index=submit.columns), ignore_index=True)

        # print(submit)

    submit.to_csv('submit.csv', index=False)

    print("submit results were written to the output")