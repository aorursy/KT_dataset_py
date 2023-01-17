!pip install --upgrade pip
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications.xception import Xception 
import pandas as pd
import numpy as np
from tensorflow.keras.layers import concatenate
import tensorflow as tf
import csv
data_path='../input/2020-athens-eestech-challenge/'
traindf_name = data_path + 'train.csv'
traindf=pd.read_csv(traindf_name,dtype=str)
testdf_name = data_path + 'test.csv'
testdf=pd.read_csv(testdf_name,dtype=str)
from keras_preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.15)
train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory= data_path + "train/",
    x_col="file",
    y_col="category",
    subset="training",
    batch_size=32,
    shuffle=True,
    class_mode="categorical",
    target_size=(128,128))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=data_path + "train/",
    x_col="file",
    y_col="category",
    subset="validation",
    batch_size=32,
    shuffle=True,
    class_mode="categorical",
    target_size=(128,128))
model = Sequential()
model.add(Xception(include_top=False,input_shape=(128,128,3),weights="imagenet"))
model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.50))
model.add(Dense(31, activation="softmax"))
model.compile(optimizer = "adam", loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()
#Fitting keras model, no test gen for now
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=30
)
model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID
)
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=data_path + "test/",
    x_col="file",
    y_col=None,
    batch_size=32,
    shuffle=False,
    class_mode=None,
    target_size=(128,128))
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,

verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

#Fetch labels from train gen for testing
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions.insert(0, 'category')
my_indexes = list(range(1, len(predictions)+1))
my_indexes = [int(i) for i in my_indexes] 
my_indexes.insert(0, 'id')
results = zip(my_indexes, predictions)

# open a file for writing.
csv_out = open('submission.csv', 'w')

# create the csv writer object.
mywriter = csv.writer(csv_out)
# all rows at once.
mywriter.writerows(results)

# always make sure that you close the file.
# otherwise you might find that it is empty.
csv_out.close()