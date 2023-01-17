import pandas as pd

import numpy as np



import time



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





from sklearn.model_selection import KFold,StratifiedKFold
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization

from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D

from tensorflow.keras.optimizers import Adadelta

from keras.utils.np_utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau,LearningRateScheduler,EarlyStopping



import tensorflow as tf

from tensorflow import keras



print(tf.version.VERSION)
# Reading the Train and Test Datasets.



train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# Let's see the shape of the train and test data

print(train.shape, test.shape)
train.head()
train.describe()
fig=plt.figure(figsize=(14,8))

columns = 8

rows = 3

for i in range(1, rows*columns+1):

    

    digit_array = train.loc[i-1, "pixel0":]

    arr = np.array(digit_array)   

    image_array = np.reshape(arr, (28,28))   

    

    

    fig.add_subplot(rows, columns, i)

    plt.title("Label:"+train.loc[i-1,"label"].astype("str"))

    plt.imshow(image_array, cmap=plt.cm.binary)

    

plt.show()
ax=sns.countplot(train.loc[:,"label"])
# dividing the data into the input and output features to train make the model learn based on what to take in and what to throw out.

train_X = train.loc[:, "pixel0":"pixel783"]

train_y = train.loc[:, "label"]



train_X = train_X / 255.0

test_X = test / 255.0



train_X = train_X.values.reshape(-1,28,28,1)

test_X = test_X.values.reshape(-1,28,28,1)

train_y = to_categorical(train_y, num_classes = 10)


def build_model(input_shape=(28, 28, 1)):

    model = Sequential()

    model.add(Conv2D(64, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(128, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(256, kernel_size = 4, activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))



    

    return model
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10, # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
#learning_rate_reduction

learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy',   # quality to be monitored 

                                            patience=3,          # no of epoch with no improvement after learning rate will be reduced

                                            verbose=1,           # update message

                                            factor=0.8,          # reducing learning rate 

                                            min_lr=0.001)       # lower bound learning rate 



# DECREASE LEARNING RATE EACH EPOCH

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)





early_stop=EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto',

    baseline=None, restore_best_weights=True)
%%time



nets=10



model = [0] *nets

history = [0] * nets





skf = StratifiedKFold(n_splits=nets, shuffle = True, random_state=1)

skf.get_n_splits(train_X, train['label'])

print(skf)



number=0





for train_index, test_index in skf.split(train_X, train['label']):

    print("SPLIT ",number," TRAIN index:", train_index, "TEST index:", test_index)

    

    X_train, X_val = train_X[train_index], train_X[test_index]

    y_train, y_val = train_y[train_index], train_y[test_index]

    

    model[number]=build_model()

    model[number].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    

    history[number] =model[number].fit(datagen.flow(X_train,y_train), epochs=100 ,validation_data = (X_val,y_val) ,

     batch_size=100, verbose = 0,callbacks = [annealer,early_stop])

    

    metrics=pd.DataFrame(history[number].history)

    display(metrics)

    

    

    number+=1
for number in range(0,nets):

    model[number].save("StratifiedKFold_10_batch100_double_val_loss_"+str(number)+".h5")
# ENSEMBLE PREDICTIONS AND SUBMIT

results = np.zeros( (test_X.shape[0],10) ) 

for j in range(nets):

    results = results + model[j].predict(test_X)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("StratifiedKFold_10_batch100_double_val_loss.csv",index=False)
def show_test_digits(indexes):    

    columns = 10

    rows = len(indexes)//columns +1    

    fig=plt.figure(figsize=(14,rows*2))    

    

    for plot_id, i in enumerate(indexes,1):     

        fig.add_subplot(rows, columns, plot_id)                     

        plt.title("predict:"+submission.loc[i,"Label"].astype("str"))

        plt.axis("off")

        plt.imshow(np.reshape(test_X[i], (28,28)), cmap=plt.cm.binary)

           

    plt.show()



show_test_digits(range(500,530))