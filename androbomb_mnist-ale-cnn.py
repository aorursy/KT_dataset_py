###############################################################

# NB: shift + tab HOLD FOR 2 SECONDS!

###############################################################







# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import metrics

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print('\n Getting traing dataset...')

train_data = pd.read_csv('../input/digit-recognizer/train.csv')

print('Traing data set obtained \n')



print('Getting test dataset...')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')

print('Test data set obtained \n')
# The first function return the matrix out from the dataframe

def return_matrix(train, n=0):

    import numpy as np # linear algebra

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import matplotlib.pyplot as plt

    

    if ((n>=0) & (n<=train_data.shape[0])) :

        img_try = train.iloc[n].values.reshape(28,28)

    else :

        print('Insert a n between 0 and '.train_data.shape[0])

        pass 

    return img_try



# The second function eats dataframe plus an integer number between 0 and 21999

def print_matrix(train, test, n=0):

    import numpy as np # linear algebra

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import matplotlib.pyplot as plt

    

    if ((n>=0) & (n<=train_data.shape[0])) :

        mat = return_matrix(train, n)

    

        fig = plt.figure(figsize=(5, 5))

        plt.imshow(mat)

        plt.title(train_label[n], fontsize=30)

    else : 

        print('Insert a n between 0 and '.train_data.shape[0])

        pass

    

    return print('Done')

        

# The third function eats the training dataframe and spits  features and labels

def prepare_train_data(df):

    import pandas as pd 

    import numpy as np

    from keras import utils as ku

    

    print('Transforming the data...')

    train_img = df.drop('label', axis=1)

    train_img = train_img/255 #normalize [0,255] to [0.0 , 1.0]

    features = train_img.values.reshape(-1,28,28,1)

    

    labels = ku.to_categorical(df['label'] ,num_classes=10)

    print('Transformation done \n')

    

    return features, labels



# The fourth function eats the test dataframe and spits the features

def prepare_test_data(df):

    import pandas as pd 

    import numpy as np

    from keras import utils as ku

    

    print('Transforming the data...')

    train_img = df

    train_img = train_img/255 #normalize [0,255] to [0.0 , 1.0]

    features = train_img.values.reshape(-1,28,28,1)

    

    print('Transformation done \n')

    

    return features

features, labels = prepare_train_data(train_data)
#!pip install --upgrade keras



import sys

import keras

print('Keras version:',keras.__version__)



from keras import backend as K



from keras.preprocessing.image import array_to_img



print('\n Done. \n')
# Train test split

from sklearn.model_selection import train_test_split



print('Splitting the dataset... ')

X_train, X_test, Y_train, Y_test = train_test_split(np.array(features), np.array(labels), test_size=0.3)

print('Splitting done \n')
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

print(X_train.shape[0])

plt.figure(figsize=(15,8))

for i in range(60):

    plt.subplot(6,10,i+1)

    plt.imshow(X_train[i].reshape((28,28)),cmap='binary')

    plt.axis("off")

plt.show()
 # Creating batches for the CNN

from keras.preprocessing.image import ImageDataGenerator



classnames = '0 1 2 3 4 5 6 7 8 9'.split()

batch_size = 128



print("Getting Data...")

datagen = ImageDataGenerator(rescale=1./255, # normalize pixel values

                             validation_split=0.3) # hold back 30% of the images for validation

        # Generate batches of tensor image data with real-time data augmentation.

        # The data will be looped over (in batches).





print("Preparing training dataset...")

train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size)

print("Training dataset done. \n")



print("Preparing validation dataset...")

validation_generator = datagen.flow(X_test, Y_test, batch_size=batch_size)

print("Test dataset done. \n")



print('Done. \n')
# Define a CNN classifier network

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Flatten, Dense

from keras import optimizers



# Define the model as a sequence of layers

model = Sequential() # Sequential() : Linear stack of layers.



## Add some convolutional layers to extract features = Feature map

model.add(Conv2D(16, (3, 3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2, 2)))





# Now we'll flatten the feature maps 

model.add(Flatten())



# and generate an output layer with a predicted probability for each class

model.add(Dense(len(labels[0]), activation='softmax'))

    # The softmax activation function is used for multi-class classifiers



# We'll use the ADAM optimizer

opt = optimizers.Adam(lr=0.001)



# With the layers defined, we can now compile the model for categorical (multi-class) classification

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])



print(model.summary())
# Train the model over 5 epochs

num_epochs = 10

history = model.fit_generator(

    train_generator,

    steps_per_epoch = (X_train.shape[0] / batch_size) ,

    validation_data = validation_generator, 

    epochs = num_epochs)
import matplotlib.pyplot as plt



epoch_nums = range(1,num_epochs+1)

training_loss = history.history["loss"]

validation_loss = history.history["val_loss"]

train_acc = history.history["accuracy"]

val_acc = history.history['val_accuracy']



plt.figure(figsize=(13,5))



plt.subplot(1,2,1)

plt.plot(epoch_nums, training_loss)

plt.plot(epoch_nums, validation_loss)

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['training', 'validation'], loc='upper right')



plt.subplot(1,2,2)

plt.plot(epoch_nums, train_acc)

plt.plot(epoch_nums, val_acc)

plt.xlabel('epoch')

plt.ylabel('Accuracy')

plt.legend(['training', 'validation'], loc='lower right')



plt.show()
import numpy as np

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline





classes ='0 1 2 3 4 5 6 7 8 9'.split()

print("Generating predictions from validation data...")

# Get the image and label arrays for the first batch of validation data

x_test = validation_generator[0][0]

y_test = validation_generator[0][1]



# Use the moedl to predict the class

class_probabilities = model.predict(x_test)



# The model returns a probability value for each class

# The one with the highest probability is the predicted class

predictions = np.argmax(class_probabilities, axis=1)



# The actual labels are hot encoded (e.g. [0 1 0], so get the one with the value 1

true_labels = np.argmax(y_test, axis=1)





# Plot the confusion matrix

cm = confusion_matrix(true_labels, predictions)

tick_marks = np.arange(len(classes))





df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"], columns = [i for i in "0123456789"])

plt.figure(figsize = (7,7))

sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)

plt.xlabel("Predicted Class")

plt.ylabel("True Class")
test_features = prepare_test_data(test_data)



print('Starting predictions...')

final = model.predict(test_features)

final = np.argmax(final,axis = 1)

final_df = pd.Series(final, name="Label")

print('Predictions done. \n')
def show_pred_imag(test_data, final, n=0):

    import matplotlib.pyplot as plt

    import pandas as pd

    import numpy as np

    

    if ((n>=0) & (n<=test_data.shape[0])):

        mat = return_matrix(test_data, n)

        

        plt.imshow(mat)

        plt.title(final[n], fontsize=30)

        plt.show()

    else :

        print('Insert a number between 0 and '.test_data.shape[0])

        pass

    

    return print('Done. \n')
show_pred_imag(test_data, final, 99)
plt.figure(figsize=(10,8))

for i in range(60):

    plt.subplot(6,10,i+1)

    mat = return_matrix(test_data, i)

    plt.imshow(mat)

    plt.title(final[i], fontsize=15)

    plt.axis("off")



plt.subplots_adjust(hspace=1.5)

plt.subplots_adjust(wspace=0.3)

plt.show()