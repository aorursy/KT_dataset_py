################################################################################################

#                                      Used packages:                                          #

#             python 3.7.3, numpy 1.16.4, pandas 0.24.2, matplotlib 3.1.0, keras 2.2.4         #

################################################################################################
!pip install tensorflow==2.0.0

import numpy as np   # linear algebra

import pandas as pd  # data processing

import tensorflow as tf

# The subprocess module enables you to start new applications from your Python program.

# subprocess command to run command with arguments and return its output as a byte string.

from subprocess import check_output

tf.test.gpu_device_name()

tf.__version__
import matplotlib.pyplot as plt  # figures

%matplotlib inline
# Read data from files. Data belongs to <class 'pandas.core.frame.DataFrame'>

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv") # DataFrame shape (42000, 785), DataFrame type int64

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")   # DataFrame shape (28000, 784), DataFrame type int64



# return numpy representation of DataFrame object

test_data = test.to_numpy(dtype='float32') # does exactly the same as the line above but for the test dataset
train_data = train.iloc[:, 1:]                         # pixel data are indexed

train_labels = train['label'].values.astype('float32') # just a one-dimensional ndarray, with answers for 42000 samples.
#train_data
# original train_data shape is (42000, 784) We reshape it into s square form: (42000, 28*28)

# In order to reshape we must convert the DataFrame into Numpy array.

train_data_image = train_data.to_numpy(dtype='float32').reshape(train_data.shape[0], 28, 28)



# show num_image images from the train set starting from startin_image 

starting_image = 204

num_image = 5 



for idx,i in enumerate(range(starting_image,starting_image+num_image)):

    plt.subplot(1,num_image,idx+1)

    # show an image. cmap is a colormap = {'gray','hot', 'autumn', 'winter', 'bone'} 

    plt.imshow(train_data_image[i], cmap=plt.get_cmap('hot'))

    # show a label (answer) 

    plt.title(train_labels[i])
train_data = train_data/255  

test_data = test_data/255    
from tensorflow.keras.utils import to_categorical



train_labels = to_categorical(train_labels)
# import some machine learning libraries from keras and sklearn

from tensorflow import keras

from tensorflow.keras.layers import Dropout 



classifier = keras.Sequential()



classifier.add(keras.layers.Convolution2D(64,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))



classifier.add(keras.layers.MaxPooling2D(pool_size=(2,2)))



#classifier.add(Dropout(0.1))



classifier.add(keras.layers.Convolution2D(64,kernel_size=(3,3),activation='relu'))



classifier.add(keras.layers.MaxPooling2D(pool_size=(2,2)))



classifier.add(keras.layers.Flatten())



#classifier.add(Dropout(0.25))



classifier.add(keras.layers.Dense(256,activation = "relu"))

#classifier.add(Dropout(0.4))



classifier.add(keras.layers.Dense(10, activation = 'softmax'))



classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])



from tensorflow.keras.utils import plot_model



# save model architechture graph into file

plot_model(classifier, show_shapes=True,to_file='model.png')
from keras.losses import categorical_crossentropy, categorical_hinge

from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adamax



model.compile(optimizer=Adagrad(lr=0.005), loss=categorical_crossentropy, metrics=['categorical_accuracy'])
fit = classifier.fit(x=train_images_reshape, y=new_labels, validation_split=0.05, epochs=30, batch_size=50)#, shuffle=True)#

#classifier.fit(train_images_reshape,train_labels, epochs = 10)
train_data.shape

train_images_reshape=train_data.to_numpy().reshape(42000,28,28,1)
from sklearn import preprocessing
np.argmin(train_labels[1],axis=0)
train_labels[1]
import numpy as np
ml = np.linspace(0,9,10)
np.zeros(10)

labeling=np.array([np.zeros(10),ml]).transpose()

(train_labels[10]*np.array([np.zeros(10),ml]).transpose()).sum()
np.argmax(train_labels,axis=-1)
train_labels[-1].argmax(axis=-1).argmax()
new_labels=train_labels.argmax(axis=-1).argmax(axis=-1)
# list of computed metrics names

metrics = list(fit.history.keys())

metrics
loss_values = fit.history[metrics[2]]

val_loss_values = fit.history[metrics[0]]



epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo')

plt.plot(epochs, val_loss_values, 'b+')



plt.xlabel('Epochs')

plt.ylabel('Loss')
acc_values = fit.history[metrics[3]]

val_acc_values = fit.history[metrics[1]]



plt.plot(epochs, acc_values, 'bo')

plt.plot(epochs, val_acc_values, 'b+')



plt.xlabel('Epochs')

plt.ylabel('Accuracy')



plt.show()
print("Final validation loss function is", val_loss_values[-1])

print("Final validation accuracy is", val_acc_values[-1])
#score = model.evaluate(X_test, Y_test, verbose=0)
predictions = classifier.predict_classes(test_data.reshape(28000,28,28,1), batch_size=64, verbose=0)
submissions = pd.DataFrame({'ImageId':list(range(1,len(predictions) + 1)), "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)
from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)
create_download_link(filename='DR.csv')