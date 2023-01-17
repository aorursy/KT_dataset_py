# Input      data     files     are    available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Import Pandas for data manipulation using dataframes

import pandas as pd

#Import Numpy for statistical calculations

import numpy as np

# Import Warnings 

import warnings

warnings.filterwarnings('ignore')

# Import matplotlib Library for data visualisation

import matplotlib.pyplot as plt

#Import train_test_split from scikit library

from sklearn.model_selection import train_test_split

# Import Keras

import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard

num_classes = 10

epochs = 5

#Loading test and train data
train_df = pd.read_csv('/kaggle/input/fashion-mnist_train.csv',sep=',')
test_df = pd.read_csv('/kaggle/input/fashion-mnist_test.csv', sep = ',')
train_df.head()
test_df.head()

train_data = np.array(train_df, dtype = 'float32')

#Similarly let us do the same process for test data

test_data = np.array(test_df, dtype='float32')

train_df.describe()
x_train = train_data[:,1:]/255

y_train = train_data[:,0]

x_test= test_data[:,1:]/255

y_test=test_data[:,0]

x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)
#Now let us visualise the sample image how it looks like in 28 * 28 pixel size

image = x_train[55,:].reshape((28,28))
plt.imshow(image)
plt.show()
# Defined the shape of the image as 3d with rows and columns and 1 for the 3d visualisation

image_rows = 28

image_cols = 28

batch_size = 512

image_shape = (image_rows,image_cols,1) 
#formating on the x_train,x_test and x_validate sets.

x_train = x_train.reshape(x_train.shape[0],*image_shape)
x_test = x_test.reshape(x_test.shape[0],*image_shape)
x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)
cnn_model = Sequential([
    Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),
    MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
    Dropout(0.2),
    Flatten(), # flatten out the layers
    Dense(32,activation='relu'),
    Dense(10,activation = 'softmax')
    
])
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
history = cnn_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=50,
    verbose=1,
    validation_data=(x_validate,y_validate),
)
score = cnn_model.evaluate(x_test,y_test,verbose=0)
print('Test Loss : {:.4f}'.format(score[0]))
print('Test Accuracy : {:.4f}'.format(score[1]))
import matplotlib.pyplot as plt
%matplotlib inline
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#get the predictions for the test data
predicted_classes = cnn_model.predict_classes(x_test)
#get the indices to be plotted
y_true = test_df.iloc[:, 0]
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_true[correct]))
    plt.tight_layout()


for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_true[incorrect]))
    plt.tight_layout()


# predict results
 
results = cnn_model.predict(x_test)

# select the indix with the maximum probability
 
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,10001),name = "id"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)



