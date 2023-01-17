from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)



copyfile(src = "../input/process-and-utils/Utils_funX.py", dst = "../working/Utils_funX.py")

copyfile(src = "../input/cnn-model2/CNN_Model.py", dst = "../working/CNN_Model.py")

copyfile(src = "../input/data-prep/x_test.npz", dst = "../working/x_test.npz")

copyfile(src = "../input/data-prep/x_train.npz", dst = "../working/x_train.npz")

copyfile(src = "../input/data-prep/y_test.npz", dst = "../working/y_test.npz")

copyfile(src = "../input/data-prep/y_train.npz", dst = "../working/y_train.npz")
# Loading the x_train,y_train,x_test,and y_test values

from numpy import load

x_train = load('x_train.npz')['arr_0']

x_test = load('x_test.npz')['arr_0']

y_train = load('y_train.npz')['arr_0']

y_test = load('y_test.npz')['arr_0']
x_train.shape,y_train.shape,x_test.shape,y_test.shape
from keras import models

from sklearn.model_selection import train_test_split
x_train,x_test1,y_train,y_test1 = train_test_split(x_train,y_train,test_size=0.20,random_state = 42)
import gc
gc.collect()
x_train.shape,y_train.shape
x_test1.shape,y_test1.shape
x_test.shape,y_test.shape
gc.collect()
n_classes = 7
from CNN_Model import *
model= CNN_Model_Initialize(n_classes)
model.summary()
CNN_model_visualize(model)
gc.collect()
#First Phase of Training -- The compilation information in the 1st version of the model

model,history=CNN_model_Compile_and_Train(model,x_train,y_train,1,100,220)

model.save('CNN_Model_Final_v1.h5', include_optimizer=False)
gc.collect()
 #Second Phase of the Training Plot



def plot_training_loss_vs_validation_loss(history):

    # Plotting the Training loss v/s validation Loss

    loss_train = history.history['loss']

    loss_val = history.history['val_loss']

    plt.plot(loss_train,color='r',label="Training Loss")

    plt.plot(loss_train,color='b',label="Validation Loss")

    plt.title('Training Loss V/S Validation Loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.show()    
gc.collect()
# Plotting the Training loss v/s validation Loss

plot_training_loss_vs_validation_loss(history)
gc.collect()
def plot_training_accuracy_vs_validation_loss(history):

    # Plotting the Training loss v/s validation Loss

    loss_train = history.history['accuracy']

    loss_val = history.history['val_accuracy']

    plt.plot(loss_train,color='r',label="Training Accuracy")

    plt.plot(loss_train,color='b',label="Validation Accuracy")

    plt.title('Training Accuracy V/S Validation Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()
# Plotting the Training Accuracy v/s validation accuracy

plot_training_accuracy_vs_validation_loss(history)
gc.collect()
#First set of Test Samples(Not HAving test Results) 

x_test.shape,y_test.shape
y_test = model.predict(x_test,batch_size=100)
gc.collect()
#First set of Test Samples Without Predicted Value

import matplotlib.pyplot  as plt

from Utils_funX import *



for i in range(50):

    plt.title(Decode_Y_Val(y_test[i]))

    plt.imshow(x_test[i])

    plt.show()

    gc.collect()
#Second Samples With Test Samples(Having test Results) 

x_test1.shape,y_test1.shape
y_pred1 =  model.predict(x_test1,batch_size=100)
gc.collect()
for i in range(50):

    plt.title("TRUE : {}  PREDICTED : {}",Decode_Y_Val(y_test1[i]),Decode_Y_Val(y_pred1[i]))

    plt.imshow(x_test1[i])

    plt.show()

    gc.collect()
gc.collect()
results = model.evaluate(x_test1,y_test1,batch_size=100,verbose=0)



print("Test Loss And Test Accuracy :\n",results)