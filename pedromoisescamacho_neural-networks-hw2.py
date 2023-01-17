%matplotlib inline

# feel free to add more imports here
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD  # use SGD in your model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
### YOUR CODE HERE ###
path_to_data = "../input/mlp_hw_data.csv"  # modify this
######################
X, y = make_classification(n_samples=200, n_features=10, 
                                     n_informative=5, 
                                     scale=2.0, 
                                     shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.5, random_state=42)
# put X_test and y_test in a "box" for later. We won't touch these until the very end.

# We also need to vectorize y_train for neural network testing (See other notebook for an explanation)
y_train_vectorized = to_categorical(y_train)
# we already convert y to categorical
y_train_vectorized = to_categorical(y_train)
### YOUR CODE HERE ###
# example:
# layer_sizes = [X_train.shape[1], 5, y_train_vectorized.shape[1]]  
# remember the first and last layers need to have the same dimensionality as your input and output
model1_layer_sizes = [X_train.shape[1], 6, y_train_vectorized.shape[1]]
model2_layer_sizes = [X_train.shape[1], 5, 5 ,10, 5, 3 , y_train_vectorized.shape[1]]
model3_layer_sizes = [X_train.shape[1], 5, 5 ,10, 5 , y_train_vectorized.shape[1]]
model4_layer_sizes = [X_train.shape[1], 5,3, y_train_vectorized.shape[1]]
model5_layer_sizes = [X_train.shape[1], 4, y_train_vectorized.shape[1]]
# feel free to add more models if you want to explore
######################
def build_model1():  # make sure you change the function name for each model!
    model1 = Sequential()

    ### YOUR CODE HERE ###
    # build your model. remember the input to the first layer needs to be layer_sizes[0]
    model1.add(Dense(input_dim=model1_layer_sizes[0],
                    units=model1_layer_sizes[1],
                    kernel_initializer="uniform",
                    activation="relu"))
    
    ######################
    # we write the last layer for you.
    # Finally, add a readout layer, mapping to output units using the softmax function
    model1.add(Dense(units=model1_layer_sizes[-1], # last layer
                    kernel_initializer='uniform',
                    activation="softmax"))
    
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent, leave these parameters fixed
    model1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])  
    # we'll have the categorical crossentropy as the loss function
    # we also want the model to automatically calculate accuracy
    return model1
def build_model2():  # make sure you change the function name for each model!
    model2 = Sequential()

    ### YOUR CODE HERE ###
    # build your model. remember the input to the first layer needs to be layer_sizes[0]
    model2.add(Dense(input_dim=model2_layer_sizes[0],
                    units=model2_layer_sizes[1],
                    kernel_initializer="uniform",
                    activation="relu"))
    model2.add(Dense(input_dim=model2_layer_sizes[1],
                    units=model2_layer_sizes[2],
                    kernel_initializer="uniform",
                    activation="relu"))
    
    model2.add(Dense(input_dim=model2_layer_sizes[2],
                    units=model2_layer_sizes[3],
                    kernel_initializer="uniform",
                    activation="relu"))
        
    model2.add(Dense(input_dim=model2_layer_sizes[3],
                    units=model2_layer_sizes[4],
                    kernel_initializer="uniform",
                    activation="relu"))
    
    model2.add(Dense(input_dim=model2_layer_sizes[4],
                    units=model2_layer_sizes[5],
                    kernel_initializer="uniform",
                    activation="relu"))
    
    ######################
    # we write the last layer for you.
    # Finally, add a readout layer, mapping to output units using the softmax function
    model2.add(Dense(units=model2_layer_sizes[-1], # last layer
                    kernel_initializer='uniform',
                    activation="softmax"))
    
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent, leave these parameters fixed
    model2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])  
    # we'll have the categorical crossentropy as the loss function
    # we also want the model to automatically calculate accuracy
    return model2
def build_model3():  # make sure you change the function name for each model!
    model3 = Sequential()

    ### YOUR CODE HERE ###
    # build your model. remember the input to the first layer needs to be layer_sizes[0]
    model3.add(Dense(input_dim=model3_layer_sizes[0],
                    units=model3_layer_sizes[1],
                    kernel_initializer="uniform",
                    activation="relu"))
    model3.add(Dense(input_dim=model3_layer_sizes[1],
                    units=model3_layer_sizes[2],
                    kernel_initializer="uniform",
                    activation="relu"))
    
    model3.add(Dense(input_dim=model3_layer_sizes[2],
                    units=model3_layer_sizes[3],
                    kernel_initializer="uniform",
                    activation="relu"))
        
    model3.add(Dense(input_dim=model3_layer_sizes[3],
                    units=model3_layer_sizes[4],
                    kernel_initializer="uniform",
                    activation="relu"))
    
    ######################
    # we write the last layer for you.
    # Finally, add a readout layer, mapping to output units using the softmax function
    model3.add(Dense(units=model3_layer_sizes[-1], # last layer
                    kernel_initializer='uniform',
                    activation="softmax"))
    
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent, leave these parameters fixed
    model3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])  
    # we'll have the categorical crossentropy as the loss function
    # we also want the model to automatically calculate accuracy
    return model3
def build_model4():  # make sure you change the function name for each model!
    model4 = Sequential()

    ### YOUR CODE HERE ###
    # build your model. remember the input to the first layer needs to be layer_sizes[0]
    model4.add(Dense(input_dim=model4_layer_sizes[0],
                    units=model4_layer_sizes[1],
                    kernel_initializer="uniform",
                    activation="relu"))
    model4.add(Dense(input_dim=model4_layer_sizes[1],
                    units=model4_layer_sizes[2],
                    kernel_initializer="uniform",
                    activation="relu"))
    
    ######################
    # we write the last layer for you.
    # Finally, add a readout layer, mapping to output units using the softmax function
    model4.add(Dense(units=model4_layer_sizes[-1], # last layer
                    kernel_initializer='uniform',
                    activation="softmax"))
    
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent, leave these parameters fixed
    model4.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])  
    # we'll have the categorical crossentropy as the loss function
    # we also want the model to automatically calculate accuracy
    return model4
def build_model5():  # make sure you change the function name for each model!
    model5 = Sequential()

    ### YOUR CODE HERE ###
    # build your model. remember the input to the first layer needs to be layer_sizes[0]
    model5.add(Dense(input_dim=model5_layer_sizes[0],
                    units=model5_layer_sizes[1],
                    kernel_initializer="uniform",
                    activation="relu"))
    
    ######################
    # we write the last layer for you.
    # Finally, add a readout layer, mapping to output units using the softmax function
    model5.add(Dense(units=model5_layer_sizes[-1], # last layer
                    kernel_initializer='uniform',
                    activation="softmax"))
    
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent, leave these parameters fixed
    model5.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])  
    # we'll have the categorical crossentropy as the loss function
    # we also want the model to automatically calculate accuracy
    return model5
# define kfold here:
k = 4
kf = kf = ms.KFold(k, shuffle=True)
# cross validation for model 1 (refer to keras-cv)
# we can pass any dataset into kf.split(), and it will return the indices of the "train" and "validation" partitions
accuracies = []

# STEP 1: partition the data chunks and iterate through them
# write the for loop here
for train_idx, val_idx in kf.split(X_train):
    
    # build your model below
    model1 = build_model1()
    
    # STEP 2: train the model on the k-1 chunks using the options above    
    model1.fit(X_train[train_idx], y_train_vectorized[train_idx], epochs=500, batch_size=50, verbose = 0)
    
    # STEP 3: predict the kth chunk and evaluate accuracy 
    # this is implemented for you
    proba = model1.predict_proba(X_train[val_idx], batch_size=32)  # predict the classes for the validation set
    classes = np.argmax(proba, axis=1)
    
    # save the accuracy (implemented for you)
    accuracies.append(accuracy_score(y_train[val_idx], classes))

# STEP 4: average across the k accuracies
model1_accuracy = np.array(accuracies).mean()  # the mean performance of model 1
print(model1_accuracy)
final_model =  build_model1() # use the build function for the model you selected

final_model.fit(X_train, y_train_vectorized, 
                epochs=1000, batch_size=50, verbose = 0)  
from keras.models import load_model

final_model.save("final_model.h5")  # by default, the save() method writes to a HDF5 file.

#del final_model
# to reload the model:
#model = load_model("model.h5")