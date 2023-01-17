# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras

from keras.datasets import cifar10

from keras.utils import to_categorical



(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(y_test[2405])

image = X_test[2405]



import matplotlib.pyplot as plt

import matplotlib as mpl

np_image = np.array(image, dtype='uint8')

plt.imshow(np_image)

plt.axis("off")

plt.show() 
batch_size = 128

num_classes = 10



# input image dimensions

img_rows, img_cols = 32, 32



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')
y_train_cat = to_categorical(y_train, 10)

y_test_cat = to_categorical(y_test, 10)

print(y_train_cat)
#Getting values between 0 and 1 for each RGB value, effectively 'scaling' it (NOT using a scaler though!)

X_train_scaled = (X_train / 255.0).astype('float32') #.astype('float32') used again here, otherwise python will round some outputs (like 0.99999) to 1.

X_test_scaled = (X_test.astype('float32') / 255.0).astype('float32')

print("Correctly scaled?:", X_train_scaled[0][0][0][0] == (X_train[0][0][0][0]/255.0).astype('float32'))

print("Number of (1st entry) and Shape (2nd and 3rd entry) of examples:", X_train_scaled.shape)
from functools import partial

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

input_shape = [X_train_scaled.shape[1], X_train_scaled.shape[2], X_train_scaled.shape[3]]

print("Shape of the examples is:",input_shape)



def createModel(custom_lr=0.01): #Default of 0.01, if the user doesnt otherwise give a value (to avoid runtime errors).

    suggestedDefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")

    nn_model = keras.models.Sequential()

    nn_model.add(Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))

    nn_model.add(Conv2D(64, (3, 3), activation='relu'))

    nn_model.add(MaxPooling2D(pool_size=(2, 2)))

    nn_model.add(Dropout(0.25))

    nn_model.add(Flatten())

    nn_model.add(Dense(128, activation='relu'))

    nn_model.add(Dropout(0.5))

    nn_model.add(Dense(10, activation='softmax'))



  

    #nn_model = keras.models.Sequential([

    #    suggestedDefaultConv2D(filters=64, kernel_size=7, input_shape=shape),

#

#        keras.layers.MaxPooling2D(pool_size=2),

#        suggestedDefaultConv2D(filters=128),

#        suggestedDefaultConv2D(filters=128),

#

#        keras.layers.MaxPooling2D(pool_size=2),

#        suggestedDefaultConv2D(filters=256),

#        suggestedDefaultConv2D(filters=256),

#

#        keras.layers.MaxPooling2D(pool_size=2),

#        keras.layers.Flatten(),

#        keras.layers.Dense(units=128, activation='relu'),

#        keras.layers.Dropout(0.5),

#        keras.layers.Dense(units=64, activation='relu'),

#        keras.layers.Dropout(0.5),

#        keras.layers.Dense(units=10, activation='softmax'),

#    ])    

    custom_sgd = keras.optimizers.SGD(learning_rate=custom_lr)    

    nn_model.compile(loss="categorical_crossentropy", optimizer=custom_sgd,  metrics=["accuracy"])

    return nn_model

def gridSearchLR(minLR=0.01, maxLR=0.1, increment=10):

    currLR = minLR

    best_val_acc = 0  #Default value, any model should have an accuracy better than 0 (otherwise it is useless!)

    iter_val = maxLR / increment

    print("Estimated runtime: ~"+str(round(increment*(20*3) / 60,2))+" minutes, [aka "+str(increment*(20*3))+" seconds].\n") #20 epochs, which take approx 10 seconds each (hence multiplication symbols).

    while(currLR <= maxLR):

        cnn_nn_model = createModel(custom_lr=currLR)

        fit_cnn_nn = cnn_nn_model.fit(X_train_scaled, y_train_cat, epochs=20, validation_split=0.15)

        curr_val_acc = fit_cnn_nn.history['val_accuracy'][-1]

        if curr_val_acc > best_val_acc:

            best_val_acc = curr_val_acc

            best_cnn_nn_model = cnn_nn_model

            best_lr = currLR

            best_history = fit_cnn_nn

            

        currLR += iter_val

    return best_cnn_nn_model, best_val_acc, best_lr, best_history
best_cnn_model, best_val_acc, best_lr, best_history = gridSearchLR()

#best_cnn_model, best_val_acc, best_lr, best_history = gridSearchLR(minLR=0.07, maxLR=0.07) #BEST RESULTS FROM TESTING.I.e. from running above commented-out grid-search result may differ on the comiled version (saved to Kaggle).
print("Best validation accuracy of:",best_val_acc)

print("When learning rate is at:",best_lr)

print("Accuracy [from metric]:",best_history.history['accuracy'][-1])

best_cnn_model.summary() 
test_results = best_cnn_model.evaluate(X_test_scaled, y_test_cat)

print("categorical_crossentropy loss info for the test dataset (predictions): ",test_results[0])
from sklearn.metrics import accuracy_score

y_pred = np.argmax(best_cnn_model.predict(X_test_scaled), axis=1)

y_pred_proba = best_cnn_model.predict_proba(X_test_scaled)



print("TEST SET ACCURACY SCORE WAS:", accuracy_score(y_test, y_pred))
print(y_pred[1])

print(y_pred_proba[1][8])
worstScoreList = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]



for categoryClass in range(0, 10):

    max_diff = 0 # the maximum p_predicted - p_correct

    correct_data = None # the images has been lassified



    worst_pred = None # the prediected label results in max error

    worst_proba = None # the predicted probability results in max error



    for data,label,pred,proba in zip(X_test, y_test, y_pred, y_pred_proba):

        label = label[0]

        

        if label == categoryClass:



            if label != pred: # find out misclassification

                correct_p = proba[label] # proba is a list of probabilities corresponding to each class    

                

                for idp,p in enumerate(proba):

                    score = p - correct_p

                    if score > max_diff and idp != label:

                        max_diff = score

                        worst_pred = idp

                        worst_proba = p

                        worstScoreList[categoryClass][0] = max_diff

                        worstScoreList[categoryClass][1] = worst_pred

                        worstScoreList[categoryClass][2] = data

                        worstScoreList[categoryClass][3] = worst_proba                        

                        worstScoreList[categoryClass][4] = correct_p



                    # find the biggest error based on p - best_p
print(worstScoreList)
actualCategory_label = 0

for score, misclassifedCategory, data, pred_p, actual_p in worstScoreList:    

    image = data #X_test[index]

    np_image = np.array(image, dtype='uint8')

    #pixels = np_image.reshape((32, 32, 3))

    plt.imshow(np_image)

    plt.axis("off")

    plt.title(str(actualCategory_label)+" was misclassified as having a label of: "+str(misclassifedCategory)+" Score: "+str(score)+", p (max, predicted): "+str(pred_p)+", p (actual for the same prediction as p_max was made in): "+str(actual_p))

    plt.show() 

    

    actualCategory_label += 1