# This notebook picks up MDIST data and uses Multi Layer Perceptron to classify handwritten digits

# using the MLP model. It also displays the misclassified images where it is evident that the source

# was difficult to classify





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt;

from tensorflow import keras;



mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

Y_train = mnist_train['label'];

X_train = mnist_train.drop('label', axis=1);

Y_test =  mnist_test['label'];

X_test = mnist_test.drop('label', axis=1);



#print(X_train.iloc[0]);



#display a raw image of any number

def display_sample(num, X_data, Y_data):

    pixel_data = np.array(X_data.iloc[num]).reshape([28,28]);

    label_data = Y_data[num];

    print("Number=", label_data);

    plt.imshow(pixel_data,cmap=plt.get_cmap('gray_r'));

    plt.show();

    return;



#for i in range(0,10):

    #display_sample(i, X_train, Y_train);

    

# preprocess the mnist data to normalize the grayscale to 1-0 &

# one_hot_encode the labels

def preprocess_data(input_x, input_y):

    unique_items = 10;

    y = keras.utils.to_categorical(input_y, unique_items);

    x = input_x/255;

    return(x,y);



#************************************************

# Train the data using Keras 

#************************************************

x_tr, y_tr = preprocess_data(X_train, Y_train);



from tensorflow.python.keras.models import Sequential;

from tensorflow.python.keras.layers import Dense,Activation;

model = Sequential();

#create layers

model.add(Dense(784,input_dim=784, activation="relu"));

model.add(Dense(392, activation="relu"));

model.add(Dense(10, activation="softmax"));

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy']);



# Now train the model using the training data;

model.fit(x_tr, y_tr, batch_size=128, epochs=2);



x_tst, y_tst = preprocess_data(X_test, Y_test);



y_predict = model.predict(x_tst, verbose=0);



#print("Y predict shape=",y_predict.shape);



# display the numbers that were misclassified

count = 0;

for i in range(500):

    hot_label = y_predict[i];

    label = y_predict[i].argmax(axis=0);

    actual_label = Y_test[i];

    if (label != actual_label):

        count = count + 1

        print(hot_label);

        print("Predicted Label =",label);

        print("Actual Label =",actual_label);

        display_sample(i, X_test, Y_test);

       

        

print("Misclassified Count = ",count);









          

          













    








