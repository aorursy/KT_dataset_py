# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
'''
Just a quick DNN model fitting with Keras.
'''
#reading the data
import pandas as pd
data = pd.read_csv(r'../input/mushrooms.csv', sep = ',', header = 0)
'''
Every one of the variables are categorical, so I need to transform them into one hot encoded variables 
'''
data.describe()

'''
Turning the categories into numeric values.
'''

from sklearn.preprocessing import LabelEncoder

column_names = data.columns
for names in column_names:
    Label_enc = LabelEncoder()
    data[names] = Label_enc.fit_transform(data[names].values) 


    
'''
Using get_dummies function in pandas to turn the numeric values into one hot encoded variables.
The loop goes through all the variables, and simply concatenates them all. 
'''
target = data['class']

for names in column_names.drop('class'):
    target = pd.concat([target.reset_index(drop=True), pd.get_dummies(data[names], prefix = [names], drop_first = True)], axis = 1)
'''
Just a quick peak to see that the data is transformed
'''

target.head(10)
from sklearn.model_selection import train_test_split

'''
creating a 60 / 20 / 20 train, validation, test split
'''

X_train, X_test, y_train, y_test = train_test_split(target[target.columns.drop('class')] ,target['class'], test_size = 0.40, random_state = 777)
X_val, X_test, y_val, y_test = train_test_split(X_test ,y_test, test_size = 0.50, random_state = 777)


from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from matplotlib import pyplot 

model = Sequential()
model.add(Dense(128, input_shape = (len(X_train.columns),)))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])

model.fit(X_train,
         y_train,
         epochs = 50,
         batch_size = 256,
         validation_data = (X_val, y_val))


'''
The model has 100% accuracy with the test set
'''

from sklearn.metrics import accuracy_score
y_prediction = model.predict_classes(X_test)
print("The Accuracy of the model is {}%".format(accuracy_score(y_test, y_prediction) * 100.)) 
'''
More things to explore

1) Early stopping for keras using callbacks. Looking at the model output history, the model fitting process 
achived 100% validation accuracy after 11 epochs. One can try to use early stopping mechanisms built into keras
using callbacks to terminate the model fitting process early to reduce wasted compute time. 
'''