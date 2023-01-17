#Importing Libraries

import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__
#Importing the dataset

dataset = pd.read_excel('../input/Final Folder/Dataset/Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Initializing ANN
ann = tf.keras.models.Sequential()


# Adding the input layer and  the first hidden layer 

ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
        
        
# Adding  the second hidden layer
        
ann.add(tf.keras.layers.Dense(units = 6, activation ='relu')) 
         
# Adding the output layer 
         
ann.add(tf.keras.layers.Dense(units = 1)) 
                
# Compiling  the  ANN 
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Training ANN model on the Training set 

ann.fit(X_train, y_train, batch_size = 32, epochs = 100 )
# Predicting the results of Test set

y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))