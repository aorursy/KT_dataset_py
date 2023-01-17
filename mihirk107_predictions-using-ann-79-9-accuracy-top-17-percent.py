# This code I have developed is not perfect. There are a lot of improvements which can be done. Any improvements would be welcome.

# I have tried to explain it as much as possible. If anyone has any doubts, please feel free to ask in the comments section.

# I have obtained an accuracy of 79.9% using this code.

# Importing the libraries.

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import confusion_matrix # For construction of the Confusion Matrix.

import seaborn as sns # I have used this library for visualising the Confusion Matrix.

# The Keras library is for the implementation of the Neural Network. 

import keras 

from keras.models import Sequential # For defining the type of the Neural Network

from keras.layers import Dense  # For defining the layers of the Neural Network

from keras.layers import Dropout # For Dropout Regularization

from keras.optimizers import RMSprop # For the RMSprop optimizer 

from keras.callbacks import ReduceLROnPlateau # For Simulated Annealing

from keras.wrappers.scikit_learn import KerasClassifier



# Importing the dataset.

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# Dropping the unncessary columns.

train = train.drop(["Name", "Cabin", "Embarked", "PassengerId"], axis = 1)

test1 = test.drop(["Name", "Cabin", "Embarked", "PassengerId"], axis = 1) #Returns a dataframe.

test = test.drop(["Name", "Cabin", "Embarked", "PassengerId"], axis = 1).values #Returns a numpy object.

# X is a set of independent variables using which we will predict the dependent variable. 

X1 = train.drop(["Survived"], axis = 1) # Returns a dataframe.

X = train.drop(["Survived"], axis = 1).values # Returns a numpy object.

y = train.iloc[:, 0].values # y is the dependent variable that is whether the passengers survived  or not.



# Combining the Training and the Test Data.

frames = [X1, test1]

result = pd.concat(frames ,keys = ['X','test'])

# We converted the result dataframe to a numpy object. 

# But why? ->Because the Imputer class cannot work on a dataframe.

result = result.values



# Checking the dataset for missing values.

# result.isnull().describe

# Handling the missing values.

from sklearn.preprocessing import Imputer

#Here we define an object of the Imputer class and our strategy is to replace the missing values (NaN) by the mean of the respective column.

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) 

imputer = imputer.fit(result[:, 2:3])

result[:, 2:3] = imputer.transform(result[:, 2:3])

imputer = imputer.fit(result[:, -1:])

result[:, -1:] = imputer.transform(result[:, -1:])



# Encoding the values that is mapping the strings to integers.

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder = LabelEncoder()

result[:,5] = labelencoder.fit_transform(result[:,5])

result[:,1] = labelencoder.fit_transform(result[:,1])

# Here we perform One Hot Encoding to remove any dependencies between the Encoded Values.

# For example, if we have a column named country with three countries A,B,C.

# If the model maps A to 0, B to 1, C to 2 then since 1>2, the model should not consider the country B greater than C.

onehotencoder = OneHotEncoder(categorical_features = [1])

result = onehotencoder.fit_transform(result).toarray()

onehotencoder = OneHotEncoder(categorical_features = [5])

result = onehotencoder.fit_transform(result).toarray()



# Feature Scaling (Reuqired for Neural Network to reduce computation).

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

result = sc.fit_transform(result)



# Splitting the combined dataframe back to the originial dataframes

result1 = pd.DataFrame(result)

X1 = result1.iloc[0:891, :].values

test1 = result1.iloc[891:, :].values 

#I have used the used following formula for the number of nodes in the hidden layer.

#Nh=Ns/(α∗(Ni+No))

#Ni  = number of input neurons.

#No = number of output neurons.

#Ns = number of samples in training data set.

#α = an arbitrary scaling factor usually 2-10.

Nh = int(891/32)

# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = Nh, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

#Adding dropout regularization to prevent overfitting without dropout I got an accuracy of 75 and with dropout it increased to 78.

classifier.add(Dropout(0.01))



# Adding the second hidden layer

classifier.add(Dense(units = Nh, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.01))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.add(Dropout(0.01))



# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model

classifier.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])



#In order to make the optimizer converge faster and closest to the global minimum of the loss function, i used an annealing method of the learning rate (LR).

#The LR is the step by which the optimizer walks through the 'loss landscape'. The higher LR, the bigger are the steps and the quicker is the convergence. However the sampling is very poor with an high LR and the optimizer could probably fall into a local minima.

#Its better to have a decreasing learning rate during the training to reach efficiently the global minimum of the loss function.

#To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically every X steps (epochs) depending if it is necessary (when accuracy is not improved).

#With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy is not improved after 3 epochs.

# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



# Fitting the ANN to the Training set

history = classifier.fit(X1, y, batch_size = 25, epochs = 1000, callbacks = [learning_rate_reduction])

# I have commented the Confusion Matrix code as I have directly predicted the test set.

# We create a Confusion Matrix to check the performance of our model that is how many correct and incorrect predictions it has made.

# Creating the Confusion Matrix

#confusion_mtx = confusion_matrix(test1, final) 

# Visualise the Confusion Matrix 

#sns.heatmap(confusion_mtx, annot=True, fmt='d')

# Predicting the Test set results

final = classifier.predict(test1)

# Creating the final dataframe in the required format

final = (final > 0.5)

final = final.astype(int)

final = pd.DataFrame(final)

final['PassengerId'] = pd.Series(data = np.arange(892,1310), index=final.index)

final.columns = ['Survived','PassengerId']

columnsTitles=["PassengerId","Survived"]

final=final.reindex(columns=columnsTitles)



# Exporting the dataframe

final.to_csv('Predictions_ANN.csv', index = False)


