import numpy as np

import pandas as pd

dataset = pd.read_csv("../input/Churn_Modelling.csv")#importing the dataset

X = dataset.iloc[:, 3:13].values #taking independent variables i.e.columns. We have neglected C0, C1 and C3 as model won't depend on these factors

y = dataset.iloc[:, 13].values #taking the dependent variable i.e "Exited" column

X

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

l_e_X1 = LabelEncoder()

X[:, 1] = l_e_X1.fit_transform(X[:,1])# For "Geography" feature

l_e_X2 = LabelEncoder()

X[:,2] = l_e_X2.fit_transform(X[:,2]) #For "Gender" feature

X

ohe = OneHotEncoder(categorical_features=[1])

X = ohe.fit_transform(X).toarray()

X = X[:, 1:]#taken only two rows of "Geography" feature to avoid dummy variable trap.

X

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #Dividing the dataset into training and testing set. 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

X_train

X_test
from keras.models import Sequential #Used for initializing our artificial neural network classifier .

from keras.layers import Dense #Used to create layers to the network.

classifier = Sequential()

#Initializing our model and creating first hidden layer with number of nurons =8 of first hidden layer,init is used to initialize weights,

#which is selected uniform. Rectifier function is taken as activation function and the last parameter "input_dim" is the number  of input features.

classifier.add(Dense(output_dim = 8, init = "uniform", activation = "relu", input_dim = 11))

#Second Hidden Layer

classifier.add(Dense(output_dim = 8, init = "uniform", activation = "relu"))

#Third Hidden Layer

classifier.add(Dense(output_dim = 8, init = "uniform", activation = "relu"))

#Output layer contain only one neuron i.e either "0" or "1"with activation taken as sigmoid function.

classifier.add(Dense(output_dim = 1, init = "uniform", activation = "sigmoid"))

#Now we are going to compile our neural network. Optimizer is used to find the best weights and for that we are using stochastic gradient descent.

#Loss function is logrithmic class. Metrics is the creteria used to evaluate our model and we have choosen accuracy as the criteria.

classifier.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ["accuracy"])
classifier.fit(X_train,y_train,batch_size = 1000, epochs = 200)
y_pred = classifier.predict(X_test) #Gives us the probability of employee leaving the bank

print(y_pred)

y_pred = (y_pred>0.5) #We use  t hreshold as 0.5

print(y_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
