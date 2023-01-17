# Importing our dataset and some necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv('../input/credit-card-details-australian-bank/Credit_Card_Applications.csv')
dataset
# This dataset has the bank data of 690. The last column that is column 'Class' tells us weather the user comitted fraud or not.
# 0 represents that no fraud was comitted while 1 says that fraud was committed
# Checking for missing values
dataset.count()
# plotting a correlation matrix for our dataset 
matrix=dataset.corr()
fig,ax=plt.subplots(figsize=(10,6))
sns.heatmap(matrix,vmax=0.8,square=True)
# Splitting our data intot test and train. In total we have bank details of 690 users out of which we will train our Artificial 
 # Network on 80% data and 20 % data we will test our networks accuracy
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Feature scaling so that one feature does not have greater weight on the results
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
xtrain=sc.fit_transform(x_train)
xtest=sc.transform(x_test)

# Importing some libraries to buikd our ANN
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Adding the input layer and the first hidden layer with dropout
# Take average of input + output for units/output_dim param in Dense
# input_dim is necessary for the first layer as it was just initialized
classifier=Sequential()
classifier.add(Dense(8, input_dim = 15, kernel_initializer = 'uniform', activation = 'relu' ))
classifier.add(Dropout( 0.2))
classifier.add(Dense(8, kernel_initializer = 'uniform', activation = 'relu' ))
classifier.add(Dropout( 0.2))
classifier.add(Dense(8, kernel_initializer = 'uniform', activation = 'relu' ))
classifier.add(Dropout( 0.2))
classifier.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu' ))
classifier.add(Dropout( 0.2))

classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid' ))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training Set
# x_train, y_train, Batch size, Epochs (whole training set)
classifier.fit(xtrain, y_train, batch_size = 12, epochs = 10)

# Predicting the Test set results
# Note that the output we have got are the probabilities of potential fraud 
# Any probability greater than 50 percent or 0.5 will be considered as 1 and the less than that will be converted to 0
y_pred = classifier.predict(xtest)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting our confusion matrix
plt.matshow(cm)
plt.show() 
cm
# See the array below for better understanding 
 # As you can see we were successfully able to predict fraud with almost 90 percent accuracy, which is a pretty good result.
    # Upvote if you liked this Artificial neural network model
from sklearn.metrics import accuracy_score
print('Accuracy Score:',accuracy_score(y_test,y_pred))
# We have achieved a good precison, recall and f1 score with our model
# Thanks
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
 # Hence we have obtained an accuracy of about 90 percent in successfully predicting Credit card fraud based on bank details.