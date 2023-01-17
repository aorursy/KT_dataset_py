from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
# fix random seed for reproducibility
np.random.seed(7)
# load pima indians dataset
dataset = pd.read_csv('../input/diabetes.csv')
# split into input (X) and output (Y) variables
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
dataset.head(5)
# create model
model = Sequential()
# number of input features
# We can specify the number of neurons in the layer as the first argument,
# the initialization method as the second argument as init (12,8,1) and
# specify the activation function using the activation argument.
# We will use the rectifier (‘relu‘) activation function on the first two layers and the sigmoid function in the output layer.
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train,y_train, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

