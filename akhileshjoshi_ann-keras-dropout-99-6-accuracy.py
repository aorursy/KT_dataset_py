import pandas as pd
import numpy as np
import os
trainData = pd.read_csv('../input/train.csv')
trainData = trainData.iloc[:,1:]
trainData.head()
testData = pd.read_csv('../input/test.csv')
testData = testData.iloc[:,1:]
testData.head()
trainData.isnull().values.any() #check null values
testData.isnull().values.any() #check null values
#trainData["species"].value_counts()
from sklearn.utils import shuffle
trainData = shuffle(trainData)
trainData = trainData.values
y = trainData[:,0:1]
X= trainData[:,1:].astype(float)
y=pd.DataFrame(y, columns=['species'])
df = pd.get_dummies(y,columns=['species'])
species = [s.replace('species_', '') for s in df.columns.tolist()]
df.columns = species
#df
y = df.values
y.shape
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential  #to initialize the neural network
from keras.layers import Dense  # to build the layers of ANN
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 100,  kernel_initializer ='uniform', activation = 'relu', input_dim = 192))
classifier.add(Dropout(0.2))
# Adding the second hidden layer
classifier.add(Dense(units = 100, kernel_initializer ='uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
# Adding the third hidden layer
classifier.add(Dense(units = 100, kernel_initializer ='uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units=99, kernel_initializer = 'uniform', activation = 'softmax'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
model = classifier.fit(X_train, y_train, batch_size = 5, nb_epoch = 500)
preds = classifier.predict(sc_X.transform(testData))
np.sum(preds[0])
preds.shape
classifier.evaluate(X_test,y_test)[1]
print ( str(classifier.metrics_names[1]) + ":" + str(classifier.evaluate(X_test,y_test)[1] * 100) + "%" )
type(preds)
df1 = pd.DataFrame(preds,columns=species)
df1.shape
testData1 = pd.read_csv('../input/test.csv')
df2 = pd.DataFrame(testData1["id"],columns=["id"])
df2.shape
submission = pd.concat([df2["id"],df1[:]],axis=1)
submission.tail()
submission.to_csv("submission.csv",index=False)

















































