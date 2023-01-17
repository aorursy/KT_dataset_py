import numpy as np; import pandas as pd; 

from pandas import DataFrame

from keras.models import Sequential; 

from keras.layers import Convolution1D, Flatten, Dense, Dropout,MaxPooling1D, regularizers

from keras.utils import to_categorical

from keras import optimizers



np.random.seed(7)

train_csv = pd.read_csv('./input/train.csv')

test_csv = pd.read_csv('./input/test.csv')



X_training_set = train_csv.iloc[:,1:785].values   

X_test_set = test_csv.iloc[:,:].values

y_training_set = train_csv.iloc[:,0].values       

categorical_labels = to_categorical(y_training_set, num_classes=10)



## 94.614% code

classifier = Sequential()

classifier.add(Dense(64, input_dim = 784, activation='relu'))

classifier.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001)))

classifier.add(Dropout(0.1))

classifier.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001)))

classifier.add(Dropout(0.2))

classifier.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001)))

classifier.add(Dropout(0.2))

classifier.add(Dense(10, activation = 'softmax'))

classifier.compile('Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])      



classifier.fit(X_training_set, categorical_labels, epochs=30)

testoutput = classifier.predict(X_test_set)



predictions_int = []



for i in range(len(testoutput)):

    predictions_int.append(int(np.argmax(testoutput[i,:])))



# np.savetxt('predictions.csv', predictions_int, fmt='%i')



predictions_file = pd.DataFrame(predictions_int)

predictions_file.index = np.arange(1,len(predictions_file)+1)

predictions_file.columns = ["Label"]

predictions_file.to_csv('predictions.csv')