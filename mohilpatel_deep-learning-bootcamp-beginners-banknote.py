# import libraries



import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score



from keras.models import Sequential

from keras.layers import Dense, Input
# import data



# train data

bank_note_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/bank_note_data/training_set_label.csv" )

# test data

test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/bank_note_data/testing_set_label.csv')
# understand data

bank_note_data.describe()
# check if any column contains null

bank_note_data.isna().any()
# get the distribution of classes

bank_note_data.Class.value_counts().plot.bar()
# splitting the data into train and test set

x_train, x_test, y_train, y_test = train_test_split(bank_note_data.drop(['Class'], axis=1), bank_note_data.Class, test_size=0.2, random_state=0)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
# simple model for classification

model = Sequential()

model.add(Input(shape=(1,4)))

model.add(Dense(10, activation='relu'))

model.add(Dense(4, activation='relu'))

model.add(Dense(2, activation='relu'))

model.add(Dense(1, activation='sigmoid'))  # sigmoid activation because it returns value in range [0,1] and it's a classification problem

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='acc')
# fitting the model

model.fit(x_train, y_train, validation_data=(x_test, y_test) , epochs=50, verbose=2)
# prediction on validation set

validation_predictions = model.predict(x_test)

validation_predictions = [int(x>0.5) for x in validation_predictions] #converting to binary



# evaluating performance

print('Accuracy = ', accuracy_score(y_test, validation_predictions))

print('Precision = ', precision_score(y_test, validation_predictions))

print('Recall = ', recall_score(y_test, validation_predictions))
# generating predictions on test data fro submission

test_predictions = model.predict(test_data)

test_predictions = [int(x>0.5) for x in test_predictions]



# saving predictions to csv

submission = pd.DataFrame(test_predictions, columns=['prediction'])

print(submission)

submission.prediction.to_csv('submissions.csv', index=False)