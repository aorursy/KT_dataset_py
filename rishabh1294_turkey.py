#import the necessary modules
import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split # for splitting the data into test and train
from keras.preprocessing.sequence import pad_sequences # for making the length of the audio files same
from keras.models import Sequential,Model # importing models
from keras.layers import LSTM,Activation, Dropout, Dense 
import itertools # for flattening the list
#listing the files required which are stored in the input directory
print(os.listdir("../input"))
#read data and store in the variables
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')
print(train.columns)
print(test.columns)
print(sample_submission.head())
print(train['audio_embedding'].head())
print(train['is_turkey'].head())
#split the data into training and validation
train_train, train_val = train_test_split(train)

xtrain = train_train['audio_embedding'].tolist()
xval = train_val['audio_embedding'].tolist()

ytrain = train_train['is_turkey'].values
yval = train_val['is_turkey'].values
#padding to make the audio length equal (10 secs)

x_train = pad_sequences(xtrain, maxlen = 10)
x_val = pad_sequences(xval, maxlen = 10)

y_train = np.asarray(ytrain)
y_val = np.asarray(yval)
#initialize the model and add layers
model = Sequential()

model.add(LSTM(128, return_sequences = True))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(LSTM(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))
#compile the model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
#fit the model on the training data
model.fit(x_train, y_train, batch_size = 100, validation_data = (x_val, y_val), epochs = 10)

#compute accuracy of the model on the validation data
loss_score, accuracy = model.evaluate(x_val, y_val, batch_size = 100)

print('Validation accuracy = ', accuracy)
#Prediction on test data
print(test.columns)
print(test['vid_id'].head())
x_test = test['audio_embedding'].tolist()
x_test = pad_sequences(x_test)
test_prediction = model.predict(x_test, batch_size = 100)
print(test_prediction)
#converting to DataFrame and flattening the list 'test_prediction' using itertools.chain
submission = pd.DataFrame({'vid_id':test['vid_id'].tolist(), 'is_turkey':list(itertools.chain(*test_prediction))})
print(submission.head())
final_submission = submission.to_csv('turkey_submission.csv', index = False)