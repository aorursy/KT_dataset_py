import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
primary_data = pd.read_csv('../input/digit-recognizer/train.csv',delimiter=',')
y_train = primary_data['label']
X_train = primary_data.drop(labels='label',axis=1)
X_data = X_train.values
y_data = y_train.values
y_data = np.reshape(y_data,(np.shape(y_data)[0],1))
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X_data)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size = 0.25, random_state = 42)
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)
model = Sequential()
model.add(Dense(392, input_dim=784, activation='relu'))
model.add(Dense(196,activation='relu'))
model.add(Dense(98,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='CategoricalCrossentropy', metrics=['accuracy'])
model.fit(x=X_train,y=y_train_cat,verbose=2,epochs=20)
score, acc = model.evaluate(X_test, y_test_cat)
print('Test score:', score)
print('Test accuracy:', acc)
test_set = pd.read_csv('../input/digit-recognizer/test.csv',delimiter=',')
np.shape(test_set)
predictions = model.predict_classes(test_set)
import csv
with open('test_result_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID','prediction'])
    for i in range(0,np.shape(test_set)[0]-1):
        writer.writerow([i,predictions[i]])