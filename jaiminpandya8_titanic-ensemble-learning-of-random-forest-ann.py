import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

import csv

from sklearn.ensemble import RandomForestClassifier





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()

y = train_data["Survived"]



features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

X["Age"].fillna(method ='ffill', inplace = True)

X_test["Age"].fillna(method ='ffill', inplace = True)



X["Fare"].fillna(method ='ffill', inplace = True)

X_test["Fare"].fillna(method ='ffill', inplace = True)



model_r1 = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=1)

model_r1.fit(X, y)

predictions_r1 = model_r1.predict(X_test)

predictions_r1 = predictions_r1.reshape(len(predictions_r1), 1)



model_r2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model_r2.fit(X, y)

predictions_r2 = model_r2.predict(X_test)

predictions_r2 = predictions_r2.reshape(len(predictions_r2), 1)



model_r3 = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=1)

model_r3.fit(X, y)

predictions_r3 = model_r3.predict(X_test)

predictions_r3 = predictions_r3.reshape(len(predictions_r3), 1)





model1 = Sequential()

model1.add(Dense(1024, input_dim=7, activation='relu'))

# model1.add(Dense(1024, activation='relu'))

model1.add(Dense(1024, activation='relu'))

model1.add(Dense(512, activation='relu'))

# model1.add(Dense(512, activation='relu'))

# model1.add(Dense(256, activation='relu'))

model1.add(Dense(256, activation='relu'))

model1.add(Dense(128, activation='relu'))

# model1.add(Dense(128, activation='relu'))

model1.add(Dense(64, activation='relu'))

# model1.add(Dense(64, activation='relu'))

model1.add(Dense(32, activation='relu'))

# model1.add(Dense(32, activation='relu'))

model1.add(Dense(16, activation='relu'))

# model1.add(Dense(16, activation='relu'))

model1.add(Dense(8, activation='relu'))

# model1.add(Dense(8, activation='relu'))

model1.add(Dense(4, activation='relu'))

# model1.add(Dense(4, activation='relu'))

model1.add(Dense(1, activation='sigmoid'))



model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model1.fit(X, y, epochs=75, batch_size=40)

_, accuracy = model1.evaluate(X, y)

print('Accuracy: %.2f' % (accuracy*100))

predictions1 = model1.predict_classes(X_test)



model2 = Sequential()

model2.add(Dense(640, input_dim=7, activation='relu'))

model2.add(Dense(160, activation='relu'))

model2.add(Dense(40, activation='relu'))

model2.add(Dense(1, activation='sigmoid'))



model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.fit(X, y, epochs=75, batch_size=40)

_, accuracy = model2.evaluate(X, y)

print('Accuracy: %.2f' % (accuracy*100))

predictions2 = model2.predict_classes(X_test)



predictions = np.concatenate((predictions_r1, predictions_r2, predictions_r3, predictions1, predictions2), axis=1)

predictions = np.round(np.mean(predictions, axis = 1))

predictions = predictions.astype(int)

# predictions = predictions.reshape(len(predictions), 1)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

        
