import numpy as np
import pandas as pd
DATASET_PATH = '../input/character-predictions.csv'
df = pd.read_csv(DATASET_PATH)
df.head()
X = df.iloc[:,[7, 16, 17, 18, 19, 20, 25, 26, 28, 29, 30, 31]]
Y = df['isAlive']
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.2)
# no feature preference
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(15, input_dim=12, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1)
y_pred = model.predict(X_test)
y_pred = y_pred > 0.5
from sklearn.metrics import accuracy_score


score = accuracy_score(y_test, y_pred)
'Accuracy score: %.2f' % (score * 100)
