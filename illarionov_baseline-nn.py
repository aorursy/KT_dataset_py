import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Activation

df = pd.read_csv('../input/dense-network/train.csv')
df.head()
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.label, train_size=0.7)
model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), activation='elu'),
    Dense(64, activation='elu'),
    Dense(128, activation='elu'),
    Dense(16, activation='elu'),
    Dense(1, activation='sigmoid'),
])

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train.values, y_train, epochs=20, batch_size=64, validation_split=0.2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
pred = model.predict(X_test) > 0.5
print(classification_report(y_test, pred))