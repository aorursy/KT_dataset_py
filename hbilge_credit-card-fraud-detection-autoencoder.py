import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import TensorBoard
from keras import regularizers
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")
print(data.shape)
print(data.head())
print(data.info())
fraud = data[data.Class==1]
no_fraud = data[data.Class==0]
print("The shape of no_fraud data:", no_fraud.shape)
print("The shape of fraud data:", fraud.shape)
plt.figure(figsize=(12,5))
sns.countplot(x="Class", data=data)
plt.title("Fraud vs No Fraud Transaction Distributions")
plt.xticks(range(2), ["No Fraud", "Fraud"])
plt.show() 
print("Non-Fraudulent Transaction")
print(no_fraud.Amount.describe())
print("\nFraudulent Transaction")
print(fraud.Amount.describe())
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.distplot(no_fraud['Amount'])
plt.title("Amount Distribution of Non-Fraudulent Transaction")

plt.subplot(1,2,2)
sns.distplot(fraud['Amount'])
plt.title("Amount Distribution of Fraudulent Transaction")

plt.show()
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
plt.scatter(no_fraud.Time, no_fraud.Amount)
plt.title("Time Distribution of Non-Fraudulent Transaction")
plt.xlabel("time")
plt.ylabel("amount")

plt.subplot(1,2,2)
plt.scatter(fraud.Time, fraud.Amount)
plt.title("Time Distribution of Fraudulent Transaction")
plt.xlabel("time")
plt.ylabel("amount")

plt.show()
data = data.drop(['Time'], axis=1)
from sklearn.preprocessing import StandardScaler

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

X_train = X_train[X_train.Class==0]
X_train = X_train.drop(['Class'], axis=1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
input_size = X_train.shape[1]
encoding_size = 14
input_layer = Input(shape=(input_size, ))

encoder_1 = Dense(encoding_size, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_layer)
encoder_2 = Dense(int(encoding_size / 2), activation="relu")(encoder_1)
decoder_1 = Dense(encoding_size, activation='relu')(encoder_2)
output_layer = Dense(input_size, activation='relu')(decoder_1)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='./logs', write_images=True)

history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, shuffle=True, validation_data=(X_test, X_test),
                  callbacks=[tensorboard]).history
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');

plt.subplot(1,2,2)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.show()
prediction = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - prediction, 2), axis=1)
error = pd.DataFrame({'reconstruction_error': mse, 'actual_class': y_test})
without_fraud = error[error.actual_class==0]
with_fraud = error[error.actual_class==1]
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.distplot(without_fraud["reconstruction_error"])
plt.title("Reconstruction error without fraud")

plt.subplot(1,2,2)
sns.distplot(with_fraud["reconstruction_error"])
plt.title("Reconstruction error with fraud")

plt.show()
from sklearn.metrics import confusion_matrix
y_pred = [1 if e > 3 else 0 for e in error.reconstruction_error.values]
print('Confusion Matrix\n' + str(confusion_matrix(y_test, y_pred)))