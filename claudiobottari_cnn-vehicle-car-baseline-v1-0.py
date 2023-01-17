import pandas as pd



from sklearn import metrics

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D

from tensorflow.keras.optimizers import Adam



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/mobile-accelerometer-car-12k/mobile_accelerometer_car_12K.csv')

df.head()
df.describe()
X = df[['acc_x', 'acc_y', 'acc_z']].values

y = df[['target']].values

print(X.shape, y.shape)
X = X.reshape(df.shape[0]//100, 100, 3)

y = y.reshape(df.shape[0]//100, 100)[:, 0]

print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23, shuffle=True)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
#create model

model = Sequential([

    Conv1D(16, kernel_size=3, activation='relu', input_shape=X.shape[1:], name='conv_1'),

    MaxPooling1D(name='max_pool_1'),

    Conv1D(32, kernel_size=3, activation='relu', name='conv_2'),

    MaxPooling1D(name='max_pool_2'),

    Conv1D(16, kernel_size=3, activation='relu', name='conv_3'),

    MaxPooling1D(name='max_pool_3'),

    GlobalMaxPooling1D(name='global_max'),

    Dense(1, activation='sigmoid', name='fully_connected_output')], name='vehicle_cnn_baseline')



model.summary()
model.compile(optimizer=Adam(learning_rate=3e-3), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), verbose=2, batch_size=150)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Accuracy through training')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss through training')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)



ax = plt.subplot()

cm = metrics.confusion_matrix(y_test, (y_pred > 0.5))

sns.heatmap(cm, cmap= 'Blues', annot= cm, ax = ax, fmt="d")



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')

ax.set_title('Test Confusion Matrix')

ax.xaxis.set_ticklabels(['Other', 'Car'])

ax.yaxis.set_ticklabels(['Other', 'Car'])

plt.show()
# accuracy: (tp + tn) / (p + n)

accuracy = metrics.accuracy_score(y_test, (y_pred > 0.5))

print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)

precision = metrics.precision_score(y_test, (y_pred > 0.5))

print('Precision: %f' % precision)

# recall: tp / (tp + fn)\\\\\

recall = metrics.recall_score(y_test, (y_pred > 0.5))

print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)

f1 = metrics.f1_score(y_test, (y_pred > 0.5))

print('F1 score: %f' % f1)