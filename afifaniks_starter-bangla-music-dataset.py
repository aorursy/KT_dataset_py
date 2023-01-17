import keras

from keras.models import Sequential

from keras.regularizers import l2, l1

from keras.layers import Dense, Dropout

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

import numpy

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import cm

%matplotlib inline

import seaborn as sn
!pwd
df = pd.read_csv('../input/dataset.csv', encoding='latin-1')

pd.options.display.max_columns=None

df.head(2)
X = df['label'].value_counts()



classes = X.keys()



label = {}



for c in classes:

    label[c] = X[c]



plt.bar(label.keys(), label.values())

plt.xlabel('Class Name')

plt.ylabel('Number of Instances')



print(X)
import numpy as np
data_adhunik = df[df['label'] == 'adhunik']

data_band = df[df['label'] == 'band']

data_hiphop = df[df['label'] == 'hiphop']

data_nazrul = df[df['label'] == 'nazrul']

data_palligeeti = df[df['label'] == 'palligeeti']

data_rabindra = df[df['label'] == 'rabindra']
data_list = [data_adhunik, data_band, data_hiphop, data_nazrul, data_palligeeti, data_rabindra]

classes = np.array(['adhunik', 'band', 'hiphop', 'nazrul', 'palligeeti', 'rabindra'])
# Zero Crossings Visualization

col_name = 'zero_crossing'

print(col_name)

y = []

for data in data_list:

    print(data[col_name].mean())

    y.append(data[col_name].mean())

y = np.array(y)

colors = cm.hsv(y / float(max(y)))

plot = plt.scatter(y, y, c = y, cmap = 'hsv')

plt.clf()

plt.colorbar(plot)

plt.bar(classes, y, color = colors, width=0.6)

plt.show()
# Zero Crossings Visualization

col_name = 'spectral_centroid'

print(col_name)

y = []

for data in data_list:

    print(data[col_name].mean())

    y.append(data[col_name].mean())

y = np.array(y)

colors = cm.hsv(y / float(max(y)))

plot = plt.scatter(y, y, c = y, cmap = 'hsv')

plt.clf()

plt.colorbar(plot)

plt.bar(classes, y, color = colors, width=0.6)

plt.show()
# Zero Crossings Visualization

col_name = 'spectral_rolloff'

print(col_name)

y = []

for data in data_list:

    print(data[col_name].mean())

    y.append(data[col_name].mean())

y = np.array(y)

colors = cm.hsv(y / float(max(y)))

plot = plt.scatter(y, y, c = y, cmap = 'hsv')

plt.clf()

plt.colorbar(plot)

plt.bar(classes, y, color = colors, width=0.6)

plt.show()
# Zero Crossings Visualization

col_name = 'spectral_bandwidth'

print(col_name)

y = []

for data in data_list:

    print(data[col_name].mean())

    y.append(data[col_name].mean())

y = np.array(y)

colors = cm.hsv(y / float(max(y)))

plot = plt.scatter(y, y, c = y, cmap = 'hsv')

plt.clf()

plt.colorbar(plot)

plt.bar(classes, y, color = colors, width=0.6)

plt.show()
# Zero Crossings Visualization

col_name = 'chroma_frequency'

print(col_name)

y = []

for data in data_list:

    print(data[col_name].mean())

    y.append(data[col_name].mean())

y = np.array(y)

colors = cm.hsv(y / float(max(y)))

plot = plt.scatter(y, y, c = y, cmap = 'hsv')

plt.clf()

plt.colorbar(plot)

plt.bar(classes, y, color = colors, width=0.6)

plt.show()
# Zero Crossings Visualization

col_name = 'tempo'

print(col_name)

y = []

for data in data_list:

    print(data[col_name].mean())

    y.append(data[col_name].mean())

y = np.array(y)

colors = cm.hsv(y / float(max(y)))

plot = plt.scatter(y, y, c = y, cmap = 'hsv')

plt.clf()

plt.colorbar(plot)

plt.bar(classes, y, color = colors, width=0.6)

plt.show()
# Zero Crossings Visualization

col_name = 'zero_crossing'

print(col_name)

y = []

for data in data_list:

    print(data[col_name].mean())

    y.append(data[col_name].mean())

y = np.array(y)

colors = cm.hsv(y / float(max(y)))

plot = plt.scatter(y, y, c = y, cmap = 'hsv')

plt.clf()

plt.colorbar(plot)

plt.bar(classes, y, color = colors, width=0.6)

plt.show()
# Zero Crossings Visualization

col_name = 'rmse'

print(col_name)

y = []

for data in data_list:

    print(data[col_name].mean())

    y.append(data[col_name].mean())

y = np.array(y)

colors = cm.hsv(y / float(max(y)))

plot = plt.scatter(y, y, c = y, cmap = 'hsv')

plt.clf()

plt.colorbar(plot)

plt.bar(classes, y, color = colors, width=0.6)

plt.show()
# Splitting data

y = df['label']

X = df.drop(['file_name', 'label'], axis=1)



# Scaling values by normal distribution

sc = StandardScaler()

X = sc.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8) #4



encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)

y_test = encoder.fit_transform(y_test)
encoder.classes_
encoder.inverse_transform([0, 1, 2, 3, 4, 5])
print(X_train.shape, X_test.shape)
# Creating model

model = Sequential()

model.name="Bangla Music Genre Classifier"

model.add(Dense(256, activation='relu', input_dim=29, name="First_Layer", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.002)))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu', name="Second_Layer", kernel_regularizer=l2(0.002), bias_regularizer=l2(0.0001)))

model.add(Dropout(0.4))

model.add(Dense(64, activation='relu', name="Third_Layer", kernel_regularizer=l1(0.002), bias_regularizer=l2(0.001)))

model.add(Dropout(0.4))

model.add(Dense(64, activation='relu', name="Fourth_Layer", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.0001)))

model.add(Dropout(0.4))

model.add(Dense(6, activation='softmax', name="Output_Layer", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.002)))

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



history = model.fit(X_train, y_train, batch_size=200, validation_data=(X_test, y_test), validation_split=0.2, epochs=100)
y_predict = model.predict_classes(X_test)

y_predict
history.history.keys()
score, acc = model.evaluate(X_test, y_test)

print(score, acc)
# model.save('model.h5')
# Summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Train vs Validation Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['training accuracy', 'validation accuracy'], loc='lower right')

plt.show()



# Summarize history for loss

# plt.plot(history.history['loss'])

# plt.title('model loss')

# plt.ylabel('loss')

# plt.xlabel('epoch')

# plt.legend(['train', 'test'], loc='upper left')

# plt.show()
# Summarize history for accuracy

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Train Loss vs Validation Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['training loss', 'vaidation loss'], loc='upper right')

plt.show()
history.history.keys()
# Confusion Matrix

con_mat = confusion_matrix(y_test, y_predict)

df_cm = pd.DataFrame(con_mat, columns=encoder.classes_, index=encoder.classes_)

sn.heatmap(df_cm, annot=True)
print(classification_report(y_test, y_predict, target_names=encoder.classes_))
false = 0

true = 0

for i in range(len(y_predict)):

    if y_predict[i] == y_test[i]:

        true += 1

    else:

        false += 1

print('Total: ', false + true, 'True: ', true, 'False: ', false)