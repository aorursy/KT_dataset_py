import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt

%matplotlib inline

!ls ../input/
types = ['containership', 'cruiser', 'destroyer','coastguard', 'smallfish', 'methanier', 'cv', 'corvette', 'submarine', 'tug']
types_id = dict(zip(types, range(len(types))))

ships = np.load('../input/ships2020.pnz')
X_data = ships['X']
Y_data = ships['Y']
len(X_data)
from keras.utils import np_utils

ran = np.random.randint(len(X_data), size = len(X_data)//10)
filtre = np.array([True,] * len(X_data))
filtre[ran] = False

X_train = X_data[filtre]
Y_train = np_utils.to_categorical(Y_data[filtre])
len(X_train)
i = np.random.randint(len(Y_data))
print("Ship #%d is a %s" % (i,types[Y_data[i]]))
print(Y_data[i])
plt.imshow(X_data[i])
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten

inputs = Input(shape=X_train[0].shape, name='cnn_input')
x = Flatten()(inputs)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(len(types), activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(X_train, Y_train, epochs=1, batch_size=8, validation_split=0.1)

X_test = X_data[~filtre]
Y_test = np_utils.to_categorical(Y_data[~filtre])


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
from sklearn.metrics import classification_report, confusion_matrix

res = model.predict(X_test).argmax(axis=1)
confu = confusion_matrix(Y_test.argmax(axis=1), res)
pd.DataFrame({types[i][:3]:confu[:,i] for i in range(len(types))}, index=types)
print(classification_report(Y_test.argmax(axis=1), res, target_names=types))
res
# predict results
res = model.predict(X_test).argmax(axis=1)
df = pd.DataFrame({"Category":res})
df.to_csv("reco_nav.csv", index_label="Id")