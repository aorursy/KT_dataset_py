import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense,BatchNormalization

import matplotlib.pyplot as plt
data = pd.read_csv('../input/heart.csv')
data.shape
data.head()
catagorialList = ['sex','cp','fbs','restecg','exang','ca','thal']

for item in catagorialList:

    data[item] = data[item].astype('object')
data.dtypes
data = pd.get_dummies(data, drop_first=True)
data.head()
y = data['target'].values

y = y.reshape(y.shape[0],1)

x = data.drop(['target'],axis=1)

minx = np.min(x)

maxx = np.max(x)

x = (x - minx) / (maxx - minx)

x.head()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
model = Sequential()

model.add(Dense(12, input_dim=21, activation='sigmoid'))

model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
output = model.fit(x_train, y_train,validation_split=0.2, epochs=200, batch_size=x_train.shape[0]//2)
scores = model.evaluate(x_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
plt.plot(output.history['acc'])

plt.plot(output.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.savefig('Accuracy.png',dpi=100)

plt.show()
plt.plot(output.history['loss'])

plt.plot(output.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.savefig('Loss.png',dpi=100)

plt.show()