# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd                # pandas is a data frame library

import keras

from keras.models import Sequential

import matplotlib.pyplot as plt    # matplot.pyplot plots data

from keras.layers import Dense

import numpy
df = pd.read_csv("/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv", sep=';')
df.shape
df.head()
del df['id']

df.head()
df['age'] = df['age'].map(lambda x : x // 365)

df.head(5)
df.isnull().values.any()
df.hist(figsize=(10,12))

plt.show()
# Visualizing the data

dataset_plot = df

dataset_plot[['active','age','alco','ap_hi','ap_lo','cholesterol','gender','gluc','height','smoke','weight']].head(100).plot(style=['o','x','r--','g^'])

plt.legend(bbox_to_anchor=(0.,1.02,1., .102), loc=3,ncol=2, mode="expand", fontsize="x-large", borderaxespad=0.)

plt.show()
# numpyMatrix=numpy.array(df.values, dtype = numpy.float64)

# X_input = numpyMatrix[:,0:11]

# X=X_input

# Y = numpyMatrix[:,8]



X = df.drop(['cardio'], axis=1)

Y = df['cardio']



# create model

model = Sequential()

model.add(Dense(32, input_dim=11, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./diabetes/logs', histogram_freq=0, write_graph=True, write_images=True)



# Fit the model

history = model.fit(X, Y,validation_split=0.20, epochs=100, batch_size=16,callbacks=[tbCallBack])



# evaluate the model

scores = model.evaluate(X, Y)



print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.predict(X)
# Comparing Validation and training results

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
myX_false = [[47,1,156,56.0,100,60,1,1,0,0,0]]

myX_true = [[48,2,169,82.0,150,100,1,1,0,0,1]]

matrixX=numpy.array(myX_false, dtype = numpy.float64)

X_input = matrixX[:,0:11]

model.predict(X_input)