import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## importing the required library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns



## keras modules

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras import optimizers

from keras.layers import LeakyReLU
##Loading data

df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
## Label Distribution of the data

labels = df_train["label"]

dist = labels.value_counts()

print(dist)

sns.barplot(dist.index,dist)
### Converting the data to a numpy array

train = np.array(df_train)

Y_train = train[:,0]

X_train = train[:,1:]
### Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=np.random.seed(1))
model = Sequential()

model.add(Dense(800, input_dim=784, activation="sigmoid",kernel_initializer='random_normal'))

model.add(Dense(600, activation=LeakyReLU(0.3)))

model.add(Dropout(0.2))

model.add(Dense(200, activation=LeakyReLU(0.3)))

model.add(Dropout(0.2))

model.add(Dense(100, activation=LeakyReLU(0.5)))

model.add(Dropout(0.3))

model.add(Dense(50, activation=LeakyReLU(0.5)))

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100, batch_size=512,validation_data=(X_val,Y_val))

_, accuracy_train = model.evaluate(X_train, Y_train)

print('Accuracy_train: %.2f' % (accuracy_train*100))
Y_pred = model.predict_classes(X_val)

cm = confusion_matrix(Y_val, Y_pred)

plt.figure(figsize = (10,7))

sns.heatmap(cm, annot=True,fmt='g')

plt.show()
count = 0

for i in range(Y_pred.shape[0]):

    if Y_pred[i] != Y_val[i] and count<10:

        img = X_val[i].reshape(28,28)

        plt.imshow(img)

        plt.show()

        print("Predited Value:",Y_pred[i])

        print("True Value:",Y_val[i])

        count = count + 1
test = np.array(df_test)

Y_pred = model.predict_classes(test)

imageid = np.array([i for i in range(1,28001)])

out = np.array((imageid,Y_pred))

out_df = pd.DataFrame(out,index = ["ImageId","Label"],dtype = "int64")

out_df = out_df.T

out_df

export_csv = out_df.to_csv("submission.csv",header = True,index = False)