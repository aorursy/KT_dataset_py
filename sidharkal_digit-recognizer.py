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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from keras.models import Sequential

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.utils import to_categorical
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train.head()
train.shape 
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test.head()
test.shape
X_train = train.drop("label",axis=1).to_numpy()

y_train = train["label"]

X_test = test.to_numpy()
sns.countplot(y_train)

y_train.value_counts()
X_train.shape ,X_test.shape
X_train = X_train.reshape(X_train.shape[0], 28, 28)
plt.figure()

plt.imshow(X_train[0])

plt.colorbar()

plt.grid(False)

plt.show()
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_train.shape ,X_test.shape
X_train = X_train / 255.0

X_test = X_test / 255.0
Y_train = to_categorical(y_train, num_classes = 10)

Y_train[0]
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
X_train.shape , Y_train.shape
X_val.shape , Y_val.shape
model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=["accuracy"])
model.summary()
datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16),

                           steps_per_epoch=500,

                           epochs=20, 

                           verbose=2,  

                           validation_data=(X_val[:400,:], Y_val[:400,:]))
final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
plt.plot(hist.history['accuracy'], label='train')

plt.plot(hist.history['val_accuracy'], label='valid')

plt.legend(loc='upper left')

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
plt.plot(hist.history['loss'], label='train')

plt.plot(hist.history['val_loss'], label='test')

plt.legend(loc='upper right')

plt.title('Model Cost')

plt.ylabel('Cost')

plt.xlabel('Epoch')

plt.show()
sample = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

sample.head()
prediction = model.predict_classes(X_test)



submission = pd.DataFrame({"ImageId": sample.ImageId,

                         "Label": prediction})

submission.to_csv("solution.csv", index=False, header=True)
submission.head()
from IPython.display import HTML

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(submission)