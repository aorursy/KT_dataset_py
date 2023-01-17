import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline 
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

from keras import backend as k

from keras.utils import to_categorical

from keras.optimizers import RMSprop, Adam
X_train=pd.read_csv("../input/train.csv")

X_test_sub=pd.read_csv("../input/test.csv")
plt.imshow(X_train.drop('label',axis=1).loc[1].values.reshape(28,28),cmap='gray')

print(X_train.loc[1]['label'])
img_rows,img_cols=28,28

batch_size=128

num_classes=10

epochs=12
y_train=X_train.pop('label')
X_train=X_train/255

X_test_sub=X_test_sub/255
X_train=X_train.values.reshape(-1,28,28,1)

X_test_sub=X_test_sub.values.reshape(-1,28,28,1)
X_train.shape,X_test_sub.shape
input_shape=(img_rows,img_cols,1)
y_train=to_categorical(y_train)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
# Implementing the CNN architecture for image recognition

model=Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,

             optimizer=keras.optimizers.Adadelta(),

             metrics=['accuracy'])

model.summary()
model.fit(X_train,y_train,batch_size=batch_size,epochs=20,validation_data=(X_test,y_test),verbose=1)
def create_model_confusion_matrix(model, X_input, y_expected):



    # let the model predict the output given X_input

    y_predicted = model.predict(X_input)

    

    # convert predicted and expected output from one-hot vector to label

    y_predicted_classes = np.argmax(y_predicted, axis=1)

    y_expected_classes = np.argmax(y_expected, axis=1)



    cm = confusion_matrix(y_expected_classes, y_predicted_classes)

    df_cm = pd.DataFrame(cm, range(10), range(10))

    

    ax = sns.heatmap(df_cm)

    ax.set(xlabel='expected', ylabel='predicted')

    ax.set_title('Confusion Matrix')

    plt.show()

    

    return df_cm



create_model_confusion_matrix(model, X_test, y_test)
y_test_pred=model.predict(X_test_sub)



y_test_pred = np.argmax(y_test_pred, axis=1)
y_test_pred = pd.Series(y_test_pred, name="Label")



# Conversion to CSV file

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), y_test_pred], axis=1)

submission.to_csv("cnn_mnist_submissions.csv",index=False)