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
#import necessary modules
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
#read data
train=pd.read_csv("../input/digit-recognizer/train.csv")
test=pd.read_csv("../input/digit-recognizer/test.csv")
#split train data
train_label=train["label"]
train_data=train.drop("label", axis=1)
#convert train_data and test data from 0~255 to 0~1
train_data/=255
test/=255
#convert train_label from 0~9 numbers to 10 One-Hot-vector
train_label_ohv=to_categorical(train_label, 10)
train_label_ohv
#convert train_data and test data shape from pandas.DataFrame to numpy.array
train_data_np=train_data.values
test_np=test.values

#convert train_data and test data from (42000, 28, 28) to (42000, 28, 28, 1)
train_data_np=train_data_np.reshape(-1, 28, 28, 1)
test_np=test_np.reshape(-1, 28, 28, 1)

#check each shape of data
print("train_data_np shape is:{0}, test_np shape is: {1}".format(train_data_np.shape, test_np.shape))
#construct CNN model
model=Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
#compile the model 
from tensorflow.keras.optimizers import RMSprop
model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(),
              metrics=["accuracy"])
#fit model to dataset
model.fit(train_data_np, train_label_ohv,
         batch_size=128,
         epochs=10,
         verbose=1)
test_prediction=model.predict(test_np)
#extract predict data using iteration
submission=[]
for i in np.arange(0, 28000):
    submission.append(np.argmax(test_prediction[i, :]))
#prepare for submission data
submission_pd=pd.Series(data=submission, name="Label")

#prepare for making ImageID
ImageID=np.arange(1, len(submission_pd)+1)
ImageID_pd=pd.Series(ImageID, name="ImageID")
submission_file=pd.concat([ImageID_pd, submission_pd], axis=1)
submission_file.to_csv("submission_file", index=False)
