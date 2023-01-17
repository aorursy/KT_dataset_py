import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.utils import normalize

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
X = train[[col for col in train.columns if "stop" in col]]
y = train[[col for col in train.columns if "start" in col]]
X= np.array(X)

y = np.array(y)
#X = normalize(X)

#y = normalize(y)
X= X.reshape(-1, 20, 20, 1)
model = Sequential()

# Adds a densely-connected layer with 64 units to the model:

model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))

model.add(MaxPooling2D(pool_size = (2,2)))

# Add another:

model.add(Conv2D(64,(3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Flatten())

model.add(Dense(64, activation='relu'))

# Add a softmax layer with 10 output units:

model.add(Dense(400, activation='softmax'))



model.compile(optimizer="adam",

              loss='binary_crossentropy',

              metrics=['accuracy'])
model.fit(X,y, epochs=10, batch_size=32, validation_split=0.1)
test.head()
x_test = test.drop(['id', 'delta'], axis=1)
x_test = np.array(x_test).reshape(-1,20,20,1)
# = normalize(x_test)
predictions = model.predict(x_test)
#np.argmax(predictions[0][340])

#int(round(predictions[1][0]))

predictions = predictions.round()

predictions = predictions.astype(int)

#predicted_val = [int(round(p)) for p in predictions]
column_names  = ['start.'+str(i) for i in range(1,401)]
submission = pd.DataFrame(data=predictions,    # values

              columns=column_names)
id_col = test[['id']]
submission['id'] = id_col
submission.head()
submission.to_csv("submission.csv", index = False)