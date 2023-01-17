import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Conv2D , MaxPooling2D , Dense, Flatten , Input , Dropout

from keras.models import Sequential

from keras import utils as np_utils

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
# df = pd.read_csv("train.csv", na_values = "?").dropna()

df = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',', na_values = "?").dropna()

y_train = df['Class']

df = df.drop(['ID', 'Class'], axis = 1)

df['Size'] = np.where(df['Size'] == "Medium", 1, 

             np.where(df['Size'] == "Small", 0, 

             np.where(df['Size'] == "Big", 2, 

                      df['Size'])))

# X_train = pd.DataFrame(MinMaxScaler().fit_transform(df.values)).fillna(-9999)

X_train = df#.fillna(-9999)

X_train.describe()
df1 = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=',', na_values = "?")

test_id = df1['ID']

df1 = df1.drop(['ID'], axis = 1)

df1['Size'] = np.where(df1['Size'] == "Medium", 1, 

              np.where(df1['Size'] == "Small", 0, 

              np.where(df1['Size'] == "Big", 2, 

                       df1['Size'])))

x_test = df1

x_test.describe()
Y = np_utils.to_categorical(np.asarray(y_train))

X = np.reshape(np.asarray(X_train), (X_train.shape[0],11,1))
model = Sequential()

model.add(Dense(128, input_shape=(11,1), activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Flatten())

model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, batch_size = 64, epochs = 800, validation_split = 0.20)

print(model.summary())
test_res = np.argmax(model.predict(np.reshape(np.asarray(x_test), (x_test.shape[0],11,1))), axis=1)

test_res
df = pd.DataFrame(data = {'ID': test_id, 'Class': test_res})#.to_csv('output.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df)