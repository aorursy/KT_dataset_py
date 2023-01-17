import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from keras.layers import Dense, Dropout

from keras.models import Sequential

from sklearn.metrics import mean_absolute_error

import keras
df = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", index_col=False)

x_test = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", index_col=False)
df["Class"].unique()
a = x_test["ID"]

x_test= x_test.set_index("ID")

x_test.head()
print(df.columns)

print(len(df))
df.head()
df = df.drop(["ID"], axis  = 1)
df.head()
for col in df.columns:

    df = df[df[col] != "?"]



print(len(df))





# df = df[~(df["Size"] == "?")]

# df = df[~(df["Number of Insignificant Quantities"] == "?")]
df.head()
df["Size"].unique()
# encoder = LabelEncoder()

# encoder.fit(df["Size"])

# encoded_Y = encoder.transform(df["Size"])

# # convert integers to dummy variables (i.e. one hot encoded)

# dummy_y = np_utils.to_categorical(encoded_Y)
df['Size'] = df['Size'].map({

    'Small': 0, 

    'Medium': 1,

    'Big' : 2

    })

x_test['Size'] = x_test['Size'].map({

    'Small': 0, 

    'Medium': 1,

    'Big' : 2

    })
df = df.reset_index(drop = True)

# df.head()

y_train = df["Class"]

df = df.drop(["Class"], axis = 1)

x_train = df

df.head()
def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(20, input_dim=11, activation='relu'))

    model.add(Dense(6, activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
x_train.shape[1]
# model = Sequential()

model = Sequential()

model.add(Dense(15,input_dim=11, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(20, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(18, activation='relu'))

model.add(Dropout(rate=0.3))

model.add(Dense(15, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(6,activation='softmax'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train = keras.utils.to_categorical(y_train, num_classes=6, dtype='int64')
model.fit(x_train, y_train,validation_split=0.2, epochs=100)

# model.fit(x_train, y_train)

y_test = model.predict(x_test)
preds = []

for i in y_test:

    mx = 0

    ind  = 0

    for j in range(0,6):

        if(mx <= i[j]):

            ind = j

            mx = i[j]

    preds.append(ind)

    
out = pd.DataFrame()
out["ID"] = a
out["Class"] = preds

out = out.set_index("ID")
out.to_csv("out.csv")
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

create_download_link(out)