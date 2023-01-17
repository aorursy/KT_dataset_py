import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from keras.layers import Dense, Dropout

from keras.models import Sequential

from keras.datasets import boston_housing



from sklearn.metrics import mean_absolute_error
data = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',', encoding = "utf-8", index_col=0)

data.head()

data.info()
data = data.replace({'?':np.nan}).dropna()
data = data.drop_duplicates()

data=pd.get_dummies(data=data,columns=['Size'])
#dict = {'Medium' : '2', 'Big' : '3', 'Small' : '1'}

#data = data.replace({"Size" : dict})

data
data = data.astype({"Class" : int})
X = data.drop("Class", axis = 1)
from keras.utils import to_categorical

y = data['Class']
y = to_categorical(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
print(y_train)

print(y_train.shape)

#print(y_train.shape)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
x_train.shape[1]
model = Sequential()

model.add(Dense(128, input_dim=13, activation='tanh'))

model.add(Dropout(rate=0.2))

model.add(Dense(64, activation='tanh'))

model.add(Dropout(rate=0.3))

model.add(Dense(32, activation='tanh'))

model.add(Dropout(rate=0.3))

model.add(Dense(16, activation='tanh'))

model.add(Dropout(rate=0.3))

model.add(Dense(8, activation='tanh'))

model.add(Dropout(rate=0.3))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint



callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
#history=model.fit(x_train, y_train, validation_split=0.2, epochs=500, batch_size=32,callbacks=callbacks,verbose=1)
history = model.fit(x_train, y_train, validation_split=.2, epochs=500, batch_size=100)
model.summary()
import matplotlib.pyplot as plt

plt.figure(figsize=(15,15))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
df_test=pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv', encoding = "utf-8", index_col=0)
df_test=pd.get_dummies(data=df_test,columns=['Size'])
df_test.head()

df_test
pred=model.predict_classes(df_test)
df_test['Class']=pred
df=df_test['Class'].reset_index()
#df.to_csv("~/Downloads/sub.csv",index=False)
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