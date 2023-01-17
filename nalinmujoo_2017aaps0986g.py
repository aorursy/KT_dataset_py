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
df = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv")
for columnname in df.columns:

    print(columnname, len(df[df[columnname] == "?"]))

df_old = df

for columnname in df.columns:

    df = df[df[columnname] != "?"]
cnameli = []

for columnname in df.columns:

    if columnname != "Class" and columnname != "ID":

        cnameli.append(columnname)

    
cnameli
df["Size"].replace("Small", 1, inplace = True) 

df["Size"].replace("Medium", 2, inplace = True) 

df["Size"].replace("Big", 3, inplace = True) 

df["Number of Quantities"] = df["Number of Quantities"].astype(int)

df["Number of Insignificant Quantities"] = df["Number of Insignificant Quantities"].astype(int)

df["Total Number of Words"] = df["Total Number of Words"].astype(int)

df["Number of Special Characters"] = df["Number of Special Characters"].astype(int)

df["Difficulty"] = df["Difficulty"].astype(float)
df.info()
df.head()
from keras.layers import Dense, Dropout, BatchNormalization

from keras.models import Sequential

from keras.models import load_model



from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler





X = df[cnameli][:].values

#X = StandardScaler().fit_transform(X)

y = df["Class"][:].values



y_new = to_categorical(y)

print(y_new)
X.shape, y.shape
X[0]
model = Sequential()

model.add(Dense(20,input_dim=11, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(20, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X, y_new, validation_split=0.2, epochs=600)
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# model=load_model("Model2")
#model.save("Model5.h5")
dft = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv")
dft["Size"].replace("Small", 1, inplace = True) 

dft["Size"].replace("Medium", 2, inplace = True) 

dft["Size"].replace("Big", 3, inplace = True) 
dft.info()
cnameli = []

for columnname in dft.columns:

    if columnname != "Class" and columnname != "ID":

        cnameli.append(columnname)

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



Xt = dft[cnameli][:].values
y_pred = model.predict(Xt)
y_pred
res = []

for row in y_pred:

    res.append(np.argmax(row))

dft["Class"] = res
dft.head()
dft.to_csv("op5.csv", columns = ["ID", "Class"], index = False)
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

create_download_link(dft)