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
import numpy as np

import pandas as pd



from keras.layers import Dense, Dropout

from keras.models import Sequential



from sklearn.metrics import mean_absolute_error

import pandas as pd

df = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv')

df_test = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
df = df.drop(['ID'], axis = 1)

df_t = df_test.drop(['ID'], axis = 1)

df.head()
df = df.replace({'?': np.nan})

null_columns = df.columns[df.isnull().any()]

null_columns



df.dropna(inplace=True)

df=df.drop_duplicates()

df.info()

df = df.astype({'Number of Quantities': 'int64'})

df = df.astype({'Number of Insignificant Quantities':'int64'})

df = df.astype({'Total Number of Words':'int64'})

df = df.astype({'Difficulty':'float64'})

df = df.astype({'Number of Special Characters':'int64'})

df.dtypes
df['Size'].replace({

    'Small':1,

    'Medium':2,

    'Big':3

    },inplace=True)

df.dtypes
df_t['Size'].replace({

    'Small':1,

    'Medium':2,

    'Big':3

    },inplace=True)

df_t.dtypes
Y2 = df['Class']

X2 = df.drop(['Class'], axis = 1)
x_train = X2

y_train = Y2
# Loading the dataset

x_train.shape, y_train.shape
# Build the architecture

model = Sequential()

model.add(Dense(20,input_dim=11, activation='relu'))

model.add(Dropout(rate=0.2))

#model.add(Dense(20, activation='relu'))

#model.add(Dropout(rate=0.2))

model.add(Dense(6,activation='softmax'))
# Compile 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
from keras.utils import to_categorical

y_t = to_categorical(y_train)

history = model.fit(x_train, y_t, validation_split=0.1, epochs=1000)
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
X_test = df_t.values

y_pred = model.predict(X_test)

array = np.arange(371,530)

y_pred
labels = np.argmax(y_pred, axis=-1)

print(labels)

dataset = pd.DataFrame({'ID': array[:], 'Class': labels[:]})

dataset['Class'].unique()

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

create_download_link(dataset)
