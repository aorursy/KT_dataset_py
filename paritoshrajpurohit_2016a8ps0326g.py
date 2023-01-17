import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',')

dft = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv",encoding="utf-8",header=0)
df.info()
df = df.drop(['ID'], axis = 1)

#dft = dft.drop(['ID'], axis = 1)
df.drop_duplicates(inplace = True)

dft.drop_duplicates(inplace = True)
df.info()
df = df.replace({"?":np.nan})

df = df.dropna()

dft = dft.replace({"?":np.nan})

dft = dft.dropna()
df.info()
df['Number of Quantities'] = pd.to_numeric(df['Number of Quantities'])

df['Number of Insignificant Quantities'] = pd.to_numeric(df['Number of Quantities'])

df['Total Number of Words'] = pd.to_numeric(df['Total Number of Words'])

df['Number of Special Characters'] = pd.to_numeric(df['Number of Special Characters'])

df['Difficulty'] = pd.to_numeric(df['Difficulty'])

# test

dft['Number of Quantities'] = pd.to_numeric(dft['Number of Quantities'])

dft['Number of Insignificant Quantities'] = pd.to_numeric(dft['Number of Quantities'])

dft['Total Number of Words'] = pd.to_numeric(dft['Total Number of Words'])

dft['Number of Special Characters'] = pd.to_numeric(dft['Number of Special Characters'])

dft['Difficulty'] = pd.to_numeric(dft['Difficulty'])
df.info()

dft.info()
df = pd.get_dummies(df, columns=['Size'])

df.head()

new_f=df

dft = pd.get_dummies(dft, columns=['Size'])

new_f1=dft

dft.head()
df = pd.get_dummies(df, columns=['Class'])

df.head()


nnew_df = df

nnew_df = nnew_df.drop(['Number of Quantities'], axis = 1)

nnew_df = nnew_df.drop(['Number of Insignificant Quantities'], axis = 1)

nnew_df = nnew_df.drop(['Total Number of Words'], axis = 1)

nnew_df = nnew_df.drop(['Total Number of Characters'], axis = 1)

nnew_df = nnew_df.drop(['Number of Special Characters'], axis = 1)

nnew_df = nnew_df.drop(['Number of Sentences'], axis = 1)

nnew_df = nnew_df.drop(['First Index'], axis = 1)

nnew_df = nnew_df.drop(['Second Index'], axis = 1)

nnew_df = nnew_df.drop(['Difficulty'], axis = 1)

nnew_df = nnew_df.drop(['Score'], axis = 1)

nnew_df = nnew_df.drop(['Size_Big'], axis = 1)

nnew_df = nnew_df.drop(['Size_Medium'], axis = 1)

#nnew_df = nnew_df.drop(['Size_Small'], axis = 1)

nnew_df.head()
y = nnew_df.drop(['Size_Small'], axis = 1)

X = new_f.drop(['Class'], axis = 1)

fin = new_f1.drop(['ID'], axis = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size = 0.2, random_state = 13)

#print(y)
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers,activations

from sklearn.metrics import mean_absolute_error





# Build the architecture

#import tensorflow as tf

#lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

model = Sequential()

model.add(Dense(128,input_dim=13, activation='tanh'))

model.add(Dropout(rate=0.1))



model.add(Dense(128, activation='tanh'))

model.add(Dropout(rate=0.1))



model.add(Dense(128, activation='tanh'))

model.add(Dropout(rate=0.1))



model.add(Dense(128, activation='tanh'))

model.add(Dropout(rate=0.1))



model.add(Dense(128, activation='tanh'))

model.add(Dropout(rate=0.1))



model.add(Dense(128, activation='tanh'))

model.add(Dropout(rate=0.1))



model.add(Dense(128, activation='tanh'))

model.add(Dropout(rate=0.1))



model.add(Dense(128, activation='tanh'))

model.add(Dropout(rate=0.1))



model.add(Dense(128, activation='tanh'))

model.add(Dropout(rate=0.1))



model.add(Dense(128, activation='tanh'))

model.add(Dropout(rate=0.1))



model.add(Dense(128, activation='tanh'))

model.add(Dropout(rate=0.1))





model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(X,y, validation_split=0.1, epochs=650,batch_size=16)
model.summary()
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
pred = model.predict([fin])

#print(fin)

ans=[]

for i in range(159):

  ans.append(np.argmax(pred[i]))
dfm = pd.read_csv('/kaggle/input/bitsf312-lab1/sample_submission.csv')

dfm["Class"]= ans

dfm.to_csv("data.csv",index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(dfm, title = "Download CSV file", filename = "data.csv"):

    csv = dfm.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(dfm)