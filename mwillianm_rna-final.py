import numpy as np

import pandas as pd

import seaborn as sns
df = pd.read_csv('../input/Iris.csv', index_col='Id')

df.info(memory_usage=True)
df.describe(include='all')
types = {'SepalLengthCm': 'float16',

         'SepalWidthCm': 'float16',

         'PetalLengthCm': 'float16',

         'PetalWidthCm': 'float16',

         'Species': 'category'

        }

df = df.astype(types)

df.info()
sns.heatmap(df.corr(),cmap='coolwarm', annot=True, vmin=-1, vmax=1, linewidths=.5)
df['SepalArea'] = df.SepalLengthCm * df.SepalWidthCm

df['PetalArea'] = df.PetalLengthCm * df.PetalWidthCm

df['SepalRatio'] = df.SepalLengthCm / df.SepalWidthCm

df['PetalRatio'] = df.PetalLengthCm / df.PetalWidthCm

df['RatioArea'] = df.SepalArea / df.PetalArea

df.info()
label = 'Species'

df.describe(include='all')
sns.pairplot(df, hue=label, diag_kind='hist')
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from sklearn.model_selection import train_test_split
x = df.drop(label, axis=1)

y = df[label]
def DataEncoder(df):

    df['SepalRatio'] = df.SepalLengthCm / df.SepalWidthCm

    df['PetalRatio'] = df.PetalLengthCm / df.PetalWidthCm

    df['SepalArea'] = df.SepalLengthCm * df.SepalWidthCm

    df['PetalArea'] = df.PetalLengthCm * df.PetalWidthCm

    df['RatioArea'] = df.SepalArea / df.PetalArea

    return df



le = LabelBinarizer()

scaler = StandardScaler()



x = scaler.fit_transform(DataEncoder(x))

y = le.fit_transform(y)

x.shape, y.shape
x[:5], y[:5]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
from keras import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

from matplotlib.pyplot import *
model = Sequential()

model.add(Dense(18, activation='tanh', input_dim=9))

model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.1)

model.compile(optimizer=sgd,

              loss='categorical_crossentropy',

              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs = 300)

fig, ax = subplots()

df_resultados = pd.DataFrame.from_dict(history.history)

df_resultados['acc'].plot()

df_resultados['loss'].plot()

ax.legend()
from sklearn.metrics import classification_report, confusion_matrix

score, acc = model.evaluate(x_test, y_test)

pred = model.predict(x_test)

pred = le.inverse_transform(np.rint(pred))

y_test = le.inverse_transform(y_test) if y_test.ndim == 2 else y_test

print(classification_report(y_test, pred))

print(confusion_matrix(y_test, pred))

print('Test score:', score)

print('Test accuracy:', acc)