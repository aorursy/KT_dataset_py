#Data

import pandas as pd

import numpy as np

from scipy import stats

import sys



#Plotting

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import colorlover as cl

from IPython.display import HTML, SVG

import random



random.seed(42)

init_notebook_mode(connected=True)

#%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_comp = pd.read_csv('../input/test.csv')
iplot(ff.create_table(df_train.iloc[0:10,0:10]), filename='jupyter-table1')
n_size = 10

sel_int = 5

num_array = df_train[df_train.label == sel_int]



fig = tools.make_subplots(rows=10, cols=10, print_grid=False)

for row in range(1,n_size+1):

    for column in range(1,n_size+1):

        trace = go.Heatmap(z=num_array.iloc[row*10-10+column-1, 1:].values.reshape((28,28))[::-1], colorscale=[[0,'rgb(0,0,0)'],[1,'rgb(255,255,255)']], showscale=False)

        fig.append_trace(trace, row, column)

        fig['layout']['xaxis'+str(((row-1)*10 + column))].update(showticklabels=False, ticks='')

        fig['layout']['yaxis'+str(((row-1)*10 + column))].update(showticklabels=False, ticks='')

        

fig['layout'].update(height=500, width=500)

fig['layout']['margin'].update(l=10, r=10, b=10, t=10)

iplot(fig, filename='number_plot')
#df.label.value_counts().values

trace = go.Bar(x=df_train.label.value_counts().index,y=df_train.label.value_counts().values)

layout = go.Layout(xaxis=dict(title='Number', nticks=10),

                  yaxis=dict(title='# Occurance'),

                  width = 600,

                  height = 400

                  )

figure = go.Figure(data = [trace],

                  layout = layout)

figure['layout']['margin'].update(l=50, r=50, b=50, t=50)

iplot(figure)
df_train.describe()
df_train.isnull().sum().sum()
from sklearn.model_selection import train_test_split
Y = df_train.label

X = df_train.drop('label', axis=1)



X = X / 255

X_comp = df_comp / 255



X_train, X_cross, Y_train, Y_cross = train_test_split(X, Y,test_size=0.1, random_state=42)

X_valid, X_test, Y_valid, Y_test = train_test_split(X_cross, Y_cross, test_size=0.5, random_state=42)
trace1 = go.Bar(x=Y_train.value_counts().index,y=Y_train.value_counts().values/Y_train.value_counts().values.sum(), name='Training set')

trace2 = go.Bar(x=Y_valid.value_counts().index,y=Y_valid.value_counts().values/Y_valid.value_counts().values.sum(), name='Validation set')

trace3 = go.Bar(x=Y_test.value_counts().index,y=Y_test.value_counts().values/Y_test.value_counts().values.sum(), name='Test set')

fig = go.Figure(data=[trace1, trace2, trace3])

fig['layout'].update(xaxis=dict(title='Number', nticks=10), yaxis=dict(title='# Occurance'), width = 600, height = 400)

fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

iplot(fig)
from keras.models import Sequential, load_model

from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

from keras.utils import plot_model, to_categorical

from keras.utils.vis_utils import model_to_dot

from keras.preprocessing.image import ImageDataGenerator



from sklearn.metrics import confusion_matrix, accuracy_score
X_train = X_train.values.reshape(X_train.shape[0],28,28,1)

X_valid = X_valid.values.reshape(X_valid.shape[0],28,28,1)

X_test = X_test.values.reshape(X_test.shape[0],28,28,1)

X_comp = X_comp.values.reshape(X_comp.shape[0],28,28,1)



Y_train = to_categorical(Y_train)

Y_valid = to_categorical(Y_valid)

Y_test = to_categorical(Y_test)
datagen = ImageDataGenerator(height_shift_range=0.1,

                             width_shift_range=0.1,

                             #brightness_range=(0,0.1),

                             rotation_range=10,

                             zoom_range=0.1,

                             fill_mode='constant',

                             cval=0

                            )



datagen.fit(X_train)
model = Sequential()

droprate = 0.175

model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(droprate))

model.add(Dense(10, activation='softmax'))



model1 = model

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochsN = 25

batch_sizeN = 63

history1 = model1.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)
model1.evaluate(X_test, Y_test, verbose=0)
model1.save('model_1.h5')
history = history1

fig = tools.make_subplots(rows=1, cols=2, print_grid=False)

trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')

trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')

trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')

trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)

fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)

fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))

fig['layout']['margin'].update(l=50, r=50, b=50, t=50)



#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)

#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

iplot(fig)
del model

model = Sequential()

droprate = 0.15

model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))

#model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

#model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))

#model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(droprate))

model.add(Dense(10, activation='softmax'))



model2 = model

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochsN = 35

batch_sizeN = 63

history2 = model2.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)
model2.evaluate(X_test, Y_test, verbose=0)
model2.save('model_2.h5')
history = history2

fig = tools.make_subplots(rows=1, cols=2, print_grid=False)

trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')

trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')

trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')

trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)

fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)

fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))

fig['layout']['margin'].update(l=50, r=50, b=50, t=50)



#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)

#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

iplot(fig)
del model

model = Sequential()

droprate = 0.2

model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(droprate))

model.add(Dense(10, activation='softmax'))



model3 = model

model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochsN = 40

batch_sizeN = 63

history3 = model3.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)
model3.evaluate(X_test, Y_test, verbose=0)
model3.save('model_3.h5')
history = history3

fig = tools.make_subplots(rows=1, cols=2, print_grid=False)

trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')

trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')

trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')

trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)

fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)

fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))

fig['layout']['margin'].update(l=50, r=50, b=50, t=50)



#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)

#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

iplot(fig)
del model

model = Sequential()

droprate = 0.20

model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=16, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=16, strides=(1,1), padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(droprate))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(droprate))

model.add(Dense(10, activation='softmax'))



model4 = model

model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochsN = 90

batch_sizeN = 63

history4 = model4.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)
model4.evaluate(X_test, Y_test, verbose=0)
model4.save('model_4.h5')
history = history4

fig = tools.make_subplots(rows=1, cols=2, print_grid=False)

trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')

trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')

trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')

trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)

fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)

fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))

fig['layout']['margin'].update(l=50, r=50, b=50, t=50)



#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)

#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

iplot(fig)
del model

model = Sequential()

droprate = 0.1

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(2,2), padding='valid',activation='relu'))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(2,2), padding='valid',activation='relu'))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=16, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(3,3), filters=16, strides=(2,2), padding='valid',activation='relu'))

model.add(Flatten())

model.add(Dropout(droprate))

model.add(Dense(128, activation='relu'))

model.add(Dropout(droprate))

model.add(Dense(10, activation='softmax'))



model5 = model

model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochsN = 90

batch_sizeN = 63

history5 = model5.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)
model5.evaluate(X_test, Y_test, verbose=0)
model5.save('model_5.h5')
history = history5

fig = tools.make_subplots(rows=1, cols=2, print_grid=False)

trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')

trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')

trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')

trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)

fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)

fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))

fig['layout']['margin'].update(l=50, r=50, b=50, t=50)



#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)

#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

iplot(fig)
del model

model = Sequential()

droprate = 0.15

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(2,2), padding='valid',activation='relu'))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=32, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=16, strides=(2,2), padding='valid',activation='relu'))

model.add(Flatten())

model.add(Dropout(droprate))

model.add(Dense(256, activation='relu'))

model.add(Dropout(droprate))

model.add(Dense(10, activation='softmax'))



model6 = model

model6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochsN = 45

batch_sizeN = 63

history6 = model6.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)
model6.evaluate(X_test, Y_test, verbose=0)
model6.save('model_6.h5')
history = history6

fig = tools.make_subplots(rows=1, cols=2, print_grid=False)

trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')

trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')

trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')

trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)

fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)

fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))

fig['layout']['margin'].update(l=50, r=50, b=50, t=50)



#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)

#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

iplot(fig)
del model

model = Sequential()

droprate = 0.35

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

#model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=64, strides=(2,2), padding='valid',activation='relu'))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(3,3), filters=128, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(3,3), filters=128, strides=(1,1), padding='same',activation='relu'))

#model.add(Conv2D(kernel_size=(3,3), filters=128, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(2,2), filters=128, strides=(2,2), padding='valid',activation='relu'))

model.add(Dropout(droprate))

model.add(Conv2D(kernel_size=(3,3), filters=256, strides=(1,1), padding='valid',activation='relu'))

model.add(Conv2D(kernel_size=(3,3), filters=256, strides=(1,1), padding='valid',activation='relu'))

#model.add(Conv2D(kernel_size=(3,3), filters=256, strides=(1,1), padding='same',activation='relu'))

model.add(Conv2D(kernel_size=(3,3), filters=256, strides=(2,2), padding='valid',activation='relu'))

model.add(Dropout(droprate))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))



model7 = model

model7.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochsN = 60

batch_sizeN = 63

history7 = model7.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_sizeN), validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train)/batch_sizeN, epochs=epochsN, verbose=2)
model7.evaluate(X_test, Y_test, verbose=0)
model7.save('model_7.h5')
history = history7

fig = tools.make_subplots(rows=1, cols=2, print_grid=False)

trace1 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['loss'], name='Training Loss')

trace2 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_loss'], name='Validation Loss')

trace3 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['acc'], name='Training Accuracy')

trace4 = go.Scatter(x=list(range(1,epochsN+1)), y=history.history['val_acc'], name='Validation Accuracy')

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)

fig['layout'].update(xaxis1=dict(title='Epoch', nticks=10), yaxis1=dict(title='Loss', type='log'), width = 1200, height = 400)

fig['layout'].update(xaxis2=dict(title='Epoch', nticks=10), yaxis2=dict(title='Accuracy', type = 'log'))

fig['layout']['margin'].update(l=50, r=50, b=50, t=50)



#fig['layout'].update(xaxis=dict(title='Epoch', nticks=10), yaxis=dict(title='Loss'), width = 600, height = 400)

#fig['layout']['margin'].update(l=50, r=50, b=50, t=50)

iplot(fig)
trained_models = [model1, model2, model3, model4, model5, model6, model7]
acc_scores = pd.Series()

for num, model in enumerate(trained_models):

    acc_scores.loc['Model ' + str(num + 1)] = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
trace = go.Bar(x=acc_scores.values ,y=acc_scores.index, orientation='h')

layout = go.Layout(xaxis=dict(title='Accuracy', nticks=10, range=[0.985, 1]),

                  #yaxis=dict(title='Model'),

                  width = 600,

                  height = 400

                  )

figure = go.Figure(data = [trace],

                  layout = layout)

figure['layout']['margin'].update(l=50, r=50, b=50, t=50)

iplot(figure)
print(acc_scores.idxmax(), ': ', acc_scores[acc_scores.idxmax()])
ind_best_model = acc_scores.reset_index().loc[:, 0].idxmax(axis=0)

Y_test_pred = trained_models[ind_best_model].predict(X_test)

confM = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_test_pred, axis=1))



fig = ff.create_annotated_heatmap(x=list(map(str, range(0,10))), y=list(map(str, range(0,10))), z=np.log(confM+1), annotation_text=confM, colorscale='Jet')

fig['layout'].update(xaxis=dict(title='Predictated Label'), yaxis=dict(title='Actual Label', autorange='reversed'), width = 600, height = 600)



iplot(fig)
def summing_classifier(data, model_list):

    total_pred_prob = model_list[0].predict(data)

    for model in model_list[1:]:

        total_pred_prob += model.predict(data)

        

    return np.argmax(total_pred_prob, axis=1)
acc_scores.loc['Summing Classifier'] = accuracy_score(np.argmax(Y_test, axis=1), summing_classifier(X_test, trained_models))

acc_scores.loc['Summing Classifier']
acc_scores.iloc[0:6].mean()
confM = confusion_matrix(np.argmax(Y_test, axis=1), summing_classifier(X_test, trained_models))



fig = ff.create_annotated_heatmap(x=list(map(str, range(0,10))), y=list(map(str, range(0,10))), z=np.log(confM+1), annotation_text=confM, colorscale='Jet')

fig['layout'].update(xaxis=dict(title='Predictated Label'), yaxis=dict(title='Actual Label', autorange='reversed'), width = 600, height = 600)



iplot(fig)
def voting_classifier(data, model_list):

    pred_list = np.argmax(model_list[0].predict(data), axis=1).reshape((1,len(data)))

    for model in model_list[1:]:

        pred_list = np.append(pred_list, [np.argmax(model.predict(data), axis=1)], axis=0)

    return np.array(list(map(lambda x: np.bincount(x).argmax(), pred_list.T)))
acc_scores.loc['Voting Classifier'] = accuracy_score(np.argmax(Y_test, axis=1), voting_classifier(X_test, trained_models))

acc_scores.loc['Voting Classifier']
confM = confusion_matrix(np.argmax(Y_test, axis=1), voting_classifier(X_test, trained_models))



fig = ff.create_annotated_heatmap(x=list(map(str, range(0,10))), y=list(map(str, range(0,10))), z=np.log(confM+1), annotation_text=confM, colorscale='Jet')

fig['layout'].update(xaxis=dict(title='Predictated Label'), yaxis=dict(title='Actual Label', autorange='reversed'), width = 600, height = 600)



iplot(fig)
trace = go.Bar(x=acc_scores.sort_values(ascending=True).values ,y=acc_scores.sort_values(ascending=True).index, orientation='h')

layout = go.Layout(xaxis=dict(title='Accuracy', nticks=10, range=[0.985, 1]),

                  #yaxis=dict(title='Model'),

                  width = 600,

                  height = 400

                  )

figure = go.Figure(data = [trace],

                  layout = layout)

figure['layout']['margin'].update(l=130, r=50, b=50, t=50)

iplot(figure)
best_model_results = pd.DataFrame({'Label' : np.argmax(trained_models[ind_best_model].predict(X_comp), axis=1)})

best_model_results = best_model_results.reset_index().rename(columns={'index' : 'ImageId'})

best_model_results['ImageId'] = best_model_results['ImageId'] + 1

best_model_results.to_csv('best_model_result_kaggle.csv', index=False)
esmbl_sum_results = pd.DataFrame({'Label' : summing_classifier(X_comp, trained_models)})

esmbl_sum_results = esmbl_sum_results.reset_index().rename(columns={'index' : 'ImageId'})

esmbl_sum_results['ImageId'] = esmbl_sum_results['ImageId'] + 1

esmbl_sum_results.to_csv('esmbl_sum_result_kaggle.csv', index=False)
esmbl_vote_results = pd.DataFrame({'Label' : voting_classifier(X_comp, trained_models)})

esmbl_vote_results = esmbl_vote_results.reset_index().rename(columns={'index' : 'ImageId'})

esmbl_vote_results['ImageId'] = esmbl_vote_results['ImageId'] + 1

esmbl_vote_results.to_csv('esmbl_vote_result_kaggle.csv', index=False)