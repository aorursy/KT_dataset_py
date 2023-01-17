import os

import gc

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm



import seaborn as sns

from collections import Counter

import matplotlib.pyplot as plt

from IPython.display import SVG



import warnings

warnings.filterwarnings('ignore')



import keras

from keras.models import Model

from keras.utils.vis_utils import model_to_dot

from keras.layers import Input, Dense, Dropout, BatchNormalization

import tensorflow as tf



from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)



from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

import plotly.graph_objects as go
DATA_PATH = '../input/ieee-fraud-detection/'

TRAIN_PATH = DATA_PATH + 'train_transaction.csv'

train_df = pd.read_csv(TRAIN_PATH)

train_df = train_df.fillna(0.0)
cat_cols = ['DeviceType', 'DeviceInfo', 'ProductCD', 

            'card4', 'card6', 'M4', 'P_emaildomain',

            'R_emaildomain','card1', 'card2', 'card3', 'card5', 'addr1',

            'addr2', 'M1', 'M2', 'M3', 'M5', 'M6','M7', 'M8', 'M9', 

            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3',

            'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']



cat_cols = [col for col in cat_cols if col in train_df.columns]
def prepare_data(df, cat_cols=cat_cols):

    cat_cols = [col for col in cat_cols if col in df.columns]

    for col in tqdm(cat_cols):\

        df[col] = pd.factorize(df[col])[0]

    return df



train_data = prepare_data(train_df)
X = train_data.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)

y = train_data.sort_values('TransactionDT')['isFraud']

del train_data
from IPython.display import HTML

HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/mfzHchd5La8?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
def get_neural_network():

    inputs = Input(shape=(X.shape[1],))

    dense_1 = Dense(10, activation='relu')(inputs)

    dense_2 = Dense(10, activation='relu')(dense_1)

    outputs = Dense(1, activation='sigmoid')(dense_2)

    

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model



model = get_neural_network()
SVG(model_to_dot(model).create(prog='dot', format='svg'))
split = np.int32(0.8 * len(X))



X_train = X[:split]

y_train = np.int32(y)[:split]



X_val = X[split:]

y_val = np.int32(y)[split:]
history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128)
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=history.history['acc'],

    name='train', mode='lines+markers',

    marker_color='crimson'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=history.history['val_acc'],

    name='val', mode='lines+markers',

    marker_color=' indigo'

))



# Set options common to all traces with fig.update_traces

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(title='Accuracy over the epochs',

                  yaxis_zeroline=False, xaxis_zeroline=False)

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), xaxis=dict(title="Epochs"))





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=history.history['acc']*100, marker={'color' : 'crimson'}),

    go.Bar(name='val', x=labels, y=history.history['val_acc']*100, marker={'color' : 'indigo'})

])

# Change the bar mode

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=history.history['loss'],

    name='train', mode='lines+markers',

    marker_color='crimson'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=history.history['val_loss'],

    name='val', mode='lines+markers',

    marker_color=' indigo'

))



# Set options common to all traces with fig.update_traces

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(title='Styled Scatter',

                  yaxis_zeroline=False, xaxis_zeroline=False)

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), xaxis=dict(title="Epochs"))





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=history.history['loss'], marker={'color' : 'crimson'}),

    go.Bar(name='val', x=labels, y=history.history['val_loss'], marker={'color' : 'indigo'})

])

# Change the bar mode

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
from IPython.display import YouTubeVideo

YouTubeVideo("VC8Jc9_lNoY", start=60*9, width=700, height=400)
def get_model_one():

    inputs = Input(shape=(X.shape[1],))

    dense_1 = Dense(100, activation='relu')(inputs)

    dense_2 = Dense(100, activation='relu')(dense_1)

    outputs = Dense(1, activation='sigmoid')(dense_2)

    

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model



model_one = get_model_one()
SVG(model_to_dot(model_one).create(prog='dot', format='svg'))
history = model_one.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128)
train_loss_1 = history.history['loss']

val_loss_1 = history.history['val_loss']



train_acc_1 = history.history['acc']

val_acc_1 = history.history['val_acc']
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_acc_1,

    name='train', mode='lines+markers',

    marker_color='blue'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_acc_1,

    name='val', mode='lines+markers',

    marker_color=' fuchsia'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_acc_1)*100, marker={'color' : 'blue'}),

    go.Bar(name='val', x=labels, y=np.array(val_acc_1)*100, marker={'color' : 'fuchsia'})

])

# Change the bar mode

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_loss_1,

    name='train', mode='lines+markers',

    marker_color='blue'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_loss_1,

    name='val', mode='lines+markers',

    marker_color=' fuchsia'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_loss_1), marker={'color' : 'blue'}),

    go.Bar(name='val', x=labels, y=np.array(val_loss_1), marker={'color' : 'fuchsia'})

])

# Change the bar mode

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
def get_model_two():

    inputs = Input(shape=(X.shape[1],))

    dense_1 = Dense(10, activation='relu')(inputs)

    dense_2 = Dense(10, activation='relu')(dense_1)

    outputs = Dense(1, activation='sigmoid')(dense_2)

    

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model



model_two = get_model_two()
SVG(model_to_dot(model_two).create(prog='dot', format='svg'))
history = model_two.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128)
train_loss_2 = history.history['loss']

val_loss_2 = history.history['val_loss']



train_acc_2 = history.history['acc']

val_acc_2 = history.history['val_acc']
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_acc_2,

    name='train', mode='lines+markers',

    marker_color='green'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_acc_2,

    name='val', mode='lines+markers',

    marker_color='red'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_acc_2)*100, marker={'color' : 'green'}),

    go.Bar(name='val', x=labels, y=np.array(val_acc_2)*100, marker={'color' : 'red'})

])

# Change the bar mode

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_loss_2,

    name='train', mode='lines+markers',

    marker_color='green'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_loss_2,

    name='val', mode='lines+markers',

    marker_color='red'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_loss_2), marker={'color' : 'green'}),

    go.Bar(name='val', x=labels, y=np.array(val_loss_2), marker={'color' : 'red'})

])

# Change the bar mode

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
preds_one_val = model_one.predict(X_val)

preds_two_val = model_two.predict(X_val)



preds_one_train = model_one.predict(X_train)

preds_two_train = model_two.predict(X_train)
import plotly.graph_objects as go

labels=['100-Neuron Model', '10-Neuron Model']



fig = go.Figure(data=[

    go.Bar(name='val', x=labels, y=[accuracy_score(y_val, np.round(preds_one_val))*100, accuracy_score(y_val, np.round(preds_two_val))*100]),

    go.Bar(name='train', x=labels, y=[accuracy_score(y_train, np.round(preds_one_train))*100, accuracy_score(y_train, np.round(preds_two_train))*100])

])

# Change the bar mode

fig.update_layout(barmode='group', yaxis_title="Accuracy", yaxis_type="log")

fig.show()
HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/Un9zObFjBH0?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
def get_model_one():

    inputs = Input(shape=(X.shape[1],))

    dense_1 = Dense(20, activation='relu')(inputs)

    dense_2 = Dense(20, activation='relu')(dense_1)

    outputs = Dense(1, activation='sigmoid')(dense_2)

    

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model



model_one = get_model_one()
SVG(model_to_dot(model_one).create(prog='dot', format='svg'))
history = model_one.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128)
train_loss_1 = history.history['loss']

val_loss_1 = history.history['val_loss']



train_acc_1 = history.history['acc']

val_acc_1 = history.history['val_acc']
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_acc_1,

    name='train', mode='lines+markers',

    marker_color='orangered'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_acc_1,

    name='val', mode='lines+markers',

    marker_color='navy'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_acc_1)*100, marker={'color' : 'orangered'}),

    go.Bar(name='val', x=labels, y=np.array(val_acc_1)*100, marker={'color' : 'navy'})

])

# Change the bar mode

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_loss_1,

    name='train', mode='lines+markers',

    marker_color='orangered'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_loss_1,

    name='val', mode='lines+markers',

    marker_color=' navy'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_loss_1), marker={'color' : 'orangered'}),

    go.Bar(name='val', x=labels, y=np.array(val_loss_1), marker={'color' : 'navy'})

])

# Change the bar mode

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
def get_model_two():

    inputs = Input(shape=(X.shape[1],))

    dense_1 = Dense(25, activation='relu')(inputs)

    dense_2 = Dense(25, activation='relu')(dense_1)

    outputs = Dense(1, activation='sigmoid')(dense_2)

    

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model



model_two = get_model_two()
SVG(model_to_dot(model_two).create(prog='dot', format='svg'))
history = model_two.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128)
train_loss_2 = history.history['loss']

val_loss_2 = history.history['val_loss']



train_acc_2 = history.history['acc']

val_acc_2 = history.history['val_acc']
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_acc_2,

    name='train', mode='lines+markers',

    marker_color='darkmagenta'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_acc_2,

    name='val', mode='lines+markers',

    marker_color='limegreen'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(titlefont=dict(size=18), yaxis_zeroline=False, xaxis_zeroline=False)



fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_acc_2)*100, marker={'color' : 'purple'}),

    go.Bar(name='val', x=labels, y=np.array(val_acc_2)*100, marker={'color' : 'limegreen'})

])

# Change the bar mode

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_loss_2,

    name='train', mode='lines+markers',

    marker_color='darkmagenta'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_loss_2,

    name='val', mode='lines+markers',

    marker_color='limegreen'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_loss_2), marker={'color' : 'darkmagenta'}),

    go.Bar(name='val', x=labels, y=np.array(val_loss_2), marker={'color' : 'limegreen'})

])

# Change the bar mode

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
preds_one = model_one.predict(X_val)

preds_two = model_two.predict(X_val)
ensemble = 0.75 * preds_one + 0.25 * preds_two
print("Accuracy of the first model : " + str(accuracy_score(y_val, np.round(preds_one))*100) + " %")

print("Accuracy of the second model : " + str(accuracy_score(y_val, np.round(preds_two))*100) + " %")

print("Accuracy of the average ensemble : " + str(accuracy_score(y_val, np.round(ensemble))*100) + " %")
acc_1 = accuracy_score(y_val, np.round(preds_one))*100

acc_2 = accuracy_score(y_val, np.round(preds_two))*100

acc_ensemble = accuracy_score(y_val, np.round(ensemble))*100
labels=['Model 1', 'Model 2', 'Ensemble']



fig = go.Figure(data=[go.Bar(x=labels, y=[acc_1 - 96.9, acc_2 - 96.9, acc_ensemble - 96.9], marker={'color' : 'crimson'})])

# Change the bar mode

fig.update_layout(title="Accuracy for different models (above 96.9)", yaxis=dict(title="Accuracy (above 96.9)"))

fig.show()
HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/u73PU6Qwl1I?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
def get_model_one():

    inputs = Input(shape=(X.shape[1],))

    dense_1 = Dense(200, activation='relu')(inputs)

    dense_2 = Dense(200, activation='relu')(dense_1)

    outputs = Dense(1, activation='sigmoid')(dense_2)

    

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model



model_one = get_model_one()
SVG(model_to_dot(model_one).create(prog='dot', format='svg'))
history = model_one.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128)
train_loss_1 = history.history['loss']

val_loss_1 = history.history['val_loss']



train_acc_1 = history.history['acc']

val_acc_1 = history.history['val_acc']
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_acc_1,

    name='train', mode='lines+markers',

    marker_color='orangered'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_acc_1,

    name='val', mode='lines+markers',

    marker_color='navy'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_acc_1)*100, marker={'color' : 'orangered'}),

    go.Bar(name='val', x=labels, y=np.array(val_acc_1)*100, marker={'color' : 'navy'})

])

# Change the bar mode

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_loss_1,

    name='train', mode='lines+markers',

    marker_color='orangered'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_loss_1,

    name='val', mode='lines+markers',

    marker_color='navy'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_loss_1), marker={'color' : 'orangered'}),

    go.Bar(name='val', x=labels, y=np.array(val_loss_1), marker={'color' : 'navy'})

])

# Change the bar mode

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
def get_model_two():

    inputs = Input(shape=(X.shape[1],))

    dense_1 = Dense(200, activation='relu')(inputs)

    dense_1 = Dropout(0.2)(dense_1)

    dense_2 = Dense(200, activation='relu')(dense_1)

    dense_2 = Dropout(0.2)(dense_2)

    outputs = Dense(1, activation='sigmoid')(dense_2)

    

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model



model_two = get_model_two()
SVG(model_to_dot(model_two).create(prog='dot', format='svg'))
history = model_two.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128)
train_loss_2 = history.history['loss']

val_loss_2 = history.history['val_loss']



train_acc_2 = history.history['acc']

val_acc_2 = history.history['val_acc']
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_acc_2,

    name='train', mode='lines+markers',

    marker_color='brown'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_acc_2,

    name='val', mode='lines+markers',

    marker_color='purple'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(titlefont=dict(size=18), yaxis_zeroline=False, xaxis_zeroline=False)



fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_acc_2)*100, marker={'color' : 'brown'}),

    go.Bar(name='val', x=labels, y=np.array(val_acc_2)*100, marker={'color' : 'purple'})

])

# Change the bar mode

fig.update_layout(title="Accuracy over the epochs", yaxis=dict(title="Accuracy"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=train_loss_2,

    name='train', mode='lines+markers',

    marker_color='brown'

))



fig.add_trace(go.Scatter(

    x=[1, 2, 3, 4, 5], y=val_loss_2,

    name='val', mode='lines+markers',

    marker_color='purple'

))



# Set options common to all traces with fig.update_traces

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), xaxis=dict(title="Epochs"))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)

fig.update_layout(yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
labels=['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']



fig = go.Figure(data=[

    go.Bar(name='train', x=labels, y=np.array(train_loss_2), marker={'color' : 'brown'}),

    go.Bar(name='val', x=labels, y=np.array(val_loss_2), marker={'color' : 'purple'})

])

# Change the bar mode

fig.update_layout(title="Loss over the epochs", yaxis=dict(title="Loss"), yaxis_type="log")

fig.update_layout(barmode='group')

fig.show()
preds_one_val = model_one.predict(X_val)

preds_two_val = model_two.predict(X_val)



preds_one_train = model_one.predict(X_train)

preds_two_train = model_two.predict(X_train)
import plotly.graph_objects as go

labels=['Without Dropout', 'With Dropout']



fig = go.Figure(data=[

    go.Bar(name='val', x=labels, y=[accuracy_score(y_val, np.round(preds_one_val))*100, accuracy_score(y_val, np.round(preds_two_val))*100]),

    go.Bar(name='train', x=labels, y=[accuracy_score(y_train, np.round(preds_one_train))*100, accuracy_score(y_train, np.round(preds_two_train))*100])

])

# Change the bar mode

fig.update_layout(barmode='group', yaxis_title="Accuracy", yaxis_type="log")

fig.show()