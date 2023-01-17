import pandas as pd
import numpy as np

from keras.models import Sequential  
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
import matplotlib.pylab as plt
from keras import metrics
from keras import backend as k

data = pd.read_csv('../input/data.csv', parse_dates=['InvoiceDate'])
data
data.describe()
prep = data.drop(['InvoiceNo', 'StockCode', 'Description', 'Country'], axis=1)
prep = prep.drop(prep[prep['Quantity'] <= 0].index)
prep = prep.drop(prep[prep['UnitPrice'] <= 0].index)
prep['Date'] = pd.DatetimeIndex(prep['InvoiceDate']).normalize()
prep['Amount'] = prep['Quantity'] * prep['UnitPrice']
prep = prep.drop(['Quantity', 'UnitPrice', 'InvoiceDate'], axis=1)
prep = prep.groupby(['CustomerID', 'Date'], as_index=False).sum().pivot('CustomerID', 'Date').fillna(0).transpose()
prep[prep > 0] = 1
prep
def _load_data(data, n_prev = 4):  
    docX, docY = [], []
    for i in range(len(data) - n_prev):
        docX.append(data.iloc[i:i + n_prev].as_matrix())
        docY.append(data.iloc[i + n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY
def train_test_split(data, test_size=0.1):  

    ntrn = round(len(data) * (1 - test_size))

    X_train, y_train = _load_data(data.iloc[0:ntrn])
    X_test, y_test = _load_data(data.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)
def precision(y_true, y_pred):
    true_positives = k.sum(k.round(y_true * y_pred))
    predicted_positives = k.sum(k.round(y_pred))
    return true_positives / (predicted_positives + k.epsilon())


def recall(y_true, y_pred):
    true_positives = k.sum(k.round(y_true * y_pred))
    possible_positives = k.sum(k.round(y_true))
    return true_positives / (possible_positives + k.epsilon())


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + k.epsilon())
in_out_neurons = prep.shape[1] 
hidden_neurons = 300

# model = Sequential()
# model.add(LSTM(hidden_neurons, return_sequences=False,
#                input_shape=(None, in_out_neurons)))
# model.add(Dense(in_out_neurons, input_dim=hidden_neurons))  
# model.add(Activation("linear"))  
# model.compile(loss="binary_crossentropy", optimizer="rmsprop",  metrics=[metrics.binary_accuracy, precision, recall, f1])
# model.summary()

model = Sequential()  
model.add(LSTM(hidden_neurons, return_sequences=True, input_shape=(None, in_out_neurons)))  
model.add(LSTM(hidden_neurons, return_sequences=True))  
model.add(Dropout(0.2))  
model.add(LSTM(hidden_neurons, return_sequences=False))  
model.add(Dropout(0.2))  
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))  
model.add(Activation("sigmoid"))  
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=[metrics.binary_accuracy, precision, recall, f1])  

(X_train, y_train), (X_test, y_test) = train_test_split(prep)  

history = model.fit(X_train, y_train, batch_size=1, epochs=1)
print(model.evaluate(X_test, y_test))
print(model.metrics_names)
predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print('predicted')
print(predicted)
print('rmse')
print(rmse)
print(history.history.keys())
plt.plot(history.history['binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
plt.rcParams["figure.figsize"] = (13, 9)
plt.plot(predicted[:100][:,0],"--")
plt.plot(predicted[:100][:,1],"--")
plt.plot(y_train[:100][:,0],":")
plt.plot(y_train[:100][:,1],":")
plt.legend(["Prediction 0", "Prediction 1", "Test 0", "Test 1"]) 