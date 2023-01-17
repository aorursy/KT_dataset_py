!pip install nb_black -q
%load_ext nb_black
import numpy as np

import pandas as pd

import plotly.express as px
data = pd.read_csv("../input/tesla-stock-price/Tesla.csv - Tesla.csv.csv")

data.Date = pd.to_datetime(data.Date)

data.head()
fig = px.line(data, x="Date", y="Open", title="Tesla Stock Price - Open")

fig.update_xaxes(rangeslider_visible=True)

fig.show()
from sklearn.preprocessing import MinMaxScaler



sc = MinMaxScaler()

data[["Open_Scaled"]] = sc.fit_transform(data[["Open"]])
def create_dataset(dataset, look_back=60):

    dataX, dataY = [], []

    for i in range(len(dataset) - look_back - 1):

        a = dataset[i : (i + look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return pd.DataFrame(np.array(dataX)), np.array(dataY)





X, Y = create_dataset(data[["Open_Scaled"]].values)
X_train = np.reshape(X[:-100].values, (X[:-100].shape[0], X[:-100].shape[1], 1))

y_tain = Y[:-100]



X_test = np.reshape(X[-100:].values, (X[-100:].shape[0], X[-100:].shape[1], 1))

y_test = Y[-100:]
%%time 



import keras

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout



rnn = Sequential()



rnn.add(LSTM(units=360, return_sequences=True, input_shape=(X.shape[1], 1)))

rnn.add(Dropout(0.2))

rnn.add(LSTM(units=360, return_sequences=True))

rnn.add(Dropout(0.2))

rnn.add(LSTM(units=360, return_sequences=True))

rnn.add(Dropout(0.2))

rnn.add(LSTM(units=360))

rnn.add(Dropout(0.2))

rnn.add(Dense(units=1))



rnn.compile(optimizer="adam", loss="mean_squared_error")





callback = keras.callbacks.EarlyStopping(

    monitor="loss",

    min_delta=0,

    patience=0,

    verbose=0,

    mode="auto",

    baseline=None,

    restore_best_weights=False,

)



rnn.fit(X_train, y_tain, epochs=10, batch_size=32)
predicted_stock_price = rnn.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
import plotly.graph_objects as go

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score





# Create random data with numpy

x = data["Date"][-100:]

y = data["Open"][-100:]

z = predicted_stock_price.T[0]



# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Real"))

fig.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Predicted"))

fig.update_layout(

    title="Stock price - Tesla Moctors",

    xaxis_title="Date",

    yaxis_title="Stock price ($)",

)





print("r2_score :", r2_score(y, z))

print("mean_squared_error :", mean_squared_error(y, z))

fig.show()