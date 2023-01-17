import matplotlib.pyplot as plt

import statsmodels.tsa.seasonal as smt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import datetime as dt

from sklearn import linear_model 

from sklearn.metrics import mean_absolute_error

import plotly



from keras.models import Sequential

from keras.layers import Activation, Dense

from keras.layers import LSTM

from keras.layers import Dropout



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import os

os.chdir('../input/Data/Stocks/')
# データを読んでみよう。

# カーネルを使用すると、zipファイルをディレクトリのように移動できます



# サイズ0のファイルを読み取ろうとするとエラーが出るため、それらをスキップします

# filenames = [x for x in os.listdir() if x.endswith('.txt') and os.path.getsize(x) > 0]

# filenames = random.sample(filenames,1)

filenames = ['prk.us.txt', 'bgr.us.txt', 'jci.us.txt', 'aa.us.txt', 'fr.us.txt', 'star.us.txt', 'sons.us.txt', 'ipl_d.us.txt', 'sna.us.txt', 'utg.us.txt']

filenames = [filenames[1]]

print(filenames)



data = []

for filename in filenames:

    df = pd.read_csv(filename, sep=',')



    label, _, _ = filename.split(sep='.')

    df['Label'] = filename

    df['Date'] = pd.to_datetime(df['Date'])

    data.append(df)
r = lambda: random.randint(0,255)

traces = []



for df in data:

    clr = str(r()) + str(r()) + str(r())

#     df = df.sample(n=100, replace=True)

    df = df.sort_values('Date')

#     print(df['Label'])

    label = df['Label'].iloc[0]



    trace = plotly.graph_objs.Scattergl(

        x=df['Date'],

        y=df['Close'],

        mode='line',

        line=dict(

            color = clr

        )

    )

    traces.append(trace)

    

layout = plotly.graph_objs.Layout(

    title='Plot',

)

fig = plotly.graph_objs.Figure(data=traces, layout=layout)



plotly.offline.init_notebook_mode(connected=True)

plotly.offline.iplot(fig, filename='dataplot')
df = data[0]

window_len = 10



#データポイント（日付とか）を作り、訓練データとテストデータを分割します。

split_date = list(data[0]["Date"][-(2*window_len+1):])[0]



#訓練データとテストデータを分割します。

training_set, test_set = df[df['Date'] < split_date], df[df['Date'] >= split_date]

training_set = training_set.drop(['Date','Label', 'OpenInt'], 1)

test_set = test_set.drop(['Date','Label','OpenInt'], 1)



#訓練データのウィンドウを作ります。

LSTM_training_inputs = []

for i in range(len(training_set)-window_len):

    temp_set = training_set[i:(i+window_len)].copy()

    

    for col in list(temp_set):

        temp_set[col] = temp_set[col]/temp_set[col].iloc[0] - 1

    

    LSTM_training_inputs.append(temp_set)

LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1



LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]

LSTM_training_inputs = np.array(LSTM_training_inputs)



#テストのためのウィンドウを作ります。

LSTM_test_inputs = []

for i in range(len(test_set)-window_len):

    temp_set = test_set[i:(i+window_len)].copy()

    

    for col in list(temp_set):

        temp_set[col] = temp_set[col]/temp_set[col].iloc[0] - 1

    

    LSTM_test_inputs.append(temp_set)

LSTM_test_outputs = (test_set['Close'][window_len:].values/test_set['Close'][:-window_len].values)-1



LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]

LSTM_test_inputs = np.array(LSTM_test_inputs)
def build_model(inputs, output_size, neurons, activ_func="linear",

                dropout=0.10, loss="mae", optimizer="adam"):

    

    model = Sequential()



    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))

    model.add(Dropout(dropout))

    model.add(Dense(units=output_size))

    model.add(Activation(activ_func))



    model.compile(loss=loss, optimizer=optimizer)

    return model
# モデル構造の初期化

nn_model = build_model(LSTM_training_inputs, output_size=1, neurons = 32)

# モデル出力は、10番前の終値に正規化された次の価格です

# データの中のモデルを訓練する

# ノート：eth_historyは各エポックの訓練エラーの情報を含む

nn_history = nn_model.fit(LSTM_training_inputs, LSTM_training_outputs, 

                            epochs=5, batch_size=1, verbose=2, shuffle=True)
plt.plot(LSTM_test_outputs, label = "actual")

plt.plot(nn_model.predict(LSTM_test_inputs), label = "predicted")

plt.legend()

plt.show()

MAE = mean_absolute_error(LSTM_test_outputs, nn_model.predict(LSTM_test_inputs))

print('The Mean Absolute Error is: {}'.format(MAE))
#https://github.com/llSourcell/How-to-Predict-Stock-Prices-Easily-Demo/blob/master/lstm.py

def predict_sequence_full(model, data, window_size):

    #ウィンドウを1個の新しい予測が出るたびに移動させ、再度予測を走らせます

    curr_frame = data[0]

    predicted = []

    for i in range(len(data)):

        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])

        curr_frame = curr_frame[1:]

        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)

    return predicted



predictions = predict_sequence_full(nn_model, LSTM_test_inputs, 10)



plt.plot(LSTM_test_outputs, label="actual")

plt.plot(predictions, label="predicted")

plt.legend()

plt.show()

MAE = mean_absolute_error(LSTM_test_outputs, predictions)

print('The Mean Absolute Error is: {}'.format(MAE))