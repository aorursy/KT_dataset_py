# COVID19 Local US-CA Forecasting (Week 1)



# Forecast daily COVID-19 spread in California, USA



import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

from sklearn.preprocessing import MinMaxScaler



import torch

import torch.nn as nn



style.use('ggplot')



for dirname, _, filename in os.walk(os.getcwd()):

    for file in filename:

        path = os.path.join(dirname, file)

        if 'csv' in path:

            print(path)



train_raw = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

test_raw = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

sample_submission = pd.read_csv(

    '../input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')



#### Data Cleaning



## 1. Drop Redundant Columns

train_drop_cols = train_raw.columns[:-3]

test_drop_cols = test_raw.columns[1:-1]



train = train_raw.copy().drop(train_drop_cols, axis=1)

test = test_raw.copy().drop(test_drop_cols, axis=1)



## 2. Reindex

train.index = pd.to_datetime(train['Date'])

train.drop(['Date'], axis=1, inplace=True)



test.index = pd.to_datetime(test['Date'])

test.drop(['Date'], axis=1, inplace=True)



## 3. Extract rows with confirmed cases greater 0

train = train[train['ConfirmedCases'] > 0]



## 4. Scale the data to values between 0 and 1

infections = train[['ConfirmedCases']]

fatality = train[['Fatalities']]



scaler_infections = MinMaxScaler()



scaler_infections = scaler_infections.fit(infections)



train_data_infections = scaler_infections.transform(infections)



scaler_fatalities = MinMaxScaler()



scaler_fatalities = scaler_fatalities.fit(fatality)



train_data_fatalities = scaler_fatalities.transform(fatality)





## 6. Break the large sequence into chunks of smaller sequences



def chunk_of_sequences(seq, chunk_length):

    """

    Slice the input sequence into chunks of smaller sequences of equal length

    """

    length_of_seq = len(seq)

    x_chunks = []

    y_chunks = []



    for i in range(length_of_seq - chunk_length + 1):

        x_chunks.append(seq[i:i + chunk_length])

        y_chunks.append(seq[i + chunk_length - 1])



    return np.array(x_chunks), np.array(y_chunks)





seq_length = 5



# confirmed cases

X_train, y_train = chunk_of_sequences(train_data_infections, seq_length)



X_train_confirmed = torch.from_numpy(X_train).float()

y_train_confirmed = torch.from_numpy(y_train).float()



# fatalities

X_train, y_train = chunk_of_sequences(train_data_fatalities, seq_length)



X_train_fatalities = torch.from_numpy(X_train).float()

y_train_fatalities = torch.from_numpy(y_train).float()





## LSTM Construction

class COVID19Estimator(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,

                 learning_rate=1e-3, epochs=5, dropout=None, optimizer='Adam',

                 loss_criterion='MSELoss'):



        super(COVID19Estimator, self).__init__()



        # hidden dimensions

        self.hidden_dim = hidden_dim

        # number of hidden layers

        self.layer_dim = layer_dim



        if dropout is not None:

            self.lstm = nn.LSTM(input_dim,

                                hidden_dim,

                                layer_dim,

                                batch_first=True,

                                dropout=dropout)

        else:

            self.lstm = nn.LSTM(input_dim,

                                hidden_dim,

                                layer_dim,

                                batch_first=True)



        self.linear = nn.Linear(hidden_dim, output_dim)



        self.learning_rate = learning_rate

        self.epochs = epochs

        self.optimizer = eval('torch.optim.' + optimizer)

        self.criterion = eval('nn.' + loss_criterion)



    def forward(self, x):

        # initialize hidden state with zeros

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()



        # initialize cell state

        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()



        out, (hn, cn) = self.lstm(x, (h0, c0))



        return self.linear(out[:, -1, :])



    def fit(self, x, y):

        criterion = self.criterion(reduction='sum')

        optimizer = self.optimizer(self.parameters(),

                                   lr=self.learning_rate)



        loss_hist = np.zeros(self.epochs)



        for epoch in range(self.epochs):

            optimizer.zero_grad()



            outputs = self.forward(x)



            loss = criterion(outputs, y)



            loss.backward()



            optimizer.step()



            loss_hist[epoch] = loss.detach()



            if epoch % 10 == 0:

                print('Epoch {} || Train MSE Loss: {:0.4f}'.format(epoch + 1,

                                                                   loss.detach()))



        self.loss_hist = loss_hist



    def predict(self, x):

        return self.forward(x)



    def loss_plot(self):

        plt.figure(figsize=(10, 6))

        plt.plot(self.loss_hist)

        plt.title('MSE Loss', fontsize=25)

        plt.xlabel('Epochs', fontsize=15)

        plt.ylabel('Loss', fontsize=15)

        plt.show()



## Training the model



### Confirmed cases

input_dim = 1

hidden_dim = 256

layer_dim = 2

output_dim = 1



CovidConfirmedCases = COVID19Estimator(input_dim=input_dim,

                                       hidden_dim=hidden_dim,

                                       layer_dim=layer_dim,

                                       output_dim=output_dim,

                                       epochs=400,

                                       dropout=0.7)



CovidConfirmedCases.fit(X_train_confirmed, y_train_confirmed)

CovidConfirmedCases.loss_plot()



### Fatalities

CovidFatalities = COVID19Estimator(input_dim=input_dim,

                                   hidden_dim=hidden_dim,

                                   layer_dim=layer_dim,

                                   output_dim=output_dim,

                                   epochs=400,

                                   dropout=0.7)



CovidFatalities.fit(X_train_fatalities, y_train_fatalities)

CovidFatalities.loss_plot()





## Predict confirmed cases

date_length = len(test)

seq_length = 2



initialize = train[['ConfirmedCases']].loc[:test.index[0]][-3:-1].values

initialize = scaler_infections.transform(initialize).reshape(1, -1, 1)

initialize = torch.from_numpy(initialize).float()



predictions = []



for i in range(date_length):

    if i == 0:

        pred = CovidConfirmedCases.predict(initialize).view(1, -1, 1)

        initialize[0, 0], initialize[0, 1] = initialize[0, 1], pred.detach()

        predictions.append(pred.detach().numpy()[0, 0, 0])



    else:

        pred = CovidFatalities.predict(initialize).view(1, -1, 1)

        initialize[0, 0], initialize[0, 1] = initialize[0, 1], pred.detach()

        predictions.append(pred.detach().numpy()[0, 0, 0])



predicted_confirmed_cases = scaler_infections.inverse_transform(

    np.expand_dims(np.array(predictions), axis=0)).flatten().astype(np.int64)





## Predict Fatalities

date_length = len(test)

seq_length = 2



initialize = train[['Fatalities']].loc[:test.index[0]][-3:-1].values

initialize = scaler_fatalities.transform(initialize).reshape(1, -1, 1)

initialize = torch.from_numpy(initialize).float()



predictions = []



for i in range(date_length):

    if i == 0:

        pred = CovidFatalities.predict(initialize).view(1, -1, 1)

        initialize[0, 0], initialize[0, 1] = initialize[0, 1], pred.detach()

        predictions.append(pred.detach().numpy()[0, 0, 0])



    else:

        pred = CovidFatalities.predict(initialize).view(1, -1, 1)

        initialize[0, 0], initialize[0, 1] = initialize[0, 1], pred.detach()

        predictions.append(pred.detach().numpy()[0, 0, 0])



predicted_fatalities = scaler_fatalities.inverse_transform(

    np.expand_dims(np.array(predictions), axis=0)).flatten().astype(np.int64)





## Submission

submission = pd.DataFrame({'ForecastId': test['ForecastId'],

                           'ConfirmedCases': predicted_confirmed_cases,

                           'Fatalities': predicted_fatalities})



submission.index = sample_submission.index

submission.head()

submission.to_csv("submission.csv", index=False)