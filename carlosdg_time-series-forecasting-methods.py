!pip install --quiet pmdarima
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import pmdarima as pm

from pmdarima.model_selection import train_test_split

from fbprophet import Prophet

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import MinMaxScaler

import torch

import torch.nn as nn

import torch.nn.functional as F



np.random.seed(8)

torch.manual_seed(8);



# Comment this during development

import warnings; warnings.filterwarnings(action='ignore')
DATA_PATH = "/kaggle/input/corona-virus-report/"



df_spain = pd.read_csv(DATA_PATH + "covid_19_clean_complete.csv", parse_dates=["Date"])



df_spain = df_spain[(df_spain["Country/Region"] == "Spain") & 

                    (df_spain.Date >= "2020-02-24") &

                    (df_spain.Date <= "2020-05-10")]    # because the dataset is constantly being updated

df_spain = (df_spain

            .loc[:, ["Date", "Confirmed"]]  # only interested in confirmed cases

            .rename(columns={"Confirmed": "TargetValue"})

            .set_index("Date")

            .diff()         # convert to daily cases

            .iloc[1:]       # remove first row, it is NaN due to the .diff()

            .clip(lower=0)) # remove <0 values



fig = go.Figure()

fig.add_trace(go.Scatter(x=df_spain.index, y=df_spain.TargetValue, mode='lines+markers'))

fig.update_layout(hovermode="x",

                  title="Confirmed daily cases in Spain",

                  xaxis_title="Date",

                  yaxis_title="Number of confirmed daily cases")



fig.show()
def fit_predict_auto_arima(df_train, df_test):

    arima = pm.auto_arima(df_train, start_p=1, start_q=1)

    forecasts, forecasts_pred_interval = arima.predict(df_test.shape[0], 

                                                       return_conf_int=True, 

                                                       alpha=0.05)



    forecasts = pd.Series(data=forecasts, index=df_test.index)

    forecasts_pred_interval = pd.DataFrame(data=forecasts_pred_interval, 

                                           index=df_test.index,

                                           columns=["low", "high"])



    return forecasts, forecasts_pred_interval, arima.order



df_train, df_test = train_test_split(df_spain)

forecasts, forecasts_pred_interval, forecast_arima_order = fit_predict_auto_arima(df_train, df_test)
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_train.index, y=df_train.TargetValue, 

                         mode='lines+markers', name="train data"))

fig.add_trace(go.Scatter(x=df_test.index, y=df_test.TargetValue, 

                         mode='lines+markers', name="test data"))

fig.add_trace(go.Scatter(x=df_test.index, y=forecasts, 

                         mode='lines+markers', name="forecast"))

fig.add_trace(go.Scatter(x=df_test.index, y=forecasts_pred_interval.low, 

                         mode="lines", fill='tonexty', line_color="lightgreen", name="lower prediction interval"))

fig.add_trace(go.Scatter(x=df_test.index, y=forecasts_pred_interval.high, 

                         mode="lines", fill='tonexty', line_color="lightgreen", name="upper prediction interval"))



fig.update_layout(hovermode="x",

                  title=f"ARIMA{forecast_arima_order} applied to the number of confirmed cases in Spain",

                  xaxis_title="Date",

                  yaxis_title="Number of confirmed daily cases")



fig.show()
def fit_predict_prophet(df_train, df_test):



    df_temp = df_train.reset_index().rename(columns={

      'Date': 'ds',

      'TargetValue': 'y'

    })



    m = (Prophet(daily_seasonality=False,    # Don't add a daily seasonal component

                 yearly_seasonality=False,   # Don't add a yearly seasonal component

                 weekly_seasonality=False,   # Don't add a weekly seasonal component

                 changepoint_prior_scale=2)  # Make the model more flexible to trend changes

      .add_seasonality(name="twoweek", 

                       period=14,          # Model a seasonal component every two week

                       fourier_order=2)    # More orders -> more periods in seasonality -> overfit

      .add_country_holidays("Spain"))      # Add one-hot encoded variables for holidays



    m.fit(df_temp)



    future = m.make_future_dataframe(periods=df_test.shape[0])

    forecast = m.predict(future)

    forecast["ds"] = pd.to_datetime(forecast.ds)



    return forecast.set_index("ds").iloc[-df_test.shape[0]:]



forecasts_prophet = fit_predict_prophet(df_train, df_test)
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_train.index, y=df_train.TargetValue, 

                         mode='lines+markers', name="train data"))

fig.add_trace(go.Scatter(x=df_test.index, y=df_test.TargetValue, 

                         mode='lines+markers', name="test data"))

fig.add_trace(go.Scatter(x=df_test.index, y=forecasts_prophet.yhat, 

                         mode='lines+markers', name="forecast"))

fig.add_trace(go.Scatter(x=df_test.index, y=forecasts_prophet.yhat_lower, 

                         mode="lines", fill='tonexty', line_color="lightgreen", name="lower prediction interval"))

fig.add_trace(go.Scatter(x=df_test.index, y=forecasts_prophet.yhat_upper, 

                         mode="lines", fill='tonexty', line_color="lightgreen", name="upper prediction interval"))



fig.update_layout(hovermode="x",

                  title="Prophet applied to the number of confirmed cases in Spain",

                  xaxis_title="Date",

                  yaxis_title="Number of confirmed daily cases")



fig.show()
X_train = df_spain.reset_index()



# Add datetime features

X_train["Month"] = X_train.Date.dt.month

X_train["Week"] = X_train.Date.dt.week

X_train["Day"] = X_train.Date.dt.day

X_train["Dayofweek"] = X_train.Date.dt.dayofweek

X_train["IsWeekEnd"] = X_train.Date.dt.dayofweek.isin([5, 6]).astype(int)

X_train.drop("Date", axis=1, inplace=True)



# Add lag features

N_SHIFTS = 14

X_train["TargetValue"] = X_train.TargetValue

for i in range(N_SHIFTS):

    X_train[f"TargetShift{i+1}"] = X_train.TargetValue.shift(i+1).fillna(0)



X_train = pd.get_dummies(X_train, columns=['Month', 'Week', 'Day', 'Dayofweek', 'IsWeekEnd'])



y_train = X_train.TargetValue

X_train = X_train.drop("TargetValue", axis=1)



n_test = df_test.shape[0]

X_train = X_train.iloc[:-n_test]

y_train = y_train.iloc[:-n_test]

X_valid = X_train.iloc[-n_test:]

y_valid = y_train.iloc[-n_test:]
model_percentile_975 = GradientBoostingRegressor(loss='quantile', alpha=0.975, random_state=8)

model_percentile_50  = GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=8)

model_percentile_025  = GradientBoostingRegressor(loss='quantile', alpha=0.025, random_state=8)



model_percentile_50.fit(X_train, y_train)

model_percentile_025.fit(X_train, y_train)

model_percentile_975.fit(X_train, y_train);
pred_975 = model_percentile_975.predict(X_valid)

pred_50  = model_percentile_50.predict(X_valid)

pred_025 = model_percentile_025.predict(X_valid)



pred_50 = pd.Series(pred_50, index=df_test.index)
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_train.index, y=df_train.TargetValue, 

                         mode='lines+markers', name="train data"))

fig.add_trace(go.Scatter(x=df_test.index, y=df_test.TargetValue, 

                         mode='lines+markers', name="test data"))

fig.add_trace(go.Scatter(x=df_test.index, y=pred_50, 

                         mode='lines+markers', name="forecast"))

fig.add_trace(go.Scatter(x=df_test.index, y=pred_025, 

                         mode="lines", fill='tonexty', line_color="lightgreen", name="lower prediction interval"))

fig.add_trace(go.Scatter(x=df_test.index, y=pred_975, 

                         mode="lines", fill='tonexty', line_color="lightgreen", name="upper prediction interval"))



fig.update_layout(hovermode="x",

                  title="Gradient Boosting applied to the number of confirmed cases in Spain",

                  xaxis_title="Date",

                  yaxis_title="Number of confirmed daily cases")



fig.show()
def create_sequences(data, length):

    xs = [data[i:(i + length)] for i in range(data.shape[0] - length - 1)]

    ys = [data[(i + length)] for i in range(data.shape[0] - length - 1)]



    xs = np.array(xs)

    ys = np.array(ys)



    xs = torch.from_numpy(xs).float().view(-1, length, 1)

    ys = torch.from_numpy(ys).float().view(-1, 1)



    return xs, ys
train = df_train.TargetValue.values.reshape(-1, 1)

test = df_test.TargetValue.values.reshape(-1, 1)



scaler = MinMaxScaler()

train = scaler.fit_transform(train)

test = scaler.transform(test)



sequence_length = 3



X_train, y_train = create_sequences(train, sequence_length)

X_test, y_test = create_sequences(test, sequence_length)
class RNN(nn.Module):

    def __init__(self, input_size=1, output_size=1,

               hidden_dim=10, num_layers=1, 

               sequence_length=sequence_length):

        """

          Good explanation:

            https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-rnn-cb6ebc594677



          input_size: 

            number of features of the input: 1 in our case because we are dealing 

            with number of cases. Could be more, for example, for one-hot encoded 

            input



          output_size:

            1 because we want to predict just the number of cases that follow

            the sequence



          num_layers:

            number of stacked RNN layers (if more than one, it means that the 

            output of the first RNN layer will pass to another RNN and so on)



          sequence_length:

            number of previous values to take into account



        """

        super(RNN, self).__init__()



        self.input_size = input_size

        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        self.sequence_length = sequence_length



        self.rnn = nn.RNN(input_size, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim * sequence_length, output_size)



    def forward(self, sequences):

        # (batch_size, sequence_length, input_size)

        sequences = sequences.view(-1, self.sequence_length, self.input_size)



        # Dummy hidden state for first input

        hidden = self.initial_hidden(sequences.shape[0])



        # Pass through the RNN layer

        out, hidden = self.rnn(sequences, hidden)



        # Flatten the output and pass it to the linear layer

        out = out.flatten(start_dim=1)

        out = self.fc(out)



        return out, hidden



    def initial_hidden(self, batch_size):

        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)
model_rnn = RNN()

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(lr=5e-3, params=model_rnn.parameters())



def train(num_epochs, model, optimizer, loss_fn, verbose=False):

    for epoch in range(1, num_epochs + 1):

        optimizer.zero_grad()

        output, _ = model_rnn(X_train)

        loss = loss_fn(output, y_train)



        with torch.no_grad():

            y_test_pred, _ = model(X_test)

            test_loss = loss_fn(y_test_pred.float(), y_test)



        loss.backward()

        optimizer.step()



        if verbose and epoch%10 == 0:

            print(f'Epoch: {epoch}/{num_epochs}. Train Loss: {loss.item():.3f}. Test loss: {test_loss.item(): .3f}')



train(100, model_rnn, optimizer, loss_fn)
with torch.no_grad():

    model_rnn.eval()

    out, _ = model_rnn(X_test)

    out = scaler.inverse_transform(out.numpy().reshape(1, -1)).reshape(-1)



    tr_out, _ = model_rnn(X_train)

    tr_out = scaler.inverse_transform(tr_out.numpy().reshape(1, -1)).reshape(-1)
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_train.index, y=df_train.TargetValue, 

                         mode='lines+markers', name="train data"))

fig.add_trace(go.Scatter(x=df_train.index[-tr_out.shape[0]:], y=tr_out, 

                         mode='lines+markers', name="train forecast"))

fig.add_trace(go.Scatter(x=df_test.index, y=df_test.TargetValue, 

                         mode='lines+markers', name="test data"))

fig.add_trace(go.Scatter(x=df_test.index[-out.shape[0]:], y=out, 

                         mode='lines+markers', name="test forecast"))



fig.update_layout(hovermode="x",

                  title="RNN applied to the number of confirmed cases in Spain",

                  xaxis_title="Date",

                  yaxis_title="Number of confirmed daily cases")



fig.show()
class KalmanFilter:

    def __init__(self, x, F, P, Q, H, u=None, G=None):

        self.x = x

        self.F = F

        self.P = P

        self.Q = Q

        self.H = H

        self.u = u

        self.G = G



        if G is None or u is None:

            self.u = np.zeros_like(x)

            self.G = np.zeros_like(F)



        self.predict()





    def predict(self):

        self.x = self.F @ self.x + self.G @ self.u

        self.P = self.F @ self.P @ self.F.T + self.Q





    def update(self, z, R):

        K = self.P + self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)

        self.x = self.x + K @ (z - self.H @ self.x)

        self.P = self.P - K @ self.H @ self.P





class KalmanFilter1D:

    def __init__(self, x, F, P, Q):

        self.x = x

        self.F = F

        self.P = P

        self.Q = Q



        self.predict()





    def predict(self):

        self.x = self.F * self.x

        self.P = self.P * self.F**2  + self.Q





    def update(self, z, R):

        K = self.P / (self.P + R)

        self.x = self.x + K * (z - self.x)

        self.P = self.P - K * self.P
x = 0

F = 1

P = 1

Q = 50

zs = df_spain.cumsum().values.reshape(-1)

Rs = np.ones_like(zs) * 100



kalman = KalmanFilter1D(x, F, P, Q)

corrections = []

predictions = []

verbose = False



for t, (z, R) in enumerate(zip(zs, Rs)):

    predictions.append(kalman.x)

    if verbose: print(f"Time {t+1}: Prediction for current state: {kalman.x:.4f}.")



    kalman.update(z, R)

    corrections.append(kalman.x)

    if verbose: print(f"Time {t+1}: Measurement: {z:.4f}. Correction: {kalman.x:.4f}")



    kalman.predict()
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_spain.index, y=zs,

                    mode='lines+markers',

                    name='measurements'))

fig.add_trace(go.Scatter(x=df_spain.index, y=predictions,

                    mode='lines+markers',

                    name='predictions'))

#fig.add_trace(go.Scatter(x=df_spain.index, y=corrections,

#                    mode='lines+markers', 

#                    name='after corrections'))



fig.update_layout(hovermode="x",

                  title="Kalman Filter applied to the total number of confirmed cases in Spain",

                  xaxis_title="Date",

                  yaxis_title="Number of total confirmed cases")



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_spain.index[1:], y=np.diff(zs).astype(int),

                    mode='lines+markers',

                    name='measurements'))

fig.add_trace(go.Scatter(x=df_spain.index[1:], y=np.diff(predictions).astype(int),

                    mode='lines+markers',

                    name='predictions'))

#fig.add_trace(go.Scatter(x=df_spain.index[1:], y=np.diff(corrections).astype(int),

#                    mode='lines+markers', 

#                    name='after corrections'))

fig.update_layout(hovermode="x",

                  title="Kalman Filter applied to the number of daily confirmed cases in Spain",

                  xaxis_title="Date",

                  yaxis_title="Number of daily confirmed cases")



fig.show()