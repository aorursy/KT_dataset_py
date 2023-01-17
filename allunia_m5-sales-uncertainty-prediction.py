import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



from sklearn.preprocessing import StandardScaler, MinMaxScaler



from fbprophet import Prophet

from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()





import time

from tqdm import tqdm_notebook as tqdm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sales_train_validation.csv")

train.head()
train.shape
calendar = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/calendar.csv")

calendar.head()
sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv")

sell_prices.head()
series_cols = train.columns[train.columns.str.contains("d_")].values

level_cols = train.columns[train.columns.str.contains("d_")==False].values
train.head(1)
sns.set_palette("colorblind")



fig, ax = plt.subplots(5,1,figsize=(20,28))

train[series_cols].sum().plot(ax=ax[0])

ax[0].set_title("Top-Level-1: Summed product sales of all stores and states")

ax[0].set_ylabel("Unit sales of all products");

train.groupby("state_id")[series_cols].sum().transpose().plot(ax=ax[1])

ax[1].set_title("Level-2: Summed product sales of all stores per state");

ax[1].set_ylabel("Unit sales of all products");

train.groupby("store_id")[series_cols].sum().transpose().plot(ax=ax[2])

ax[2].set_title("Level-3: Summed product sales per store")

ax[2].set_ylabel("Unit sales of all products");

train.groupby("cat_id")[series_cols].sum().transpose().plot(ax=ax[3])

ax[3].set_title("Level-4: Summed product sales per category")

ax[3].set_ylabel("Unit sales of all products");

train.groupby("dept_id")[series_cols].sum().transpose().plot(ax=ax[4])

ax[4].set_title("Level-4: Summed product sales per product department")

ax[4].set_ylabel("Unit sales of all products");
submission = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sample_submission.csv")

submission.head(10)
submission.shape
np.random.choice(submission.id.values, replace=False, size=15)
def find_quartil(l):

    

    if "0.005" in l:

        return 0.005

    elif "0.025" in l:

        return 0.025

    elif "0.165" in l:

        return 0.165

    elif "0.25" in l:

        return 0.25

    elif "0.5" in l:

        return 0.5

    elif "0.75" in l:

        return 0.75

    elif "0.835" in l:

        return 0.835

    elif "0.975" in l:

        return 0.975

    elif "0.995" in l:

        return 0.995

    else:

        return 0

    

def find_state(l):

    if "CA" in l:

        return "CA"

    elif "TX" in l:

        return "TX"

    elif "WI" in l:

        return "WI"

    else:

        return "Unknown"

    

def find_category(l):

    if "FOODS" in l:

        return "foods"

    elif "HOBBIES" in l:

        return "hobbies"

    elif "HOUSEHOLD" in l:

        return "household"

    else:

        return "Unknown"
submission_eda = pd.DataFrame(submission.id, columns=["id"])

submission_eda.loc[:, "lb_type"] = np.where(submission.id.str.contains("validation"), "validation", "evaluation")

submission_eda.loc[:, "u"] = submission.id.apply(lambda l: find_quartil(l))

submission_eda.loc[:, "state"] = submission.id.apply(lambda l: find_state(l))

submission_eda.loc[:, "category"] = submission.id.apply(lambda l: find_category(l))
sns.set_palette("husl")



fig, ax = plt.subplots(3,3,figsize=(20,20))

sns.countplot(submission_eda.u, ax=ax[0,0]);

sns.countplot(submission_eda.lb_type, ax=ax[0,1]);

sns.countplot(submission_eda.state, ax=ax[1,0]);

sns.countplot(submission_eda.loc[submission_eda.lb_type=="validation"].state, ax=ax[1,1]);

sns.countplot(submission_eda.loc[submission_eda.lb_type=="evaluation"].state, ax=ax[1,2]);

sns.countplot(submission_eda.category, ax=ax[2,0]);

sns.countplot(submission_eda.loc[submission_eda.lb_type=="validation"].category, ax=ax[2,1]);

sns.countplot(submission_eda.loc[submission_eda.lb_type=="evaluation"].category, ax=ax[2,2]);

for n in range(1,3):

    ax[n,2].set_title("in evaluation")

    ax[n,1].set_title("in validation")
def spl_denominator(train_series):

    N = len(train_series)

    sumup = 0

    for n in range(1, N):

        sumup += np.abs(train_series[n]-train_series[n-1])

    return sumup/(N-1)
def spl_numerator(dev_series, Q, u):

    sumup = 0

    for m in range(len(dev_series)):

        if Q[m] <= dev_series[m]:

            sumup += (dev_series[m] - Q[m])*u

        else:

            sumup += (Q[m] - dev_series[m])*(1-u)

    return sumup
def spl(train_series, dev_series, Q, u):

    h = len(dev_series)

    spl_denomina = spl_denominator(train_series)

    spl_numera = spl_numerator(dev_series, Q, u)

    

    return spl_numera/(h*spl_denomina)
idx = 1000
train[level_cols].iloc[idx]
train.loc[train.item_id=="HOUSEHOLD_1_445"].store_id.unique()
plt.figure(figsize=(20,5))

plt.plot(train[series_cols].iloc[idx].values, 'o')

plt.title("Item 445 daily sales in shop CA_1");

plt.xlabel("observed days")

plt.ylabel("Unit sales");
timeseries = train[series_cols].iloc[idx].values

h = 28



train_timeseries = timeseries[0:len(timeseries)-h]

dev_timeseries = timeseries[(len(timeseries)-h)::]



print(len(train_timeseries), len(dev_timeseries))
naive_val = train_timeseries[-1]

naive_Q = np.ones(dev_timeseries.shape) * naive_val

naive_Q
spl(train_timeseries, dev_timeseries, naive_Q, 0.5)
naive_val
residuals = train_timeseries - naive_val
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(residuals, ax=ax[0], kde=False)

ax[0].set_xlabel("residuals")

ax[0].set_ylabel("frequency");

ax[0].set_title("Distribution of residuals");
np.mean(residuals)
std_dev = np.std(residuals)

std_h = np.ones(dev_timeseries.shape)



for h in range(1, 29):

    std_h[h-1] = std_dev * np.sqrt(h)
std_h
y_lower = np.ones(len(std_h))

y_upper = np.ones(len(std_h))

for h in range(len(std_h)):

    low_val = naive_Q[h] - 2.58 * std_h[h]

    if low_val < 0:

        y_lower[h] = 0

    else:

        y_lower[h] = low_val

    y_upper[h] = naive_Q[h] + 2.58 * std_h[h]
plt.figure(figsize=(20,5))

plt.plot(y_lower, c="r", label="0.005 boundary")

plt.plot(y_upper, c="g", label="0.995 boundary")

plt.plot(naive_Q, 'o', c="b", label="predicted value")

plt.title("Computing 99% PI for one timeseries of level 12");

plt.xlabel("time horizont h=28 days")

plt.ylabel("Unit sales");

plt.legend();
timeseries = train[series_cols].sum().values

len(timeseries)
train_timeseries = timeseries[0:-28]

eval_timeseries = timeseries[-28::]

print(len(train_timeseries), len(eval_timeseries))

days = np.arange(1, len(series_cols)+1)
plt.figure(figsize=(20,5))

plt.plot(days[0:-28], train_timeseries, label="train")

plt.plot(days[-28::], eval_timeseries, label="validation")

plt.title("Top-Level-1: Summed product sales of all stores and states");

plt.legend()

plt.xlabel("Day")

plt.ylabel("Unit sales");
dates = calendar.iloc[0:len(timeseries)].date.values

df = pd.DataFrame(dates, columns=["ds"])

df.loc[:, "y"] = timeseries

df.head()
train_df = df.iloc[0:-28]

train_df.shape
eval_df = df.iloc[-28::]

eval_df.shape
uncertainty_interval_width = 0.25
m = Prophet(interval_width=uncertainty_interval_width)

m.fit(train_df)

future = m.make_future_dataframe(periods=28)

forecast = m.predict(future)

forecast.head()
col_int = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']

forecast[col_int].head()
plt.plot(forecast.iloc[-28::].yhat.values, 'o', label="predicted yhat")

plt.plot(eval_df.y.values, 'o-', label="target")

plt.legend();
fig = plot_plotly(m, forecast)  

py.iplot(fig)
uncertainty_interval_width = 0.25
f_cols = [col for col in submission.columns if "F" in col]
submission_val = submission[submission.id.str.contains("validation")].copy()
def plugin_total_predictions():

    

    for uncertainty_interval_width in [0.005, 0.025, 0.165, 0.25]:

        upper = 1-uncertainty_interval_width

        lower = uncertainty_interval_width

    

        m = Prophet(interval_width=uncertainty_interval_width)

        m.fit(df)

        future = m.make_future_dataframe(periods=28)

        forecast = m.predict(future)

    

        submission_val.loc[

            (submission_val.id.str.contains("Total")) & (submission_val.id.str.contains(str(lower))),f_cols

        ] = np.round(forecast.yhat_lower.values[-28::])

    

        submission_val.loc[

            (submission_val.id.str.contains("Total")) & (submission_val.id.str.contains(str(upper))),f_cols

        ] = np.round(forecast.yhat_upper.values[-28::])

    

    submission_val.loc[

        (submission_val.id.str.contains("Total")) & (submission_val.id.str.contains(str(0.5))),f_cols

    ] = forecast.yhat.values[-28::]

    

    return submission_val
submission_val = plugin_total_predictions()

submission_val.loc[submission_val.id.str.contains("Total")]
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
class MyLSTM(nn.Module):

    

    def __init__(self, input_dim, hidden_dim, batch_size, num_layers=1, output_dim=1):

        super().__init__()

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

        self.num_layers = num_layers

        

        self.lstm = nn.LSTM(input_size=self.input_dim,

                            hidden_size=self.hidden_dim,

                            num_layers=self.num_layers,

                            dropout = 0.25)

        self.linear = nn.Linear(self.hidden_dim, output_dim)

        

    def init_hidden(self):

        self.h_zero = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)

        self.c_zero = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)

    

    def forward(self, x):

        lstm_output, (h_n, c_n) = self.lstm(x.view(len(x), self.batch_size, -1),

                                           (self.h_zero, self.c_zero))

        last_time_step = lstm_output.view(self.batch_size, len(x), self.hidden_dim)[-1]

        pred = self.linear(last_time_step)

        return pred

    



def train_model(model, data_dict, lr=1e-4, num_epochs=500):

    

    loss_fun = torch.nn.MSELoss(reduction="mean")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    

    train_losses = np.zeros(num_epochs)

    phases = ["train", "eval"]

    losses_dict = {"train": [], "eval": []}

    predictions_dict = {"train": [], "eval": [] }

    

    for n in range(num_epochs):

        

        for phase in phases:

            

            x = data_dict[phase]["input"].to(device, dtype=torch.float)

            y = data_dict[phase]["target"].to(device, dtype=torch.float)

            

            if phase == "train":

                model.train()

            else:

                model.eval()

        

            optimiser.zero_grad()

            

            model.init_hidden()

            y_pred = model(x)

            

            if n == (num_epochs-1):

                predictions_dict[phase] = y_pred.float().cpu().detach().numpy()

            

            loss = loss_fun(y_pred.float(), y)

            losses_dict[phase].append(loss.item())

            

            if n % 50 == 0:

                print("{} loss: {}".format(phase, loss.item()))

            

            if phase == 'train':

                loss.backward()

                optimiser.step()

        

    return losses_dict, predictions_dict



def create_sequences(timeseries, seq_len):

    inputs = []

    targets = []

    

    max_steps = len(timeseries) - (seq_len+1)

    

    for t in range(max_steps):

        x = timeseries[t:(t+seq_len)]

        y = timeseries[t+seq_len]

        inputs.append(x)

        targets.append(y)

    

    return np.array(inputs), np.array(targets)
diff_series = np.diff(timeseries)

train_size = np.int(0.7 * len(diff_series))

train_diff_series = diff_series[0:train_size]

eval_diff_series = diff_series[train_size::]

scaler = MinMaxScaler(feature_range=(-1,1))

scaled_train = scaler.fit_transform(train_diff_series.reshape(-1, 1))

scaled_eval = scaler.transform(eval_diff_series.reshape(-1,1))
fig, ax = plt.subplots(1,2,figsize=(20,5))

ax[0].plot(scaled_train, '-o', c="b")

ax[1].plot(scaled_eval, '-o', c="g")

ax[0].set_title("Single preprocessed top timeseries in train")

ax[1].set_title("Single preprocessed top timeseries in eval");

ax[0].set_xlabel("Days in dataset")

ax[1].set_xlabel("Days in dataset")

ax[0].set_ylabel("$\Delta y$ scaled")

ax[1].set_ylabel("$\Delta y$ scaled");
seq_len = 400

input_dim = 1

hidden_dim = 128

num_epochs = 600

lr=0.0005





x_train, y_train = create_sequences(scaled_train, seq_len)

x_eval, y_eval = create_sequences(scaled_eval, seq_len)



x_train = torch.from_numpy(x_train).float()

y_train = torch.from_numpy(y_train).float()

x_eval = torch.from_numpy(x_eval).float()

y_eval = torch.from_numpy(y_eval).float()



data_dict = {"train": {"input": x_train, "target": y_train},

             "eval": {"input": x_eval, "target": y_eval}}
model = MyLSTM(input_dim=input_dim,

               hidden_dim=hidden_dim,

               batch_size=seq_len)

model = model.to(device)
run_training = True

if run_training:

    losses_dict, predictions_dict = train_model(model, data_dict, num_epochs=num_epochs, lr=lr)
if run_training:

    

    fig, ax = plt.subplots(3,1,figsize=(20,20))

    ax[0].plot(losses_dict["train"], '.-', label="train", c="red")

    ax[0].set_xlabel("Epochs")

    ax[0].set_ylabel("MSE")

    ax[0].plot(losses_dict["eval"], '.-', label="eval", c="blue");

    ax[0].legend();



    ax[1].plot(predictions_dict["train"], '-o', c="red")

    ax[1].plot(y_train, '-o', c="green")

    ax[1].set_title("Fitted and true values of y in train");

    ax[1].set_ylabel("Unit sales y");

    ax[1].set_xlabel("Number of days in train");



    ax[2].plot(predictions_dict["eval"], '-o', c="red")

    ax[2].plot(y_eval, '-o', c="green")

    ax[2].set_title("Predicted and true values of y in eval");

    ax[2].set_xlabel("Number of days in eval");

    ax[2].set_ylabel("Unit sales y");
from statsmodels.graphics.tsaplots import plot_acf



if run_training:

    

    train_residuals = y_train-predictions_dict["train"]

    eval_residuals = y_eval-predictions_dict["eval"]

    

    fig, ax = plt.subplots(2,2,figsize=(20,10))

    sns.distplot(train_residuals, ax=ax[0,0], color="red")

    sns.distplot(eval_residuals, ax=ax[0,1], color="green")

    ax[0,0].set_title("Train residuals")

    ax[0,1].set_title("Eval residuals")

    ax[0,0].set_xlabel("$y_{true} - y_{pred}$")

    ax[0,1].set_xlabel("$y_{true} - y_{pred}$")

    ax[0,0].set_ylabel("density")

    ax[0,1].set_ylabel("density")

    

    plot_acf(train_residuals, ax=ax[1,0])

    plot_acf(eval_residuals, ax=ax[1,1])
sampled_residuals = np.random.choice(train_residuals[:, 0], size=len(y_train), replace=True)

sampled_residuals = sampled_residuals.reshape(-1,1)

new_response = predictions_dict["train"] + sampled_residuals
fig, ax = plt.subplots(2,2,figsize=(20,10))

ax[0,0].plot(predictions_dict["train"][0:200], 'o-', color="purple")

ax[0,0].set_title("Original fitted values $y_{pred}$ in ")

ax[0,0].set_xlabel("200 example days")

ax[0,0].set_ylim(-0.4, 0.4)

ax[0,0].set_ylabel("$y_{fitted}$")



ax[0,1].plot(new_response[0:200,0], 'o-', color="orange")

ax[0,1].set_title("Response values $y^{*}$ using sampled residuals");

ax[0,1].set_xlabel("200 example days")

ax[0,1].set_ylabel("$y^{*}$");

ax[0,1].set_ylim(-0.4, 0.4)



ax[1,0].plot(sampled_residuals[0:200], 'o-', color="cornflowerblue")

ax[1,0].set_title("Sampled residuals")

ax[1,0].set_xlabel("200 example days")

ax[1,0].set_ylabel("$\epsilon$")



ax[1,1].plot(y_train[0:200], 'o-', color="firebrick")

ax[1,1].set_title("True values $y_{train}$")

ax[1,1].set_xlabel("200 example days")

ax[1,1].set_ylabel("$y_{train}$");

responses = []

for n in range(100):

    # sample residuals using the historical residuals found in train

    sampled_residuals = np.random.choice(train_residuals[:, 0], size=len(y_eval), replace=True)

    sampled_residuals = sampled_residuals.reshape(-1,1)

    # create a synthetic future timeseries of eval by adding sampled residuals

    new_response = predictions_dict["eval"] + sampled_residuals

    # reverse the scaling

    new_response = scaler.inverse_transform(new_response)

    # concat the first value of the evaluation series and the response series

    new_response = np.hstack((timeseries[train_size], new_response[:,0]))

    # reverse the differnciation (trend removal) using cumsum

    new_response = np.cumsum(new_response)

    # save the future timeseries

    responses.append(new_response)

    

responses = np.array(responses)

responses.shape
y_eval.shape
median_series = np.median(responses, axis=0)

eval_series = scaler.inverse_transform(y_eval)

eval_series = np.cumsum(np.hstack((timeseries[train_size-1], eval_series[:,0])))

low_q = 0.25

up_q = 0.75
plt.figure(figsize=(20,5))

plt.plot(np.arange(0, len(median_series)), median_series, 'o-', label="median predicted series")

plt.plot(eval_series, '.-', color="cornflowerblue", label="true eval series")

lower = np.quantile(responses, low_q, axis=0)

upper = np.quantile(responses, up_q, axis=0)

plt.fill_between(np.arange(0, len(median_series)), lower, upper, alpha=0.5)

plt.title("Prediction interval {}% of eval timeseries".format((up_q-low_q)*100));

plt.xlabel("Days in eval")

plt.ylabel("Unit sales");

plt.legend();