import numpy as np

import re

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option("display.max_columns", None)

pd.set_option("display.max_rows", None)

%matplotlib inline
ts_confirmed_new = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
ts_confirmed_new[ts_confirmed_new["Country/Region"] == "France"]
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))



axs[0].plot(ts_confirmed_new.sum()[2:])

axs[0].set_ylabel("Cases")

axs[1].set_ylabel("Cases")

axs[1].set_xlabel("time")

# axs[1].grid(True)

# axs[1].set_yscale("log")

# axs[0].set_yscale("log")



axs[1].plot(ts_confirmed_new[ts_confirmed_new["Country/Region"] != "China"].sum()[2:])

fig.tight_layout()
f_confirmed = ts_confirmed_new[ts_confirmed_new["Country/Region"] == "France"].iloc[0][

    4:

]



i_confirmed = ts_confirmed_new[ts_confirmed_new["Country/Region"] == "Italy"].sum(

    axis=0

)[4:]



c_confirmed = ts_confirmed_new[

    ts_confirmed_new["Country/Region"] == "Mainland China"

].sum(axis=0)[4:]

sg_confirmed = ts_confirmed_new[ts_confirmed_new["Country/Region"] == "Singapore"].sum(

    axis=0

)[4:]

ger_confirmed = ts_confirmed_new[ts_confirmed_new["Country/Region"] == "Germany"].sum(

    axis=0

)[4:]

sp_confirmed = ts_confirmed_new[ts_confirmed_new["Country/Region"] == "Spain"].sum(

    axis=0

)[4:]

skorea_confirmed = ts_confirmed_new[

    ts_confirmed_new["Country/Region"] == "Korea, South"

].sum(axis=0)[4:]



taiwan_confirmed = ts_confirmed_new[

    ts_confirmed_new["Country/Region"] == "Taiwan*"

].sum(axis=0)[4:]



f_confirmed.index = pd.to_datetime(f_confirmed.index)

i_confirmed.index = pd.to_datetime(i_confirmed.index)

c_confirmed.index = pd.to_datetime(c_confirmed.index)

sg_confirmed.index = pd.to_datetime(sg_confirmed.index)

sp_confirmed.index = pd.to_datetime(sp_confirmed.index)

ger_confirmed.index = pd.to_datetime(ger_confirmed.index)

skorea_confirmed.index = pd.to_datetime(skorea_confirmed.index)

taiwan_confirmed.index = pd.to_datetime(taiwan_confirmed.index)
fig = plt.figure(figsize=(10, 20))

ax = fig.add_subplot(2, 1, 1)

ax.set_yscale("log")



f_confirmed.plot(label="France", marker="^")

i_confirmed.plot(label="Italy")

c_confirmed.plot(label="China")

sg_confirmed.plot(label="Singapore")

sp_confirmed.plot(label="Spain")

skorea_confirmed.plot(label="South Korea")

ger_confirmed.plot(label="Germany")

taiwan_confirmed.plot(label="Taiwan")



plt.legend(loc="best")
def growth(serie):

    g = []

    index = []

    for i in range(len(f_confirmed) - 2):

        if serie.diff()[i] == 0:

            g.append(0)

        else:

            g.append(serie.diff()[i + 1] / serie.diff()[i])

        index.append(serie.index[i])

    return pd.Series(data=g, index=index).replace([np.inf, -np.inf], np.nan).fillna(0)
fig, ax = plt.subplots(2, 1, figsize=(6, 6))



growth(f_confirmed).plot(ax=ax[0], label="France")

growth(i_confirmed).plot(ax=ax[1], label="Italie")



# growth(c_confirmed).pct_change().plot(label='China')

# growth(sg_confirmed).pct_change().plot(label="Singapore")



ax[0].legend(loc="best")

ax[1].legend(loc="best")
f_confirmed_new = ts_confirmed_new[ts_confirmed_new["Country/Region"] == "France"].sum(axis=0)[10:]

f_confirmed_new.index = pd.to_datetime(f_confirmed_new.index)

def ratio_change(serie):

    g = []

    index = []

    for i in range(len(serie) - 1):

        if i == 0:

            g.append(1)

        else:

            g.append(serie[i] / serie[i - 1])

        index.append(serie.index[i])

    return pd.Series(data=g, index=index).replace([np.inf, -np.inf], np.nan).fillna(0)
sns.boxplot(

    ratio_change(

        ts_confirmed_new[ts_confirmed_new["Country/Region"] != "China"].sum()[10:]

    )

)

print(

    ratio_change(

        ts_confirmed_new[ts_confirmed_new["Country/Region"] != "China"].sum()[10:]

    ).describe()

)
sns.boxplot(ratio_change(f_confirmed_new[10:]))

print(ratio_change(f_confirmed_new[10:]).describe())
dates = []

serie = ts_confirmed_new[ts_confirmed_new["Country/Region"] != "China"].sum()[2:]

serie.index = pd.to_datetime(serie.index)

for i in range(len(serie) - 1):

    j = i

    while (serie[j] < 2 * serie[i]) & (j < len(serie) - 1):

        j = j + 1

    if j > i + 1:

        dates.append(serie.index[j] - serie.index[i])

print("Moyenne pour doubler le chiffre est de ", np.mean(dates))
ilist = i_confirmed.where(i_confirmed > 30).dropna().tolist()

flist = f_confirmed.where(f_confirmed > 40).dropna().tolist()



fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(2, 1, 1)

# ax.set_yscale("log")

ax.plot(ilist, "g--", label="Italy")

ax.plot(flist, "b--", label="France")
import math

from datetime import timedelta

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import mean_squared_error
# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence) - 1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.asarray(X), np.asarray(y)
f_confirmed = ts_confirmed_new[ts_confirmed_new["Country/Region"] == "France"].sum(axis=0)[10:]

f_confirmed.index = pd.to_datetime(f_confirmed.index)



# define input sequence

# choose a number of time steps ( equivalent to # of features)

n_steps = 4



X, y = split_sequence(f_confirmed[:'2020-03-14'], n_steps)
n_features = 1



X = X.reshape((X.shape[0], X.shape[1], n_features))



# define model

model = Sequential()

model.add(LSTM(10, activation="relu", input_shape=(n_steps, n_features)))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")





# fit model

history = model.fit(X, y, epochs=200, verbose=0)



# history for loss

plt.plot(history.history["loss"])

plt.title("model loss")

plt.ylabel("loss")

plt.xlabel("epoch")

plt.legend(["train", "test"], loc="upper left")

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(6, 6))



pred = model.predict(X)

pred_serie = pd.Series(data=pred.flatten(), index=f_confirmed[4:].index)



pred_serie.plot(ax=ax, label="prediction")

f_confirmed[4:].plot(ax=ax, label="training")



ax.legend(loc="best")
n = 30

index30 = pd.date_range(f_confirmed[:'2020-03-11'].index[-1] + timedelta(days=1), periods=n, freq="D")

last_values = f_confirmed[:'2020-03-11'][-n_steps:].values

prediction30 = []
for i in range(n):

    x_input = last_values[-n_steps:]

    x_input = x_input.reshape((1, n_steps, n_features))

    y_pred = model.predict(x_input)

    prediction30.append(int(y_pred[0][0]))

    last_values = np.append(last_values, int(y_pred[0][0]))



serie30_france = pd.Series(data=prediction30, index=index30)
fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(2, 1, 1)



ax.set_yscale("log")



f_confirmed_new = ts_confirmed_new[ts_confirmed_new["Country/Region"] == "France"].sum(

    axis=0

)[4:]

f_confirmed_new.index = pd.to_datetime(f_confirmed_new.index)





f_confirmed.plot(ax=ax, label="france_new", marker="o", linestyle="")

f_confirmed[:'2020-03-11'].plot(ax=ax, label="france")

# c_confirmed.plot(ax=ax, c="red", label="China")



serie30_france.plot(ax=ax, c="grey", linestyle="--", label="predition for 30 days")
# define input sequence

# choose a number of time steps ( equivalent to # of features)

n_steps = 3



X, y = split_sequence(skorea_confirmed[:-1], n_steps)



n_features = 1



# scaler = StandardScaler()

# trainX = scaler.fit_transform(X)

#

# trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], n_features))



X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model

model = Sequential()

model.add(LSTM(10, activation="relu", input_shape=(n_steps, n_features)))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")





# fit model

history = model.fit(X, y, epochs=250, verbose=0)



# history for loss

plt.plot(history.history["loss"])

plt.title("model loss")

plt.ylabel("loss")

plt.xlabel("epoch")

plt.legend(["train", "test"], loc="upper left")

plt.show()
n = 100

index30 = pd.date_range(f_confirmed.index[-1] + timedelta(days=1), periods=n, freq="D")

last_values = f_confirmed[-n_steps:].values

prediction30 = []





for i in range(n):

    x_input = last_values[-n_steps:]

    x_input = x_input.reshape((1, n_steps))

    x_input = x_input.reshape((1, n_steps, n_features))

    y_pred = model.predict(x_input)

    prediction30.append(int(y_pred[0][0]))

    last_values = np.append(last_values, int(y_pred[0][0]))



serie30 = pd.Series(data=prediction30, index=index30)
fig = plt.figure(figsize=(15, 30))

ax = fig.add_subplot(2, 1, 1)



ax.set_yscale("log")



f_confirmed[:'2020-03-11'].plot(ax=ax, label="Train data France")

f_confirmed_new.plot(ax=ax, label="france_new", marker="o", linestyle="")

skorea_confirmed[10:].plot(ax=ax, marker="^", label="Corée du Sud")



serie30.plot(

    ax=ax, c="grey", linestyle="--", label="prediction en suivant la corée du sud"

)





serie30_france.plot(

    ax=ax, c="red", linestyle="--", label="predition suivant la progression france"

)





ax.set_xlabel("Time")

ax.set_ylabel("Cases")

ax.legend(loc="best")
hosp = f_confirmed_new.apply(lambda x: x * 0.05)

hosp_france = serie30_france.apply(lambda x: x * 0.05)

hosp_ks = serie30.apply(lambda x: x * 0.05)
fig = plt.figure(figsize=(10, 20))

ax = fig.add_subplot(2, 1, 1)

ax.set_yscale("log")



hosp.plot(ax=ax, c="grey", linestyle="--", label="cas hospitalisé")

hosp_ks.plot(

    ax=ax, c="blue", linestyle="--", label="prediction en suivant la corée du sud"

)

hosp_france.plot(

    ax=ax, c="red", linestyle="--", label="predition suivant la progression france"

)



ax.set_ylabel("# Hospitalisés")

ax.set_xlabel("Time")

ax.legend(loc="best")

ax.axhline(y=5000, linewidth=3, color="black", alpha=0.5)
print(

    "Date de saturation du système hospitalien Francais dans le pire scénario :"

    + hosp_france.where(hosp_france >= 5000).dropna().index[0].strftime("%d/%m/%Y")

)

print(

    "Date de saturation du système hospitalien Francais dans le meilleur scénario :"

    + hosp_ks.where(hosp_ks >= 5000).dropna().index[0].strftime("%d/%m/%Y")

)