import pandas as pd

# from tqdm import tqdm_notebook as tqdm

from tqdm import tqdm

import matplotlib.pyplot as plt



from pykalman import KalmanFilter

from pykalman import UnscentedKalmanFilter
#Loading data

train_df = pd.read_csv("../input/data-without-drift/train_clean.csv")

test_df = pd.read_csv("../input/data-without-drift/test_clean.csv")
fig, ax = plt.subplots(figsize=(36, 6))

ax.plot(train_df.time, train_df.signal, label="train", linewidth = 0.05)

ax.plot(test_df.time, test_df.signal, label="test", linewidth = 0.05)

plt.legend()

plt.show()
# train_df = train_df[train_df.time < .1]

# test_df  = test_df[test_df.time < 500.1]
# fig, ax = plt.subplots(figsize=(36, 6))

# ax.plot(train_df.time, train_df.signal, color="blue",label="train", linewidth = 1)

# plt.legend()

# plt.show()
# fig, ax = plt.subplots(figsize=(36, 6))

# ax.plot(test_df.time, test_df.signal,  color="orange",label="test", linewidth = 1)

# plt.legend()

# plt.show()
def Kalman1D(observations,damping=1):

    # To return the smoothed time series data

    observation_covariance = damping

    initial_value_guess = observations[0]

    transition_matrix = 1

    transition_covariance = 0.1

    initial_value_guess

    

    kf = KalmanFilter(

            initial_state_mean=initial_value_guess,

            initial_state_covariance=observation_covariance,

            observation_covariance=observation_covariance,

            transition_covariance=transition_covariance,

            transition_matrices=transition_matrix

        )

    

    pred_state, state_cov = kf.smooth(observations)

    return pred_state





# Kalman Filter

observation_covariance = .0015

train_df["signal_kf"] = Kalman1D(train_df.signal.values, observation_covariance)

test_df["signal_kf"] = Kalman1D(test_df.signal.values, observation_covariance)
fig, ax = plt.subplots(figsize=(36, 6))

ax.plot(train_df.time, train_df.signal_kf, color="blue",label="train", linewidth = 1)

plt.legend()

plt.show()
fig, ax = plt.subplots(figsize=(36, 6))

ax.plot(test_df.time, test_df.signal_kf,  color="orange",label="test", linewidth = 1)

plt.legend()

plt.show()
def UnscentedKalman1D(observations, damping=1):

    # To return the smoothed time series data

    observation_covariance = damping

    initial_value_guess = observations[0]

    transition_matrix = 1

    transition_covariance = 0.1

    initial_value_guess

       

    ukf = UnscentedKalmanFilter(

            initial_state_mean=initial_value_guess,

            initial_state_covariance=observation_covariance,

            observation_covariance=observation_covariance,

            transition_covariance=transition_covariance,

        )

    

    pred_state, state_cov = ukf.smooth(observations)

    return pred_state





# Kalman Filter

observation_covariance = .0015

train_df["signal_ukf"] = UnscentedKalman1D(train_df.signal.values, observation_covariance)

test_df["signal_ukf"] = UnscentedKalman1D(test_df.signal.values, observation_covariance)
fig, ax = plt.subplots(figsize=(36, 6))

ax.plot(train_df.time, train_df.signal_ukf, color="blue",label="train", linewidth = 1)

plt.legend()

plt.show()
fig, ax = plt.subplots(figsize=(36, 6))

ax.plot(test_df.time, test_df.signal_ukf,  color="orange",label="test", linewidth = 1)

plt.legend()

plt.show()
def AdditiveUnscentedKalman1D(observations, damping=1):

    # To return the smoothed time series data

    observation_covariance = damping

    initial_value_guess = observations[0]

    transition_matrix = 1

    transition_covariance = 0.1

    initial_value_guess

       

    aukf = AdditiveUnscentedKalmanFilter(

            initial_state_mean=initial_value_guess,

            initial_state_covariance=observation_covariance,

            observation_covariance=observation_covariance,

            transition_covariance=transition_covariance,

        )

    

    pred_state, state_cov = aukf.smooth(observations)

    return pred_state





# Kalman Filter

observation_covariance = .0015

train_df["signal_aukf"] = UnscentedKalman1D(train_df.signal.values, observation_covariance)

test_df["signal_aukf"] = UnscentedKalman1D(test_df.signal.values, observation_covariance)
fig, ax = plt.subplots(figsize=(36, 6))

ax.plot(train_df.time, train_df.signal_aukf, color="blue",label="train", linewidth = 1)

plt.legend()

plt.show()
fig, ax = plt.subplots(figsize=(36, 6))

ax.plot(test_df.time, test_df.signal_aukf,  color="orange",label="test", linewidth = 1)

plt.legend()

plt.show()
for type in ["signal", "signal_kf", "signal_ukf", "signal_aukf"]:

    fig, ax = plt.subplots(figsize=(36, 6))

    print(type)

    ax.plot(train_df.time, train_df[type], label="train", linewidth = 0.05)

    ax.plot(test_df.time, test_df[type], label="test", linewidth = 0.05)

    plt.legend()

    plt.show()
train_df_ukf["signal"] = train_df.signal_ukf

train_df_ukf = train_df_ukf[["time", "signal", "open_channels"]]

train_df_ukf.to_csv("train_clean_kalman_ukf.csv", index=False, float_format="%.4f")



test_df_ukf["signal"] = test_df.signal_ukf

test_df_ukf = test_df_ukf[["time", "signal_ukf"]]

test_df_ukf.to_csv("test_clean_kalman_ukf.csv", index=False, float_format="%.4f")
train_df_ukf.head()
test_df_ukf.head()
train_df_aukf["signal"] = train_df.signal_aukf

train_df_aukf = train_df_aukf[["time", "signal", "open_channels"]]

train_df_aukf.to_csv("train_clean_kalman_aukf.csv", index=False, float_format="%.4f")



test_df_aukf["signal"] = test_df.signal_aukf

test_df_aukf = test_df_ukf[["time", "signal_ukf"]]

test_df_aukf.to_csv("test_clean_kalman_aukf.csv", index=False, float_format="%.4f")
train_df_aukf.head()
test_df_aukf.head()