import pandas as pd
import numpy as np
import math
import time
from tqdm import tqdm
from IPython.display import Image
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_style('whitegrid')

from sklearn.metrics import mean_absolute_error

np.random.seed(123)
def linear_trend(t, k=1):
    """
    linear_trend = k*t
    """
    return k*t.reshape(-1,1)

def cyclical_component_1(t):
    """
    cyclical_component_1 = t * sin(t) + 200 + NormalNoise(0, t/3)  
    """
    noise = t/3 * np.random.randn(t.shape[0])
    f = lambda t: t * np.sin(t) + 200
    vfunc = np.vectorize(f)
    f_time = vfunc(t)
    return (f_time+noise).reshape(-1,1)

def cyclical_component_2(t, period):
    """
    cyclical_component_2 = 100 * sin(t // period) + 2*t
    """
    f = lambda t: 100 * np.sin(t // period) + 100 + 2*t
    vfunc = np.vectorize(f)
    f_time = vfunc(t)
    return f_time.reshape(-1,1)

def target_ts(features):
    """
    target_ts = averaged sum of features
    
    returns: y = 1/n * Sum  exogenous_i
    """
    a = (np.ones(features.shape[1])/features.shape[1]).reshape(-1,1)
    y = features @ a
    return y
t = np.arange(0, 200)
exog = pd.DataFrame(
    np.concatenate([linear_trend(t, 0.3), cyclical_component_1(t), cyclical_component_2(t, period=2)], axis = 1),
    columns=['linear_trend', 'cyclical_component_1', 'cyclical_component_2'])

y = pd.DataFrame(target_ts(exog.values), columns=['y'])
fig, ((ax1, ax2, ax3,)) = plt.subplots(1, 3, figsize=(10, 5))
fig.subplots_adjust(top=2, bottom=1, right=2, left=0, wspace=0.2)

ax1.plot(t, exog['linear_trend'])
ax1.set_title("Exogenous feature 1: linear_trend", fontsize=20)

ax2.plot(t, exog['cyclical_component_1'])
ax2.set_title('Exogenous feature 2: cyclical_component_1', fontsize=20)

ax3.plot(t, exog['cyclical_component_2'])
ax3.set_title('Exogenous feature 3: cyclical_component_2', fontsize=20)
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(t, y['y'])
plt.title('Target: average of features', fontsize=20)
plt.show()
y["ds"] = pd.date_range(start='1/1/2018', periods=len(y), freq='M')
y.head()
exog["ds"] = pd.date_range(start='1/1/2018', periods=len(exog), freq='M')
exog.head()
N = 100

y_train = y.iloc[:N]
y_test = y.iloc[N:]

exog_train = exog.iloc[:N]
exog_test = exog.iloc[N:]
del y_test["y"]
y_train.head()
exog_train.head()
data_train_joined_2_regressor = pd.merge(y_train, exog_train[['linear_trend', 'cyclical_component_1', 'ds']], on = "ds")
data_test_joined_2_regressor = pd.merge(y_test, exog_test[['linear_trend', 'cyclical_component_1', 'ds']], on = "ds")

# Take first 3 points for prediction
data_test_joined_2_regressor = data_test_joined_2_regressor[:10].copy()

print('Train shape: {0}'.format(data_train_joined_2_regressor.shape))
print("Test shape: {0}".format(data_test_joined_2_regressor.shape))
data_train_joined_2_regressor.head()
data_test_joined_2_regressor
from fbprophet import Prophet

model = Prophet(n_changepoints=1)
model.add_seasonality(name='yearly', period=365, fourier_order=1, prior_scale=0.1, mode='multiplicative') 
model.add_seasonality(name='monthly', period=30, fourier_order=1, prior_scale=0.1, mode='multiplicative')
model.add_regressor('linear_trend', mode="multiplicative")
model.add_regressor('cyclical_component_1', mode="additive")
model.fit(data_train_joined_2_regressor)
forecast = model.predict(data_test_joined_2_regressor)
forecast.head()
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
model.params
df_train = model.setup_dataframe(data_train_joined_2_regressor)
df_train.head(5)
df_test = model.setup_dataframe(data_test_joined_2_regressor)
df_test.head(5)
# Start of the time
start = data_train_joined_2_regressor["ds"].min()

# Timedelta to the end of training data
t_scale = data_train_joined_2_regressor["ds"].max() - start 

# scaled time to the range [0, 1]
t = ((data_test_joined_2_regressor["ds"] - start) / t_scale).values
print("Standartized time: {0}".format(t))
print("Standartized time by prophet: {0}".format(df_test["t"].values))
mu_train = data_train_joined_2_regressor['linear_trend'].mean()
std_train = data_train_joined_2_regressor['linear_trend'].std()

print("Standartized f1 regressor by our calculations: {0}".format(
    ((data_test_joined_2_regressor['linear_trend'] - mu_train) / std_train).values))
print("Standartized f1 regressor by prophet: {0}".format(df_test["linear_trend"].values))
changepoint_ts = model.changepoints_t
print("Changepoints: {0}".format(changepoint_ts))
# Just extracted the indices of the minimum and maximum
i0, i1 = df_train['ds'].idxmin(), df_train['ds'].idxmax()

# Calculate the time difference for a standardized time
T = df_train['t'].iloc[i1] - df_train['t'].iloc[i0] 

# Calculate k
k = (df_train['y_scaled'].iloc[i1] - df_train['y_scaled'].iloc[i0]) / T

# Calculate m
m = df_train['y_scaled'].iloc[i0] - k * df_train['t'].iloc[i0]

print("initial value for k: {0}".format(k))
print("initial value for m: {0}".format(m))
k = np.nanmean(model.params['k'])
m = np.nanmean(model.params['m'])

print("Value for k after stan optimization: {0}".format(k))
print("Value for m after stan optimization: {0}".format(m))
deltas = np.nanmean(model.params['delta'], axis = 0)
gammas = -changepoint_ts * deltas
gammas = -changepoint_ts * deltas

k_t = k * np.ones_like(t)
m_t = m * np.ones_like(t)

for s, t_s in enumerate(changepoint_ts):
    indx = t >= t_s
    k_t[indx] += deltas[s]
    m_t[indx] += gammas[s]

trend = k_t * t + m_t
floor = 0
trend = trend * model.y_scale + floor
print("Manually calculated trend: {0}".format(trend))
print("Trend calculated by Prophet: {0}".format(forecast.trend.values))
seasonal_components = model.predict_seasonal_components(df_test)
seasonal_components.head()
seasonal_features, _, component_cols, _ = model.make_all_seasonality_features(df_test)
seasonal_features.head()
t = np.array(
    (data_test_joined_2_regressor["ds"] - pd.datetime(1970, 1, 1))
    .dt.total_seconds()
    .astype(np.float)
) / (3600 * 24.)
print("Time in month passed from 01.01.1970 for every test point: {0}".format(t))
model.seasonalities
period = model.seasonalities['yearly']["period"]
series_order = model.seasonalities["yearly"]["fourier_order"]

np.column_stack([
    fun((2.0 * (i + 1) * np.pi * t / period))
    for i in range(series_order)
    for fun in (np.sin, np.cos)
])
component_cols
# Extract the data
X = seasonal_features.values

# For each seasonal component we calculate
for component in component_cols.columns:
    
    # Here, in fact, we leave the beta non-zero only in those positions where we have 1
    # This is equivalent to the fact that we just made up a mask so that later in the matrix seasonal_features take
    # only the parameters we need
    beta_c = model.params['beta'] * component_cols[component].values
    
    # This is a basic dot product, that is, for each date it's easy
    # multiply the scaled value by the trained beta and add
    # only the parameters that we need
    comp = np.matmul(X, beta_c.transpose())
    
    # If we have additive feature, then we should scale, that is also obvious
    if component in model.component_modes['additive']:
         comp *= model.y_scale
    
    # Exactly this will be in  'multiplicative' and 'additive'
    df_test[component] = np.nanmean(comp, axis=1)
df_test
additive_terms = model.params["beta"].squeeze()[5] * seasonal_features["cyclical_component_1"] * model.y_scale
print("Manually calculated additive terms: {0}".format(additive_terms.values))
print("Additive terms calculated by Prophet: {0}".format(
    seasonal_components["additive_terms"].values))
multiplicative_terms = (model.params["beta"].squeeze()[0] * seasonal_features["yearly_delim_1"] + 
        model.params["beta"].squeeze()[1] * seasonal_features["yearly_delim_2"] + 
        model.params["beta"].squeeze()[2] * seasonal_features["monthly_delim_1"] + 
        model.params["beta"].squeeze()[3] * seasonal_features["monthly_delim_2"] + 
        model.params["beta"].squeeze()[4] * seasonal_features["linear_trend"])
print("Manually calculated multiplicative terms: {0}".format(multiplicative_terms.values))
print("Multiplicative terms calculated by Prophet: {0}".format(
    seasonal_components["multiplicative_terms"].values))
prediction = trend * (1 + multiplicative_terms) + additive_terms
print("Manually calculated prediction: {0}".format(prediction.values))
forecast = model.predict(data_test_joined_2_regressor)
print("Prediction calculated by Prophet: {0}".format(forecast["yhat"].values))