# Libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.python.client import device_lib

# To eliminate chained assignment warnings.
pd.options.mode.chained_assignment = None

# For plotting in the notebook itself.
%matplotlib inline

# Changing the style and size of plots.
sns.set(style='darkgrid', palette='muted')
rcParams['figure.figsize'] = 16, 8

# Generating pseudi-random number for repeatable outputs.
np.random.seed(1)
tf.random.set_seed(1)

# Printing the tensorslow version.
print('Tensorflow version:', tf.__version__)
# Checking for available hardware for model training and printing it.
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())
# Importing CSV file in DataFrame and printing first few values.
df = pd.read_csv('../input/fandata.csv')
df.head()
# Interactive plotting of amplitude vs time.
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.time, y=df.amplitude,
                    mode='lines',
                    name='close'))
fig.update_layout(showlegend=True)
fig.show()
# 80% of data used for training & rest for testing purpose.
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

# Printing the shapes for train & test data (rows, columns).
print(train.shape, test.shape)
# First, fitting weights to the scaler.
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(train[['amplitude']])

# Second, transforming train & test data using the scaler.
train['amplitude'] = scaler.transform(train[['amplitude']])
test['amplitude'] = scaler.transform(test[['amplitude']])
# Function for data-reshaping.
def create_dataset(X, y, time_steps = 1):
    Xs, ys = [], []
    
    # Looping within the lenght of dataframe minus time steps.
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)       
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
# Defining time steps for our model.
time_steps = 30

# Creating datasets for training & testing purpose.
X_train, y_train = create_dataset(train[['amplitude']], train.amplitude, time_steps)
X_test, y_test = create_dataset(test[['amplitude']], test.amplitude, time_steps)

# Printing the shape of training data.
print(X_train.shape, y_train.shape)
# Storing the number of timesteps & features.
timesteps = X_train.shape[1]
num_features = X_train.shape[2]

# Printing the same for verification.
timesteps, num_features
# Clearing any backend session for a brand new training.
tf.keras.backend.clear_session()

# Defining the neural network topology or in simple terms, the design network.
# Creating an instance of a Sequential model.
model = Sequential([
    
    # Adding LSTM Layers.
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    LSTM(50, input_shape = (timesteps, num_features), return_sequences=True),
    
    # Adding the final feed-forward layer to complete the autoencoder.
    Dense(num_features)                 
])

# Compiling model with loss & optimizer parameter.
model.compile(loss='mae', optimizer='adam')

# Printing model summary for more information.
model.summary()
# Creating the keras callback.
es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min')

# Fitting the model to training data.
history = model.fit(
    
    # Passing training data.
    X_train, y_train,
    
    # Setting initial number of epochs.
    epochs = 50,
    
    # Setting batch size hyper-parameter.
    batch_size = 72,
    
    # Using 10% of training data for validation & rest 90% for actual training.
    validation_split = 0.1,
    
    # Parsing the callback array while training.
    callbacks = [es],
    
    # Not shuffling the data, since order for time-series data matters.
    shuffle = False
)
# Plotting training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend();
# Evaluating the model for test data in MAE
model.evaluate(X_test, y_test)
# Calling predictions for training data
X_train_pred = model.predict(X_train)

# Calculating MAE for the predictions done on training data and storing that in a Pandas DataFrame.
train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train), axis = 1), columns = ['Error'])
# Looking at the distribution of error column for training data
sns.distplot(train_mae_loss, bins = 50, kde = True);
# Calling predictions for testing data.
X_test_pred = model.predict(X_test)

# Calculating MAE for the predictions done on testing data.
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis = 1)
# Looking at the distribution of errors for testing data
sns.distplot(test_mae_loss, bins = 50, kde = True);
# Setting the loss threshold for anamoly detection.
THRESHOLD = 0.055

# Creating a new pandas dataframe for storing the following.
test_score_df = pd.DataFrame(test[time_steps:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['amplitude'] = test[time_steps:].amplitude
# Printing first few values.
test_score_df.head()
# Checking the number of anamolies.
test_score_df.anomaly.value_counts()
# Interactive plotting for test values with an overlay line showing threshold.
fig = go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:].time, y=test_score_df.loss,
                    mode='lines',
                    name='Test Loss'))
fig.add_trace(go.Scatter(x=test[time_steps:].time, y=test_score_df.threshold,
                    mode='lines',
                    name='Threshold'))
fig.update_layout(showlegend=True)
fig.show()
# Creating a new DataFrame with only the anamolies & printing few values to verify.
anomalies = test_score_df.query('anomaly == True')
anomalies.head()
# Plotting the inversely transformed data with an overlay of anomalies points for test data.
# Inverse transform is done to obtain the original values.
fig = go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:].time, y=scaler.inverse_transform([test[time_steps:].amplitude]).flatten(),
                    mode='lines',
                    name='Amplitudes'))
fig.add_trace(go.Scatter(x=anomalies.time, y=scaler.inverse_transform([anomalies.amplitude]).flatten(),
                    mode='markers',
                    name='Anomaly'))
fig.update_layout(showlegend=True)
fig.show()
