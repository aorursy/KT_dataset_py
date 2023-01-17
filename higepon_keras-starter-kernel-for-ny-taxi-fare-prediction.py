# Import ML libraries.
import csv as csv
import os as os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import Callback
from keras.models import Sequential
from numpy.random import seed
from tensorflow import set_random_seed
from IPython.display import clear_output
# Stablize our result by setting fixed seed.
seed(1)
set_random_seed(2)
train_df =  pd.read_csv("../input/train.csv", nrows = 50000, parse_dates=["pickup_datetime"])
train_df.dtypes
# Credit: dimitreoliveira
# https://www.kaggle.com/dimitreoliveira/tensorflow-dnn-coursera-ml-course-tutorial
def clean(df):
    # Drop null values.
    df = df.dropna(how = 'any', axis = 'rows')
    
    # Delimiter lats and lons to NY only
    df = df[(-76 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72)]
    df = df[(-76 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72)]
    df = df[(38 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 42)]
    df = df[(38 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 42)]
    # Remove possible outliers
    df = df[(0 < df['fare_amount']) & (df['fare_amount'] <= 250)]
    # Remove inconsistent values
    df = df[(df['dropoff_longitude'] != df['pickup_longitude'])]
    df = df[(df['dropoff_latitude'] != df['pickup_latitude'])]
    return df

# Process data frame.
def process_df(df):
    # Squared distance between pickup and dropoff location.
    df["sq_distance"] = (df.pickup_longitude - df.dropoff_longitude) ** 2 + (df.pickup_latitude - df.dropoff_latitude) ** 2
    
    # Month of pickup date. We think there should be busy season.
    df["month"] = df.pickup_datetime.apply(lambda t: t.month)
    
    # Hour of pickup date. We thinks there should be busy time. eg) Commute time.
    df["hour"] = df.pickup_datetime.apply(lambda t: t.hour)
    
    # Weekday of pickup date. eg) Weekend may be busy.
    df["weekday"] = df.pickup_datetime.apply(lambda t: t.weekday())
    return df

train_df = clean(train_df)
train_df = process_df(train_df)
train_df.head()
# Modified https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.acc = []
        self.val_acc = []

        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        
        self.i += 1
        
        clear_output(wait=True)
        plt.subplot(1, 2, 2)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.subplot(1, 2, 1)
        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x, self.val_acc, label="val_acc")
        plt.legend()
        
        plt.show();
        
plot_losses = PlotLosses()
# Make a simple 4 layer NN.
#features = ["sq_distance", "month", "hour", "passenger_count", "weekday"]
features = ["sq_distance"]


model = Sequential()
model.add(Dense(units=32, input_dim=len(features)))
model.add(Dense(units=24, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(loss="mse",  optimizer="adam",  metrics=["accuracy"])
x_train = train_df[features].values
y_train = train_df.fare_amount.values

train_df[features].head()
# Training
model.fit(x_train, y_train, epochs=200, batch_size=512, validation_split=0.1, callbacks=[plot_losses], verbose=0)
# Load and process the test data
test_df =  pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])
test_df = process_df(test_df)
x_test = test_df[features].values
key = test_df["key"].values
# Prediction
target = np.round(model.predict(x_test, batch_size=32))
submission = pd.DataFrame(
    {'key': key, 'fare_amount': target[:, 0]},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)
print(os.listdir('.'))