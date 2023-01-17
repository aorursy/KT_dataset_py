import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import Adadelta
from keras import regularizers

from sklearn.preprocessing import PolynomialFeatures
train_df = pd.read_csv('../input/train.csv', nrows = 10 ** 6)
test_df = pd.read_csv('../input/test.csv')
datasets = [train_df, test_df]

for df in datasets:
    missing_values = df.isnull().sum().to_frame().sort_values(0, ascending = False)
    display(missing_values.head())
print("Train before cleaning:")
display(train_df.describe())

train_df = train_df.dropna(how = 'any', axis = 'rows')
train_df = train_df[(train_df.pickup_longitude > -75.0) & (train_df.pickup_longitude < -73.0)]
train_df = train_df[(train_df.pickup_latitude > 40.0) & (train_df.pickup_latitude < 42.0)]
train_df = train_df[(train_df.dropoff_longitude > -75.0) & (train_df.dropoff_longitude < -73.0)]
train_df = train_df[(train_df.dropoff_latitude > 40.0) & (train_df.dropoff_latitude < 42.0)]
train_df = train_df[(train_df.passenger_count > 0.0) & (train_df.passenger_count <= 6.0)]


print("Train after cleaning:")
display(train_df.describe())

print("Test for comparison:")
display(test_df.describe())
def calc_haversine(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

    df['dlat'] = np.radians(df.dropoff_latitude - df.pickup_latitude)
    df['dlon'] = np.radians(df.dropoff_longitude - df.pickup_longitude)
    df['haversine_a'] = np.sin(df.dlat/2) * np.sin(df.dlat/2) + np.cos(np.radians(df.pickup_latitude)) \
            * np.cos(np.radians(df.dropoff_latitude)) * np.sin(df.dlon/2) * np.sin(df.dlon/2)
    df['haversine'] = 6371 * 2 * np.arctan2(np.sqrt(df.haversine_a), np.sqrt(1-df.haversine_a))

    return df.drop(columns=['pickup_datetime'])

train_df = calc_haversine(train_df)
test_df = calc_haversine(test_df)
corr = train_df.corr()
f, ax = plt.subplots(figsize=(10, 10)) 
cmap = sns.diverging_palette(220, 10, as_cmap=True) 
sns.heatmap(corr, cmap=cmap, vmax=1.0, square=True, linewidths=.3, cbar_kws={"shrink": .5}, ax=ax) 
plt.show()
# filter interesting columns and label
train_y = np.array(train_df['fare_amount'])
train_X = train_df.drop(columns=['fare_amount','key'])

print("Shape for X:")
print(train_X.shape)
print("Shape for Y:")
print(train_y.shape)

test_X = test_df.drop(columns=['key'])
print("Shape for test X:")
print(test_X.shape)
def run_model(X, Y, dnn_layers_size, dropout_value, batch_size, epochs):
    
    input_size = X.shape[1]
    
    model = Sequential()
    
    for idx, l in enumerate(dnn_layers_size):
        model.add(Dense(l, input_dim=input_size,
                           kernel_initializer='normal',
                           activation='selu'))
        model.add(Dropout(dropout_value))
        input_size = l
        
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    train_history = model.fit([X], Y, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=True)
    
    return train_history, model

def build_layers(layers, n_features):
    if len(layers) == 0:
        n_features = int(n_features * 2.5)
    else:
        n_features = int(math.sqrt(n_features))
        
    if n_features < 3:
        return layers
    else:
        layers.append(n_features)
        return build_layers(layers, n_features)
    
def plot_build(train_history):    
    
    # plotting train_history
    plt.figure(0)
    axes = plt.gca()
    axes.set_ylim([0,90])
    plt.plot(train_history.history['loss'],'g')
    plt.plot(train_history.history['val_loss'],'b')
    plt.rcParams['figure.figsize'] = (8, 6) 
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.grid()
    plt.legend(['train','validation'])

    plt.show()
layers = build_layers([],train_X.shape[1])
print('Layers:', layers)
print('-' * 15)
train_history, model = run_model(train_X, train_y, layers, 0.2, batch_size = 32, epochs = 100)
plot_build(train_history)
# Generating DNN submission
pred_y = model.predict([test_X])
test_df['pred'] = pred_y

submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_df.pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission_dnn.csv', index = False)

print(os.listdir('.'))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as msq
import xgboost as xgb

stdscaler = StandardScaler()
xgb_train_X = stdscaler.fit_transform(train_X)
xgb_test_X  = stdscaler.fit_transform(test_X)

x_train, x_test, y_train, y_test = train_test_split(xgb_train_X, train_y, random_state=70, test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'},
                    dtrain=matrix_train,
                    num_boost_round=100, 
                    early_stopping_rounds=10,
                    evals=[(matrix_test,'test')])
    return model

xgb_model = XGBmodel(x_train,x_test,y_train,y_test)
xgb_pred = xgb_model.predict(xgb.DMatrix(xgb_test_X), ntree_limit = xgb_model.best_ntree_limit)

# Generating XGB submission
test_df['pred_xgb'] = xgb_pred

submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_df.pred_xgb},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission_xgb.csv', index = False)

print(os.listdir('.'))
test_df['ensemble'] = (test_df.pred + test_df.pred_xgb) / 2.0

submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_df.ensemble},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission_ensemble.csv', index = False)

print(os.listdir('.'))
