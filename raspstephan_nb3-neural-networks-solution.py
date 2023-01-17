import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
df_train = pd.read_csv('/kaggle/input/weather-postprocessing/pp_train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/weather-postprocessing/pp_test.csv', index_col=0)

X_train = pd.read_csv('/kaggle/input/nb1-linear-regression/X_train.csv', index_col=0)
y_train = pd.read_csv('/kaggle/input/nb1-linear-regression/y_train.csv', index_col=0, squeeze=True)
X_valid = pd.read_csv('/kaggle/input/nb1-linear-regression/X_valid.csv', index_col=0)
y_valid = pd.read_csv('/kaggle/input/nb1-linear-regression/y_valid.csv', index_col=0, squeeze=True)
X_test = pd.read_csv('/kaggle/input/nb1-linear-regression/X_test.csv', index_col=0)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import mean_squared_error
X_train1 = X_train[['t2m_fc_mean']].values
X_train1.shape
model = Sequential([Dense(1, input_shape=(1,))])
model.summary()
model.layers[0].weights
a, b = [p.numpy().squeeze() for p in model.layers[0].weights]
a, b
plt.scatter(X_train1[::1000], y_train[::1000], alpha=0.2)
x = np.linspace(-15, 20, 2)
plt.plot(x, a*x+b, c='r', lw=2)
preds = model(X_train1)  # Same as model.predict(X_train1)
y_train.shape, preds.shape
mse = mean_squared_error(y_train, preds[:, 0])
mse
e = preds[:, 0] - y_train
dl_da = np.mean(2 * X_train1[:, 0] * e)
dl_db = np.mean(2 * e)
dl_da, dl_db
with tf.GradientTape() as g:
    preds = model(X_train1)
    mse = mean_squared_error(y_train, preds[:, 0])
dloss_dparam = g.gradient(mse, model.trainable_weights)
dl_da, dl_db = dloss_dparam
dl_da, dl_db
lr = 1e-3
a, b = model.get_weights()
a -= lr * dl_da
b -= lr * dl_db
model.set_weights([a, b])
a, b = [p.numpy().squeeze() for p in model.layers[0].weights]
plt.scatter(X_train1[::1000], y_train[::1000], alpha=0.2)
x = np.linspace(-15, 20, 2)
plt.plot(x, a*x+b, c='r', lw=2)
preds = model(X_train1)
mse = mean_squared_error(y_train, preds[:, 0])
mse
def gradient_descent_step(model, lr):
    with tf.GradientTape() as g:
        preds = model(X_train1)
        mse = mean_squared_error(y_train, preds[:, 0])
    dl_da, dl_db = g.gradient(mse, model.trainable_weights)
    a, b = model.get_weights()
    a -= lr * dl_da
    b -= lr * dl_db
    model.set_weights([a, b])
    return (a, b), mse.numpy()
model = Sequential([Dense(1, input_shape=(1,))])
from IPython.display import clear_output
from time import sleep
def plot_line():
    a, b = [p.numpy().squeeze() for p in model.layers[0].weights]
    plt.scatter(X_train1[::1000], y_train[::1000], alpha=0.2)
    x = np.linspace(-15, 20, 2)
    plt.plot(x, a*x+b, c='r', lw=2)
    plt.show()
params = []
loss = []
for _ in range(20):
    p, l = gradient_descent_step(model, 1e-3)
    params.append(p)
    loss.append(l)
    plot_line()
    sleep(0.2)
    clear_output(True)
loss[-1]
plt.plot(loss);
model = Sequential([Dense(1, input_shape=(1,))])
model.compile(tf.keras.optimizers.SGD(1e-3), 'mse')
model.fit(
    X_train1, 
    y_train.values, 
    epochs=20,
    batch_size=len(X_train1)
)
model = Sequential([Dense(1, input_shape=(1,))])
model.compile(tf.keras.optimizers.SGD(1e-3), 'mse')
model.fit(
    X_train1, 
    y_train.values, 
    batch_size=128,
    epochs=2,
)
X_train.shape, y_train.shape
model = Sequential([Dense(1, input_shape=(22,))])
model.compile(tf.keras.optimizers.SGD(1e-4), 'mse')
model.summary()
model.fit(
    X_train.values,
    y_train.values,
    validation_data=(X_valid.values, y_valid.values)
)
X_train.std().plot.bar()
plt.yscale('log')
mean = X_train.mean()
std = X_train.std()
X_train_norm = ((X_train - mean) / std).values
X_valid_norm = ((X_valid - mean) / std).values
X_test_norm = ((X_test - mean) / std).values
y_train = y_train.values
y_valid = y_valid.values
X_train_norm.std(0)
model = Sequential([Dense(1, input_shape=(22,))])
model.compile(tf.keras.optimizers.Adam(1e-3), 'mse')
model.fit(
    X_train_norm,
    y_train,
    batch_size=128,
    epochs=4,
    validation_data=(X_valid_norm, y_valid)
)
from sklearn.metrics import r2_score, mean_squared_error
def print_scores(model, X_train=X_train_norm, X_valid=X_valid_norm):
    preds_train = model.predict(X_train, batch_size=10_000)
    preds_valid = model.predict(X_valid, batch_size=10_000)
    r2_train = r2_score(y_train, preds_train)
    r2_valid = r2_score(y_valid, preds_valid)
    mse_train = mean_squared_error(y_train, preds_train)
    mse_valid = mean_squared_error(y_valid, preds_valid)
    print(f'Train R2 = {r2_train}\nValid R2 = {r2_valid}\nTrain MSE = {mse_train}\nValid MSE = {mse_valid}')
print_scores(model)
model = Sequential([
    Dense(32, input_shape=(22,), activation='relu'),
    Dense(1, activation='linear')
])
model.compile(tf.keras.optimizers.Adam(1e-3), 'mse')
model.fit(X_train_norm, y_train, 1024, epochs=12, validation_data=(X_valid_norm, y_valid))
print_scores(model)
model = Sequential([
    Dense(256, input_shape=(22,), activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='linear')
])
model.summary()
model.compile(tf.keras.optimizers.Adam(1e-4), 'mse')
model.summary()
h = model.fit(X_train_norm, y_train, 1024, epochs=30, validation_data=(X_valid_norm, y_valid))
plt.plot(h.history['loss'][1:])
plt.plot(h.history['val_loss'][1:])
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
model = Sequential([
    Dense(256, input_shape=(22,), activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(tf.keras.optimizers.Adam(1e-4), 'mse')
model.fit(X_train_norm, y_train, 1024, epochs=30, validation_data=(X_valid_norm, y_valid), callbacks=[early_stopping])
print_scores(model)
preds = model.predict(X_test, 10_000).squeeze()
sub =  pd.DataFrame({'id': range(len(preds)), 'Prediction': preds})
sub.to_csv('submission1.csv', index=False)
split_date = '2015-01-01'
df_train = df_train.dropna(subset=['t2m_obs'])
stations = df_train.station
stations_train = df_train.station[df_train.time < split_date]
stations_valid = df_train.station[df_train.time >= split_date]
stations_test = df_test.station
stations_train.head()
unique_stations = pd.concat([df_train.station, df_test.station]).unique()
len(unique_stations)
stat2id = {s: i for i, s in enumerate(unique_stations)}
ids = stations.apply(lambda x: stat2id[x])
ids_train = ids[df_train.time < split_date]
ids_valid = ids[df_train.time >= split_date]
ids_test = stations_test.apply(lambda x: stat2id[x])
ids_train.head()
features_in = Input(shape=(22,))
id_in = Input(shape=(1,))
emb_layer = Embedding(len(unique_stations), 2)
emb = emb_layer(id_in)
emb_layer.get_weights()[0].shape
id_in, emb
emb = Flatten()(emb)
emb
x = Concatenate()([features_in, emb])
x
x = Dense(100, activation='relu')(x)
out = Dense(1, activation='linear')(x)
model = tf.keras.models.Model(inputs=[features_in, id_in], outputs=out)
model.summary()
tf.keras.utils.plot_model(model)
model.compile(tf.keras.optimizers.Adam(1e-3), 'mse')
model.fit([X_train_norm, ids_train], y_train, 1024, 20, 
          validation_data=([X_valid_norm, ids_valid], y_valid))
print_scores(model, X_train=[X_train_norm, ids_train], X_valid=[X_valid_norm, ids_valid])
preds = model.predict([X_test, ids_test], 10_000).squeeze()
sub =  pd.DataFrame({'id': range(len(preds)), 'Prediction': preds})
sub.to_csv('submission2.csv', index=False)
e = emb_layer.get_weights()[0]
cat = pd.concat([df_train, df_test])
lats = [cat[cat.station == s].iloc[0].station_lat for s in unique_stations]
lons = [cat[cat.station == s].iloc[0].station_lon for s in unique_stations]
e.shape, len(lats), len(lons)
plt.scatter(lons, lats, c=e[:, 0], cmap='jet')
plt.scatter(lons, lats, c=e[:, 1], cmap='jet')
