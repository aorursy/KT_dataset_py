import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set(font_scale=1.6)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()

news_train_df['subjects'] = news_train_df['subjects'].str.findall(f"'([\w\./]+)'")
news_train_df['audiences'] = news_train_df['audiences'].str.findall(f"'([\w\./]+)'")
news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'")

(market_train_df.shape, news_train_df.shape)
# We will reduce the number of samples for memory reasons
toy = False

if toy:
    market_train_df = market_train_df.tail(100_000)
    news_train_df = news_train_df.tail(300_000)
else:
    market_train_df = market_train_df.tail(3_000_000)
    news_train_df = news_train_df.tail(6_000_000)

(market_train_df.shape, news_train_df.shape)
def generator(data, lookback, min_index, max_index,
              shuffle=False, batch_size=128, step=1):
    if max_index is None:
        max_index = data.shape[0] - 1
        
    i = min_index + lookback
    while True:
        # Select the rows that will be used for the batch
        if shuffle:
            rows = np.random.randint(min_index + lookback, size=batch_size)
        else:
            rows = np.arange(i, min(i + batch_size, max_index))
            i += rows.shape[0]
            if i + batch_size >= max_index:
                i = min_index + lookback
                
        samples = np.zeros((len(rows),
                   lookback // step,
                   data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j]][1]
            
        yield samples, targets
from keras.callbacks import Callback

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop, Adagrad

lookback = 30 
batch_size = 1024
steps_per_epoch = 100
epochs = 10
data_split = 0.8
step = 1

def generators(market_data_float):
    l = market_data_float.shape[0]
    train_gen = generator(market_data_float,
                          min_index=0,
                          max_index=int(data_split * l),
                          batch_size=batch_size,
                          lookback=lookback,
                          step=step)
    
    val_gen = generator(market_data_float,
                        min_index=int(data_split * l),
                        max_index=None,
                        batch_size=batch_size,
                        lookback=lookback,
                        step=step)
    
    return (train_gen, val_gen)

def learn_model(market_data_float):
    (train_gen, val_gen) = generators(market_data_float)
    input_shape = (None, market_data_float.shape[-1])
    
    model = Sequential()
    model.add(layers.GRU(4, input_shape=input_shape))   
    model.add(layers.Dense(1))
    model.compile(optimizer=Adagrad(), loss='mae')
    
    callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1)]
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=10,
                                  validation_data=val_gen,
                                  validation_steps=100,
                                  callbacks=callbacks)
    
    return (model, history)

def learn_models(market_train_df, Histories):
    for asset_code, market_data in tqdm(market_train_df.groupby('assetCode')):
        # drop the non-numeric columns and handle the nans.
        market_float_data = market_data.drop(['assetCode', 'assetName', 'time'], axis=1).fillna(0)
        
        # normalize the data
        scaler = StandardScaler().fit(market_float_data)
        
        # learn a model using the normalized data
        (model, history) = learn_model(scaler.transform(market_float_data))
        
        # save the history
        Histories[asset_code] = history
        yield asset_code, (scaler, model)
n = 5
n_random_assets = np.random.choice(market_train_df.assetCode.unique(), n)

market_train_sampled_df = market_train_df[market_train_df.assetCode.isin(n_random_assets)]
(fig, ax) = plt.subplots(figsize=(15, 8))

market_train_sampled_df.groupby('assetCode').plot(x='time', y='close', ax=ax)

ax.legend(n_random_assets)
plt.xlabel('time')
plt.ylabel('close')
plt.title('Closing Price of %i random assets' % n)
plt.show()
Histories = {}
Models = dict(learn_models(market_train_sampled_df, Histories))
for asset, history in Histories.items():
    (fig, ax) = plt.subplots(figsize=(15, 8))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1) 
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title(asset)
    plt.legend()
    plt.show()