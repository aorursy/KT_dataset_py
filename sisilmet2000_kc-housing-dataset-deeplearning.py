from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from math import radians, cos, sin, asin, sqrt
import pydot
import seaborn as sns
import keras
from keras import metrics
from keras import regularizers
from keras.models import Model, Sequential, load_model
from keras.layers import Embedding, Flatten, Input, Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
kc_raw_data = pd.read_csv('../input/kc_house_data.csv')
list(kc_raw_data.columns)
kc_raw_data['sale_yr'] = pd.to_numeric(kc_raw_data.date.str.slice(0,4))
kc_raw_data['sale_month'] = pd.to_numeric(kc_raw_data.date.str.slice(4,6))
kc_raw_data['sale_day'] = pd.to_numeric(kc_raw_data.date.str.slice(6,8))
kc_raw_data['house_age'] = kc_raw_data['sale_yr'] - kc_raw_data['yr_built']
FIXED_LONG = -122.213896
FIXED_LAT = 47.560053
kc_raw_data['distance'] = kc_raw_data.apply(lambda row: haversine(FIXED_LONG, FIXED_LAT, row['long'], row['lat']), axis=1)
kc_raw_data['greater_long'] = (kc_raw_data['long'] >= FIXED_LONG).astype(int)
kc_raw_data['less_long'] = (kc_raw_data['long'] < FIXED_LONG).astype(int)
kc_raw_data['greater_lat'] = (kc_raw_data['lat'] >= FIXED_LAT).astype(int)
kc_raw_data['less_lat'] = (kc_raw_data['lat'] < FIXED_LAT).astype(int)

kc_data = pd.DataFrame(kc_raw_data, columns=[
        'sale_yr','sale_month','sale_day','house_age', 'distance', 'greater_long', 'less_long',
        'greater_lat','less_lat', 'view', 'waterfront',
        'bedrooms','bathrooms','sqft_living','sqft_lot','floors',
        'condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
        'zipcode','sqft_living15','sqft_lot15','price'])

label_column = 'price'

#embedding code change
zipcode_constant = 98000
kc_data['zipcode'] = kc_data['zipcode'] - zipcode_constant
kc_data = kc_data.sample(frac=1)
kc_data.describe()

train = kc_data.sample(frac=0.8)
'Train:' + str(train.shape)
validate = kc_data.sample(frac=0.1)
'Validate:' + str(validate.shape)
test = kc_data.sample(frac=0.1)
'Test:' + str(test.shape)

cols = list(kc_data.columns)
cols.remove(label_column)
cols.remove('zipcode')

train_x = train[cols]
train_y = train[label_column]
train_zipcode_x = train['zipcode']
train_x_mean = train_x[cols].mean(axis=0)
train_x_std = train_x[cols].std(axis=0)

validate_x = validate[cols]
validate_y = validate[label_column]
validate_zipcode_x = validate['zipcode']

test_x = test[cols]
test_y = test[label_column]
test_zipcode_x = test['zipcode']

# zscores
train_x[cols] = (train_x[cols] - train_x_mean) / train_x_std
validate_x[cols] = (validate_x[cols] - train_x_mean) / train_x_std
test_x[cols] = (test_x[cols] - train_x_mean) / train_x_std
def model3(train_x, train_zipcode_t, train_y):
    data_zipcode = train_zipcode_x
    data_main = train_x
    data_y = train_y

    zipcode_input = Input(shape=(1,), dtype='int32', name='zipcode_input')
    x = Embedding(output_dim=5, input_dim=200, input_length=1)(zipcode_input)
    zipcode_out = Flatten()(x)
    zipcode_output = Dense(1, activation='relu', name='zipcode_model_out')(zipcode_out)

    main_input = Input(shape=(data_main.shape[1],), name='main_input')
    lyr = keras.layers.concatenate([main_input, zipcode_out])
    lyr = Dense(100, activation="relu")(lyr)
    lyr = Dense(50, activation="relu")(lyr)
    main_output = Dense(1, name='main_output')(lyr)

    t_model = Model(
        inputs=[main_input, zipcode_input], 
        outputs=[main_output, zipcode_output]
    )
    t_model.compile(
        loss="mean_squared_error",
        optimizer=Adam(lr=0.001),
        metrics=[metrics.mae],
        loss_weights=[1.0, 0.5]
    )
    return t_model

keras_callbacks = [
    # ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=2)
    # ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0)
    # TensorBoard(log_dir='/tmp/keras_logs/model_3', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),
    # EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)
]
model = model3(train_x, train_zipcode_x, train_y)
print(model.summary())
plot_model(model, to_file='kc_housing.png', show_shapes=True, show_layer_names=True)
epochs = 500
batch = 128

cols = list(train.columns)
cols.remove(label_column)
history = model.fit(
    [train_x, train_zipcode_x], [train_y, train_y],
    batch_size=batch,
    epochs=epochs,
    shuffle=True,
    verbose=0,
    validation_data=([validate_x, validate_zipcode_x],[validate_y, validate_y]),
    callbacks=keras_callbacks
)
score = model.evaluate([test_x, test_zipcode_x], [test_y, test_y], verbose=0)
from sklearn.manifold import TSNE
import seaborn as sns

zipcode_embeddings = model.layers[1].get_weights()[0]
labels = train_zipcode_x
zipcode_embeddings.shape
tsne_model = TSNE(perplexity=200, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(zipcode_embeddings)

x1 = []
y1 = []
avg_price1 = []
x2 = []
y2 = []
avg_price2 = []
for index, value in enumerate(train_zipcode_x):
    zipcode = train_zipcode_x.iloc[index]
    price = train_y.iloc[index]
    if price > 0:
        avg_price1.append(price)
        x1.append(new_values[zipcode][0])
        y1.append(new_values[zipcode][1])
    else:
        avg_price2.append(price)
        x2.append(new_values[zipcode][0])
        y2.append(new_values[zipcode][1])

f, ax = plt.subplots(1, 1)

#cmap = sns.choose_colorbrewer_palette('s', as_cmap=True)
cmap = sns.cubehelix_palette(n_colors=10, start=0.3, rot=0.4, gamma=1.0, hue=1.0, light=0.9, dark=0.1, as_cmap=True)
axs0 = ax.scatter(x1, y1, s=20, c=avg_price1, cmap=cmap)
f.colorbar(axs0, ax=ax, orientation='vertical')

#cmap = sns.choose_colorbrewer_palette('s', as_cmap=True)
#cmap = sns.cubehelix_palette('s', as_cmap=True)
#axs1 = ax[1].scatter(x2, y2, s=8, c=avg_price2, cmap=cmap)
#f.colorbar(axs1, ax=ax[1], orientation='vertical')
f   