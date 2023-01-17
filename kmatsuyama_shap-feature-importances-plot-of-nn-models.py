import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import gc

import os

from tqdm.notebook import tqdm
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
path = "../input/m5-forecasting-accuracy"



calendar = pd.read_csv(os.path.join(path, "calendar.csv"))

selling_prices = pd.read_csv(os.path.join(path, "sell_prices.csv"))

sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))
sales = pd.read_csv(os.path.join(path, "sales_train_validation.csv"))
from sklearn.preprocessing import OrdinalEncoder



def prep_calendar(df):

    df = df.drop(["date", "weekday"], axis=1)

    df = df.assign(d = df.d.str[2:].astype(int))

    df = df.fillna("missing")

    cols = list(set(df.columns) - {"wm_yr_wk", "d"})

    df[cols] = OrdinalEncoder(dtype="int").fit_transform(df[cols])

    df = reduce_mem_usage(df)

    return df



calendar = prep_calendar(calendar)
def prep_selling_prices(df):

    gr = df.groupby(["store_id", "item_id"])["sell_price"]

    df["sell_price_rel_diff"] = gr.pct_change()

    df["sell_price_roll_sd7"] = gr.transform(lambda x: x.rolling(7).std())

    df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())

    df = reduce_mem_usage(df)

    return df



selling_prices = prep_selling_prices(selling_prices)
def reshape_sales(df, drop_d = None):

    if drop_d is not None:

        df = df.drop(["d_" + str(i + 1) for i in range(drop_d)], axis=1)

    df = df.assign(id=df.id.str.replace("_validation", ""))

    df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1913 + i + 1) for i in range(2 * 28)])

    df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],

                 var_name='d', value_name='demand')

    df = df.assign(d=df.d.str[2:].astype("int16"))

    return df



sales = reshape_sales(sales, 1000)
def prep_sales(df):

    df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28))

    df['rolling_mean_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())

    df['rolling_mean_t30'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())

    df['rolling_mean_t60'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(60).mean())

    df['rolling_mean_t90'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())

    df['rolling_mean_t180'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())

    df['rolling_std_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())

    df['rolling_std_t30'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())



    # Remove rows with NAs except for submission rows. rolling_mean_t180 was selected as it produces most missings

    df = df[(df.d >= 1914) | (pd.notna(df.rolling_mean_t180))]

    df = reduce_mem_usage(df)



    return df



sales = prep_sales(sales)
sales = sales.merge(calendar, how="left", on="d")

gc.collect()
sales = sales.merge(selling_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])

sales.drop(["wm_yr_wk"], axis=1, inplace=True)

gc.collect()
del selling_prices
cat_id_cols = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]

cat_cols = cat_id_cols + ["wday", "month", "year", "event_name_1", 

                          "event_type_1", "event_name_2", "event_type_2"]



# In loop to minimize memory use

for i, v in tqdm(enumerate(cat_id_cols)):

    sales[v] = OrdinalEncoder(dtype="int").fit_transform(sales[[v]])



sales = reduce_mem_usage(sales)

sales.head()

gc.collect()
num_cols = ["sell_price", "sell_price_rel_diff", "sell_price_roll_sd7", "sell_price_cumrel",

#            "lag_t28", "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60", 

#            "rolling_mean_t90", "rolling_mean_t180", "rolling_std_t7", "rolling_std_t30"]

           ]

bool_cols = ["snap_CA", "snap_TX", "snap_WI"]

dense_cols = num_cols + bool_cols



# Need to do column by column due to memory constraints

for i, v in tqdm(enumerate(num_cols)):

    sales[v] = sales[v].fillna(sales[v].median())

    

sales.head()
test = sales[sales.d >= 1914]

test = test.assign(id=test.id + "_" + np.where(test.d <= 1941, "validation", "evaluation"),

                   F="F" + (test.d - 1913 - 28 * (test.d > 1941)).astype("str"))

test.head()

gc.collect()
from sklearn.model_selection import train_test_split

# Submission data

#X_test = make_X(test)



# One month of validation data

flag = (sales.d < 1914) & (sales.d >= 1914 - 28)

#valid = (make_X(sales[flag]),

#         sales["demand"][flag])



# Rest is used for training

flag = sales.d < 1914 - 28



sales = sales.sample(n=10000, random_state=1)

X_train = sales[flag].drop('id', axis=1)

y_train = sales["demand"][flag]



#X_train, X_test, y_train, y_test = train_test_split(

#         X, y, test_size=0.33, random_state=42)

                             

del sales, flag

gc.collect()
import tensorflow as tf

import tensorflow.keras as keras



from tensorflow.keras import regularizers

from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, concatenate, Flatten

from tensorflow.keras.models import Model
from keras.models import Sequential

from keras.layers import Dense, Activation, LSTM

from keras.callbacks import TensorBoard

import keras.backend as K

EarlyStopping = tf.keras.callbacks.EarlyStopping()





def rmse(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred -y_true)))



epochs= 12

batch_size = 28

verbose = 1

validation_split = 0.3

input_dim = X_train.shape[1]

n_out = 1

lr = 0.5



model = Sequential()

model.add(Dense(1024, input_shape=(input_dim,)))

model.add(Activation('relu'))

model.add(Dense(1024))

model.add(Activation('relu'))





model.add(Dense(n_out))

#model.add(Activation('tanh'))

model.summary()

          

model.compile(loss=keras.losses.mean_squared_error,

                  metrics=["mse"],

                  optimizer='adam')



hist = model.fit(X_train, y_train,

                         batch_size = batch_size, epochs = epochs,

                         callbacks = [EarlyStopping],

                         verbose=verbose, validation_split=validation_split)
plt.clf()

plt.figsize=(15, 10)

loss = hist.history['loss']

val_loss = hist.history['val_loss']

epochs = range(1, len(loss) +1)



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
#It takes so long a time!

import shap

summary = shap.kmeans(X_train.values, 25)



explainer = shap.KernelExplainer(model.predict, summary)

shap_values = explainer.shap_values(X_train.values)
shap.summary_plot(shap_values[0], X_train)