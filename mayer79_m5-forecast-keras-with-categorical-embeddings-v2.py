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
calendar.head()
for i, var in enumerate(["year", "weekday", "month", "event_name_1", "event_name_2", 

                         "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]):

    plt.figure()

    g = sns.countplot(calendar[var])

    g.set_xticklabels(g.get_xticklabels(), rotation=45)

    g.set_title(var)
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
calendar.head()
selling_prices.info()
def prep_selling_prices(df):

    gr = df.groupby(["store_id", "item_id"])["sell_price"]

    df["sell_price_rel_diff"] = gr.pct_change()

    df["sell_price_roll_sd7"] = gr.transform(lambda x: x.rolling(7).std())

    df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())

    df = reduce_mem_usage(df)

    return df



selling_prices = prep_selling_prices(selling_prices)
selling_prices.tail()
sales.head()
for i, var in enumerate(["state_id", "store_id", "cat_id", "dept_id"]):

    plt.figure()

    g = sns.countplot(sales[var])

    g.set_xticklabels(g.get_xticklabels(), rotation=45)

    g.set_title(var)
sales.item_id.value_counts()
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
sales.head()
sns.countplot(sales["demand"][sales["demand"] <= 10]);
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
sales.head()
sales = sales.merge(calendar, how="left", on="d")

gc.collect()

sales.head()
sales = sales.merge(selling_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])

sales.drop(["wm_yr_wk"], axis=1, inplace=True)

gc.collect()

sales.head()
del selling_prices
sales.info()
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

            "lag_t28", "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60", 

            "rolling_mean_t90", "rolling_mean_t180", "rolling_std_t7", "rolling_std_t30"]

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
# Input dict for training with a dense array and separate inputs for each embedding input

def make_X(df):

    X = {"dense1": df[dense_cols].to_numpy()}

    for i, v in enumerate(cat_cols):

        X[v] = df[[v]].to_numpy()

    return X



# Submission data

X_test = make_X(test)



# One month of validation data

flag = (sales.d < 1914) & (sales.d >= 1914 - 28)

valid = (make_X(sales[flag]),

         sales["demand"][flag])



# Rest is used for training

flag = sales.d < 1914 - 28

X_train = make_X(sales[flag])

y_train = sales["demand"][flag]

                             

del sales, flag

gc.collect()
np.unique(X_train["state_id"])
import tensorflow as tf

import tensorflow.keras as keras



from tensorflow.keras import regularizers

from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, concatenate, Flatten

from tensorflow.keras.models import Model
def create_model(lr=0.002):

    tf.random.set_seed(173)



    tf.keras.backend.clear_session()

    gc.collect()



    # Dense input

    dense_input = Input(shape=(len(dense_cols), ), name='dense1')



    # Embedding input

    wday_input = Input(shape=(1,), name='wday')

    month_input = Input(shape=(1,), name='month')

    year_input = Input(shape=(1,), name='year')

    event_name_1_input = Input(shape=(1,), name='event_name_1')

    event_type_1_input = Input(shape=(1,), name='event_type_1')

    event_name_2_input = Input(shape=(1,), name='event_name_2')

    event_type_2_input = Input(shape=(1,), name='event_type_2')

    item_id_input = Input(shape=(1,), name='item_id')

    dept_id_input = Input(shape=(1,), name='dept_id')

    store_id_input = Input(shape=(1,), name='store_id')

    cat_id_input = Input(shape=(1,), name='cat_id')

    state_id_input = Input(shape=(1,), name='state_id')



    wday_emb = Flatten()(Embedding(7, 1)(wday_input))

    month_emb = Flatten()(Embedding(12, 1)(month_input))

    year_emb = Flatten()(Embedding(6, 1)(year_input))

    event_name_1_emb = Flatten()(Embedding(31, 1)(event_name_1_input))

    event_type_1_emb = Flatten()(Embedding(5, 1)(event_type_1_input))

    event_name_2_emb = Flatten()(Embedding(5, 1)(event_name_2_input))

    event_type_2_emb = Flatten()(Embedding(5, 1)(event_type_2_input))



    item_id_emb = Flatten()(Embedding(3049, 3)(item_id_input))

    dept_id_emb = Flatten()(Embedding(7, 1)(dept_id_input))

    store_id_emb = Flatten()(Embedding(10, 1)(store_id_input))

    cat_id_emb = Flatten()(Embedding(3, 1)(cat_id_input))

    state_id_emb = Flatten()(Embedding(3, 1)(state_id_input))



    # Combine dense and embedding parts and add dense layers. Exit on linear scale.

    x = concatenate([dense_input, wday_emb, month_emb, year_emb, 

                     event_name_1_emb, event_type_1_emb, 

                     event_name_2_emb, event_type_2_emb, 

                     item_id_emb, dept_id_emb, store_id_emb,

                     cat_id_emb, state_id_emb])

    x = Dense(150, activation="tanh")(x)

    x = Dense(75, activation="tanh")(x)

    x = Dense(10, activation="tanh")(x)

    outputs = Dense(1, activation="linear", name='output')(x)



    inputs = {"dense1": dense_input, "wday": wday_input, "month": month_input, "year": year_input, 

              "event_name_1": event_name_1_input, "event_type_1": event_type_1_input,

              "event_name_2": event_name_2_input, "event_type_2": event_type_2_input,

              "item_id": item_id_input, "dept_id": dept_id_input, "store_id": store_id_input, 

              "cat_id": cat_id_input, "state_id": state_id_input}



    # Connect input and output

    model = Model(inputs, outputs)



    model.compile(loss=keras.losses.mean_squared_error,

                  metrics=["mse"],

                  optimizer=keras.optimizers.RMSprop(learning_rate=lr))

    return model
model = create_model(0.0002)

model.summary()

keras.utils.plot_model(model, 'model.png', show_shapes=True)
history = model.fit(X_train, 

                    y_train,

                    batch_size=10000,

                    epochs=30,

                    shuffle=True,

                    validation_data=valid)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()
history.history["val_loss"]
model.save('model.h5')
pred = model.predict(X_test, batch_size=10000)
test["demand"] = pred.clip(0)

submission = test.pivot(index="id", columns="F", values="demand").reset_index()[sample_submission.columns]

submission = sample_submission[["id"]].merge(submission, how="left", on="id")

submission.head()
submission[sample_submission.id=="FOODS_1_001_TX_2_validation"].head()
submission.to_csv("submission.csv", index=False)