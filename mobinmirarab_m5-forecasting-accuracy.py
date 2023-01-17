import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import gc

import os
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

    temp = ["wday", "month", "year", "event_name_1", "event_type_1", "event_name_2", 

            "event_type_2", "snap_CA", "snap_TX", "snap_WI"]

    df = df[["wm_yr_wk", "d"] + temp]

    df.fillna("missing", inplace=True)

    df[temp] = OrdinalEncoder().fit_transform(df[temp])

    for v in temp:

        df[temp] = df[temp].astype("uint8")

    df.wm_yr_wk = df.wm_yr_wk.astype("uint16")

    return df



calendar = prep_calendar(calendar)
calendar.info()
sales.head()
for i, var in enumerate(["state_id", "store_id", "cat_id", "dept_id"]):

    plt.figure()

    g = sns.countplot(sales[var])

    g.set_xticklabels(g.get_xticklabels(), rotation=45)

    g.set_title(var)
sales.item_id.value_counts()
sales.drop(["d_" + str(i+1) for i in range(800)], axis=1, inplace=True)
def melt_sales(df):

    df = df.drop(["item_id", "dept_id", "cat_id", "store_id", "state_id"], axis=1).melt(

        id_vars=['id'], var_name='d', value_name='demand')

    return df



sales = melt_sales(sales)
sales.head()
sns.countplot(sales["demand"][sales["demand"] <= 20], );
sample_submission.head()
# Turn strings like "F1" to "d_1914"

def map_f2d(d_col, id_col):

    eval_flag = id_col.str.endswith("evaluation")

    return "d_" + (d_col.str[1:].astype("int") + 1913 + 28 * eval_flag).astype("str")



# Reverse

def map_d2f(d_col, id_col):

    eval_flag = id_col.str.endswith("evaluation")

    return "F" + (d_col.str[2:].astype("int") - 1913 - 28 * eval_flag).astype("str")



# Example

map_f2d(pd.Series(["F1", "F2", "F28", "F1", "F2", "F28"]), 

        pd.Series(["validation", "validation", "validation", "evaluation", "evaluation", "evaluation"]))
submission = sample_submission.melt(id_vars="id", var_name="d", value_name="demand").assign(

    demand=np.nan,

    d = lambda df: map_f2d(df.d, df.id))

submission.head()
sales = pd.concat([sales, submission])

sales.tail()
sales.id = sales.id.str.replace("evaluation", "validation")
from sklearn.preprocessing import StandardScaler



def add_lagged_features(df):

    df['lag_t56'] = df.groupby('id')['demand'].transform(lambda x: np.log1p(x).shift(56))

    df['rolling_mean_t30'] = df.groupby('id')['demand'].transform(lambda x: np.log1p(x).shift(56).rolling(30, min_periods=1).mean())

    df['cummean'] = df.groupby('id')['demand'].transform(lambda x: x.shift(1).expanding().mean())

    temp = ['lag_t56', 'rolling_mean_t30', 'cummean']

    df.dropna(subset=temp, inplace=True)    

    df[temp] = StandardScaler().fit_transform(df[temp])

    for v in temp:

        df[v] = df[v].astype("float32")

    return df



sales = add_lagged_features(sales)
sales.head()
gc.collect()
def expand_id(id):

    return id.str.split("_", expand=True).assign(

        dept_id=lambda df: df.iloc[:,0] + "_" + df.iloc[:,1], 

        item_id=lambda df: df.iloc[:,0] + "_" + df.iloc[:,1] + "_" + df.iloc[:, 2],

        store_id=lambda df: df.iloc[:,3] + "_" + df.iloc[:,4]).drop(np.arange(6), axis=1)



# Example

expand_id(sales["id"].head())
uid = pd.Series(sales["id"].unique())

id_lookup = expand_id(uid)

id_lookup["id"] = uid



encode_item_id = OrdinalEncoder()

encode_dept_id = OrdinalEncoder()

encode_store_id = OrdinalEncoder()

id_lookup["item_id"] = encode_item_id.fit_transform(id_lookup[["item_id"]]).astype("uint16")

id_lookup["dept_id"] = encode_dept_id.fit_transform(id_lookup[["dept_id"]]).astype("uint8")

id_lookup["store_id"] = encode_store_id.fit_transform(id_lookup[["store_id"]]).astype("uint8")



id_lookup.head()
sales = sales.merge(id_lookup, on="id", how="left")

del sales["id"]
sales.head()
sales.info()
selling_prices.head()
# Add relative change

def prep_selling_prices(df):

    df = df.copy()

    df["store_id"] = encode_store_id.transform(df[["store_id"]]).astype("uint8")

    df["item_id"] = encode_item_id.transform(df[["item_id"]]).astype("uint16")

    df["wm_yr_wk"] = df["wm_yr_wk"].astype("uint16")

    

    df["sell_price_rel_diff"] = df.groupby(["store_id", "item_id"])["sell_price"].pct_change()

    sell_price_cummin = df.groupby(["store_id", "item_id"])["sell_price"].cummin()

    sell_price_cummax = df.groupby(["store_id", "item_id"])["sell_price"].cummax()

    df["sell_price_cumrel"] = (df["sell_price"] - sell_price_cummin) / (sell_price_cummax - sell_price_cummin)

    df.fillna({"sell_price_rel_diff": 0, "sell_price_cumrel": 1}, inplace=True)

    floats = ["sell_price_cumrel", "sell_price_rel_diff", "sell_price"]

    sc = StandardScaler()

    df[floats] = sc.fit_transform(df[floats])

    for v in floats:

        df[v] = df[v].astype("float32")

    return df



selling_prices = prep_selling_prices(selling_prices)
selling_prices.head()
gc.collect()

sales = sales.merge(calendar, how="left", on="d")

del sales["d"]
gc.collect()

sales = sales.merge(selling_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])

del sales["wm_yr_wk"]
sales.fillna({"sell_price": 0, "sell_price_rel_diff": 0, "sell_price_cumrel": 0}, inplace=True)
gc.collect()
training_flag = pd.notna(sales.demand)
def make_Xy(df, ind=None, return_y = True):

    if ind is not None:

        df = df[ind]

    X = {"dense1": df[["lag_t56", "rolling_mean_t30", "cummean",

                       "snap_CA", "snap_TX", "snap_WI",

                       "sell_price", "sell_price_rel_diff", 

                       "sell_price_cumrel"]].to_numpy(dtype="float32"),

         "item_id": df[["item_id"]].to_numpy(dtype="uint16")}

    for i, v in enumerate(["wday", "month", "year", "event_name_1", 

                           "event_type_1", "dept_id", "store_id"]):

        X[v] = df[[v]].to_numpy(dtype="uint8")

    if return_y:

        return X, df.demand.to_numpy(dtype="float32")

    else:

        return X
X_train, y_train = make_Xy(sales, training_flag) # make_Xy(sales[0:1000000])

y_train.shape
X_test = make_Xy(sales, ~training_flag, return_y=False)
del sales

gc.collect()
import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers



tf.keras.backend.clear_session()  # For easy reset of notebook state.



from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, concatenate, Flatten

from tensorflow.keras.models import Model

from tensorflow.keras import regularizers
# Dense part

dense_input = Input(shape=(9, ), name='dense1')

dense_branch = Dense(100, activation="relu")(dense_input)

#dense_branch = Dense(100, activation="relu")(dense_branch)



# Embedded input

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



# Embedding layers

wday_emb = Flatten()(Embedding(7, 2)(wday_input))

month_emb = Flatten()(Embedding(12, 2)(month_input))

year_emb = Flatten()(Embedding(6, 2)(year_input))

event_name_1_emb = Flatten()(Embedding(31, 5)(event_name_1_input))

event_type_1_emb = Flatten()(Embedding(5, 2)(event_type_1_input))

event_name_2_emb = Flatten()(Embedding(5, 2)(event_name_2_input))

event_type_2_emb = Flatten()(Embedding(5, 2)(event_type_2_input))

item_id_emb = Flatten()(Embedding(len(encode_item_id.categories_[0]), 30)(item_id_input))

item_id_emb = Dropout(0.3)(item_id_emb)

dept_id_emb = Flatten()(Embedding(7, 2)(dept_id_input))

store_id_emb = Flatten()(Embedding(10, 2)(store_id_input))



x = concatenate([dense_branch, wday_emb, month_emb, year_emb, event_name_1_emb,

                event_type_1_emb, item_id_emb, dept_id_emb, store_id_emb])

x = Dense(150, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(20, activation="relu")(x)

prediction = Dense(1, activation="linear", name='output')(x)



model = Model(inputs={"dense1": dense_input, "wday": wday_input, "month": month_input,

                      "year": year_input, "event_name_1": event_name_1_input, "event_type_1": event_type_1_input,

                      "item_id": item_id_input, "dept_id": dept_id_input, "store_id": store_id_input},

              outputs=prediction)
model.summary()



keras.utils.plot_model(model, 'model.png', show_shapes=True)
model.compile(loss=keras.losses.mean_squared_error,

              optimizer=keras.optimizers.RMSprop(),

              metrics=['mse'])



history = model.fit(X_train, 

                    y_train,

                    batch_size=4096,

                    epochs=3,

                    validation_split=0.1)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()
model.save('model.h5')
pred = model.predict(X_test, batch_size=4096)
pred.shape
submission.shape
submission.tail()
submission = submission.assign(

    demand = np.clip(pred, 0, None),

    d = lambda df: map_d2f(df.d, df.id))

submission.head()
# Right column order

col_order = ["id"] + ["F" + str(i + 1) for i in range(28)]

submission = submission.pivot(index="id", columns="d", values="demand").reset_index()[col_order]



# Right row order

submission = sample_submission[["id"]].merge(submission, how="left", on="id")
submission.head()
submission.to_csv("submission.csv", index=False)