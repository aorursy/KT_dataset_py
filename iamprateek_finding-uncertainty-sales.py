# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
price=pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv")

calen=pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/calendar.csv")
price.head()
calen.head()
price.info()
calen.info()
# helper function to reduce memory

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

    if verbose: print('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

#

def autocorrelation(ys, t=1):

    return np.corrcoef(ys[:-t], ys[t:])
# preprocess sales data

def preprocess_sales(sales, start=1400, upper=1970):

    if start is not None:

        print("dropping...")

        to_drop = [f"d_{i+1}" for i in range(start-1)]

        print(sales.shape)

        sales.drop(to_drop, axis=1, inplace=True)

        print(sales.shape)

    #=======

    print("adding...")

    new_columns = ['d_%i'%i for i in range(1942, upper, 1)]

    for col in new_columns:

        sales[col] = np.nan

    print("melting...")

    sales = sales.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id","scale","start"],

                        var_name='d', value_name='demand')



    print("generating order")

    if start is not None:

        skip = start

    else:

        skip = 1

    sales["nb"] =sales.index // 42840 + skip

    return sales
# preprocess calender data

def preprocess_calendar(calendar):

    global maps, mods

    calendar["event_name"] = calendar["event_name_1"]

    calendar["event_type"] = calendar["event_type_1"]



    map1 = {mod:i for i,mod in enumerate(calendar['event_name'].unique())}

    calendar['event_name'] = calendar['event_name'].map(map1)

    map2 = {mod:i for i,mod in enumerate(calendar['event_type'].unique())}

    calendar['event_type'] = calendar['event_type'].map(map2)

    calendar['nday'] = calendar['date'].str[-2:].astype(int)

    maps["event_name"] = map1

    maps["event_type"] = map2

    mods["event_name"] = len(map1)

    mods["event_type"] = len(map2)

    calendar["wday"] -=1

    calendar["month"] -=1

    calendar["year"] -= 2011

    mods["month"] = 12

    mods["year"] = 6

    mods["wday"] = 7

    mods['snap_CA'] = 2

    mods['snap_TX'] = 2

    mods['snap_WI'] = 2



    calendar.drop(["event_name_1", "event_name_2", "event_type_1", "event_type_2", "date", "weekday"], 

                  axis=1, inplace=True)

    return calendar
# apply helper functions and create a dataset

def make_dataset(categorize=False ,start=1400, upper= 1970):

    global maps, mods

    print("loading calendar...")

    calendar = pd.read_csv("../input/m5-forecasting-uncertainty/calendar.csv")

    print("loading sales...")

    sales = pd.read_csv("../input/walmartadd/sales.csv")

    cols = ["item_id", "dept_id", "cat_id","store_id","state_id"]

    if categorize:

        for col in cols:

            temp_dct = {mod:i for i, mod in enumerate(sales[col].unique())}

            mods[col] = len(temp_dct)

            maps[col] = temp_dct

        for col in cols:

            sales[col] = sales[col].map(maps[col])

        #



    sales =preprocess_sales(sales, start=start, upper= upper)

    calendar = preprocess_calendar(calendar)

    calendar = reduce_mem_usage(calendar)

    print("merge with calendar...")

    sales = sales.merge(calendar, on='d', how='left')

    del calendar



    print("reordering...")

    sales.sort_values(by=["id","nb"], inplace=True)

    print("re-indexing..")

    sales.reset_index(inplace=True, drop=True)

    gc.collect()



    sales['n_week'] = (sales['nb']-1)//7

    sales["nday"] -= 1

    mods['nday'] = 31

    sales = reduce_mem_usage(sales)

    gc.collect()

    return sales
import time

import gc
%%time

CATEGORIZE = True;

START = 1400; UPPER = 1970;

maps = {}

mods = {}

sales = make_dataset(categorize=CATEGORIZE ,start=START, upper= UPPER)
from tqdm import tqdm
sales["x"] = sales["demand"] / sales["scale"]

LAGS = [28, 35, 42, 49, 56, 63]

FEATS = []

for lag in tqdm(LAGS):

    sales[f"x_{lag}"] = sales.groupby("id")["x"].shift(lag)

    FEATS.append(f"x_{lag}")
sales = sales.loc[sales.nb>sales.start]

print(sales.shape)

nb = sales['nb'].values

MAX_LAG = max(LAGS)

#tr_mask = np.logical_and(nb>START + MAX_LAG, nb<=1913)

tr_mask = np.logical_and(nb>START + MAX_LAG, nb<=1941)

val_mask = np.logical_and(nb>1913, nb<=1941)

te_mask = np.logical_and(nb>1941, nb<=1969)
scale = sales['scale'].values

ids = sales['id'].values

ys = sales['x'].values

Z = sales[FEATS].values
sv = scale[val_mask]

se = scale[te_mask]

ids = ids[te_mask]

ids = ids.reshape((-1, 28))
ca = sales[['snap_CA']].values

tx = sales[['snap_TX']].values

wi = sales[['snap_WI']].values

wday = sales[['wday']].values

month = sales[['month']].values

year = sales[['year']].values

event = sales[['event_name']].values

nday = sales[['nday']].values
item = sales[['item_id']].values

dept = sales[['dept_id']].values

cat = sales[['cat_id']].values

store = sales[['store_id']].values

state = sales[['state_id']].values
def make_data(mask):

    x = {"snap_CA":ca[mask], "snap_TX":tx[mask], "snap_WI":wi[mask], "wday":wday[mask], 

         "month":month[mask], "year":year[mask], "event":event[mask], "nday":nday[mask], 

         "item":item[mask], "dept":dept[mask], "cat":cat[mask], "store":store[mask], 

         "state":state[mask], "num":Z[mask]}

    t = ys[mask]

    return x, t
xt, yt = make_data(tr_mask) #train

xv, yv = make_data(val_mask) # val

xe, ye = make_data(te_mask) # test
import tensorflow.keras.layers as L

import tensorflow.keras.models as M

import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import tensorflow as tf
def qloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.mean(v)
def make_model(n_in):

    

    num = L.Input((n_in,), name="num")

    

    ca = L.Input((1,), name="snap_CA")

    tx = L.Input((1,), name="snap_TX")

    wi = L.Input((1,), name="snap_WI")

    wday = L.Input((1,), name="wday")

    month = L.Input((1,), name="month")

    year = L.Input((1,), name="year")

    event = L.Input((1,), name="event")

    nday = L.Input((1,), name="nday")

    item = L.Input((1,), name="item")

    dept = L.Input((1,), name="dept")

    cat = L.Input((1,), name="cat")

    store = L.Input((1,), name="store")

    state = L.Input((1,), name="state")

    inp = {"snap_CA":ca, "snap_TX":tx, "snap_WI":wi, "wday":wday, 

           "month":month, "year":year, "event":event, "nday":nday,

           "item":item, "dept":dept, "cat":cat, "store":store, 

           "state":state, "num":num} 

    #

    ca_ = L.Embedding(mods["snap_CA"], mods["snap_CA"], name="ca_3d")(ca)

    tx_ = L.Embedding(mods["snap_TX"], mods["snap_TX"], name="tx_3d")(tx)

    wi_ = L.Embedding(mods["snap_WI"], mods["snap_WI"], name="wi_3d")(wi)

    wday_ = L.Embedding(mods["wday"], mods["wday"], name="wday_3d")(wday)

    month_ = L.Embedding(mods["month"], mods["month"], name="month_3d")(month)

    year_ = L.Embedding(mods["year"], mods["year"], name="year_3d")(year)

    event_ = L.Embedding(mods["event_name"], mods["event_name"], name="event_3d")(event)

    nday_ = L.Embedding(mods["nday"], mods["nday"], name="nday_3d")(nday)

    item_ = L.Embedding(mods["item_id"], 10, name="item_3d")(item)

    dept_ = L.Embedding(mods["dept_id"], mods["dept_id"], name="dept_3d")(dept)

    cat_ = L.Embedding(mods["cat_id"], mods["cat_id"], name="cat_3d")(cat)

    store_ = L.Embedding(mods["store_id"], mods["store_id"], name="store_3d")(store)

    state_ = L.Embedding(mods["state_id"], mods["state_id"], name="state_3d")(state)

    

    p = [ca_, tx_, wi_, wday_, month_, year_, event_, nday_, item_, dept_, cat_, store_, state_]

    emb = L.Concatenate(name="embds")(p)

    context = L.Flatten(name="context")(emb)

    

    x = L.Concatenate(name="x1")([context, num])

    x = L.Dense(500, activation="relu", name="d1")(x)

    x = L.Dropout(0.3)(x)

    x = L.Concatenate(name="m1")([x, context])

    x = L.Dense(500, activation="relu", name="d2")(x)

    x = L.Dropout(0.3)(x)

    x = L.Concatenate(name="m2")([x, context])

    x = L.Dense(500, activation="relu", name="d3")(x)

    preds = L.Dense(9, activation="linear", name="preds")(x)

    model = M.Model(inp, preds, name="M1")

    model.compile(loss=qloss, optimizer="adam")

    return model
net = make_model(len(FEATS))

ckpt = ModelCheckpoint("w.h5", monitor='val_loss', verbose=1, save_best_only=True,mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=5, min_lr=0.001)

es = EarlyStopping(monitor='val_loss', patience=3)

print(net.summary())
net.fit(xt, yt, batch_size=50_000, epochs=20, validation_data=(xv, yv), callbacks=[ckpt, reduce_lr, es])
nett = make_model(len(FEATS))

nett.load_weights("w.h5")
pv = nett.predict(xv, batch_size=50_000, verbose=1)

pe = nett.predict(xe, batch_size=50_000, verbose=1)
nett.evaluate(xv, yv, batch_size=50_000)
pv = pv.reshape((-1, 28, 9))

pe = pe.reshape((-1, 28, 9))
sv = sv.reshape((-1, 28))

se = se.reshape((-1, 28))
Yv = yv.reshape((-1, 28))
import matplotlib.pyplot as plt
k = np.random.randint(0, 42840)

#k = np.random.randint(0, 200)

print(ids[k, 0])

plt.plot(np.arange(28, 56), Yv[k], label="true")

plt.plot(np.arange(28, 56), pv[k ,:, 3], label="q25")

plt.plot(np.arange(28, 56), pv[k ,:, 4], label="q50")

plt.plot(np.arange(28, 56), pv[k, :, 5], label="q75")

plt.legend(loc="best")

plt.show()
names = [f"F{i+1}" for i in range(28)]

piv = pd.DataFrame(ids[:, 0], columns=["id"])

QUANTILES = ["0.005", "0.025", "0.165", "0.250", "0.500", "0.750", "0.835", "0.975", "0.995"]

VALID = []

EVAL = []



for i, quantile in tqdm(enumerate(QUANTILES)):

    t1 = pd.DataFrame(pv[:,:, i]*sv, columns=names)

    t1 = piv.join(t1)

    t1["id"] = t1["id"] + f"_{quantile}_validation"

    t2 = pd.DataFrame(pe[:,:, i]*se, columns=names)

    t2 = piv.join(t2)

    t2["id"] = t2["id"] + f"_{quantile}_evaluation"

    VALID.append(t1)

    EVAL.append(t2)
sub = pd.DataFrame()

sub = sub.append(VALID + EVAL)

del VALID, EVAL, t1, t2

sub.to_csv("submission.csv", index=False)