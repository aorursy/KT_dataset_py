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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from tqdm import tqdm

import gc
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

#

def autocorrelation(ys, t=1):

    return np.corrcoef(ys[:-t], ys[t:])
#==========================================================================

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

    sales = sales.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id","scale1",

                                "sales1","start","sales2","scale2"],

                        var_name='d', value_name='demand')

    #

    return sales

#===============================================================

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

    

    calendar["nb"] = calendar.index + 1



    calendar.drop(["event_name_1", "event_name_2", "event_type_1", "event_type_2", "date", "weekday"], 

                  axis=1, inplace=True)

    return calendar

#=========================================================

def make_dataset(categorize=False ,start=1400, upper= 1970):

    global maps, mods

    print("loading calendar...")

    calendar = pd.read_csv("../input/m5-forecasting-uncertainty/calendar.csv")

    print("loading sales...")

    sales = pd.read_csv("../input/walmartadd/sales_aug.csv")

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

    #del calendar



    print("reordering...")

    sales.sort_values(by=["id","nb"], inplace=True)

    print("re-indexing..")

    sales.reset_index(inplace=True, drop=True)

    gc.collect()



    sales['n_week'] = (sales['nb']-1)//7

    sales["nday"] -= 1

    mods['nday'] = 31

    sales = reduce_mem_usage(sales)

    calendar = calendar.loc[calendar.nb>=start]

    gc.collect()

    return sales, calendar

#===============================================================================#
#=========================

def qloss_np(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]

    q = np.array([qs])

    e = y_true - y_pred

    v = np.maximum(q*e, (q-1)*e)

    return np.mean(v)
%%time

CATEGORIZE = True;

START = 1400; UPPER = 1970;

maps = {}

mods = {}

sales, calendar = make_dataset(categorize=CATEGORIZE ,start=START, upper= UPPER)
sales.nb.min(), sales.nb.max()
sales["x"] = sales["demand"] / sales["scale1"]

LAGS = [28, 35, 42, 49, 56, 63]

FEATS = []

groups = sales.groupby("id")

for lag in tqdm(LAGS):

    #sales[f"x_{lag}"] = sales.groupby("id")["x"].shift(lag)

    sales[f"x_{lag}"] = groups["x"].shift(lag)

    FEATS.append(f"x_{lag}")

NITEMS = 42840

scale = sales['scale1'].values.reshape((NITEMS, -1))

ids = sales['id'].values.reshape((NITEMS, -1))

ys = sales[['x','sales1']].values.reshape((NITEMS, -1, 2))

Z = sales[FEATS].values.reshape((NITEMS, -1, len(FEATS)))
scale.shape, ids.shape, ys.shape, Z.shape
LEN = 1969 - START + 1

MAX_LAG = max(LAGS)

print(LEN, MAX_LAG)
CATCOLS = ['snap_CA', 'snap_TX', 'snap_WI', 'wday', 'month', 'year', 'event_name', 'nday',

           'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
C = sales[CATCOLS].values.reshape((NITEMS, -1, len(CATCOLS)))
C.shape, Z.shape, ys.shape, LEN
import tensorflow.keras.layers as L

import tensorflow.keras.models as M

import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import tensorflow as tf

import itertools as it
def make_data(c, z, y):

    x = {"snap_CA":c[:,:,0], "snap_TX":c[:,:,1], "snap_WI":c[:,:,2], "wday":c[:,:,3], 

         "month":c[:,:,4], "year":c[:,:,5], "event":c[:,:,6], "nday":c[:,:,7], 

         "item":c[:,:,8], "dept":c[:,:,9], "cat":c[:,:,10], "store":c[:,:,11], 

         "state":c[:,:,12], "num":z}

    t = y

    return x, t
#========================

class DataGenerator(tf.keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, x, brks, batch_size=32, shuffle=True):

        'Initialization'

        self.batch_size = batch_size

        self.c = x[0]

        self.z = x[1]

        self.y = x[2]

        self.brks = brks.copy()

        self.list_IDs = np.array(range(42840))

        self.shuffle = shuffle

        self.nb_batch = int(np.ceil(len(self.list_IDs) / self.batch_size))

        self.n_windows = brks.shape[0]

        self.on_epoch_end()

        #B+H <= idx <= LEN - 28



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.ceil(len(self.list_IDs) / self.batch_size))*(self.n_windows)



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        idx, kx = self.ids[index]

        batch_ids = self.list_IDs[idx*self.batch_size:(idx+1)*self.batch_size]



        # Generate data

        X, y = self.__data_generation(batch_ids, kx)



        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.ids = list( it.product(np.arange(0, self.nb_batch), self.brks )) 

        if self.shuffle == True:

            np.random.shuffle(self.ids)



    def __data_generation(self, batch_ids, kx):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        x_batch, y_batch = make_data(self.c[batch_ids, kx-28:kx], self.z[batch_ids,kx-28:kx], 

                                     self.y[batch_ids, kx-28:kx])

        return x_batch, y_batch

    #

#=================================================================
#xt, yt = make_data(C[:,MAX_LAG:LEN-56,:], Z[:,MAX_LAG:LEN-56,:], ys[:,MAX_LAG:LEN-56]) #train

xv, yv = make_data(C[:,LEN-56:LEN-28,:], Z[:,LEN-56:LEN-28,:], ys[:,LEN-56:LEN-28]) # val

xe, ye = make_data(C[:,LEN-28:LEN,:], Z[:,LEN-28:LEN,:], ys[:,LEN-28:LEN]) # test
xv['snap_CA'].shape, xe['snap_CA'].shape
#========================

def wqloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true[:,:,:1] - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    psl = tf.reduce_mean(v, axis=[1,2])

    weights = y_true[:,0,1] / K.sum(y_true[:,0,1])

    return tf.reduce_sum(psl * weights)

#========================

def qloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]

    q = tf.constant(np.array([[qs]]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.mean(v)

#============================#

def make_model(n_in):

    

    seq_len = 28

    num = L.Input((seq_len, n_in,), name="num")

    

    ca = L.Input((seq_len,), name="snap_CA")

    tx = L.Input((seq_len,), name="snap_TX")

    wi = L.Input((seq_len,), name="snap_WI")

    wday = L.Input((seq_len,), name="wday")

    month = L.Input((seq_len,), name="month")

    year = L.Input((seq_len,), name="year")

    event = L.Input((seq_len,), name="event")

    nday = L.Input((seq_len,), name="nday")

    item = L.Input((seq_len,), name="item")

    dept = L.Input((seq_len,), name="dept")

    cat = L.Input((seq_len,), name="cat")

    store = L.Input((seq_len,), name="store")

    state = L.Input((seq_len,), name="state")

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

    context = L.Concatenate(name="context")(p)

    

    x = L.Concatenate(name="x1")([context, num])

    x = L.GRU(100, return_sequences=True, name="d1")(x)

    x = L.Dropout(0.3)(x)

    x = L.Concatenate(name="m1")([x, context])

    x = L.GRU(100, return_sequences=True, name="d2")(x)

    x = L.Dropout(0.3)(x)

    x = L.Concatenate(name="m2")([x, context])

    x = L.GRU(100, return_sequences=True, name="d3")(x)

    preds = L.Dense(9, activation="linear", name="preds")(x)

    model = M.Model(inp, preds, name="M1")

    model.compile(loss=wqloss, optimizer="adam")

    return model
net = make_model(Z.shape[2])

ckpt = ModelCheckpoint("w.h5", monitor='val_loss', verbose=1, save_best_only=True,mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=5, min_lr=0.0005)

es = EarlyStopping(monitor='val_loss', patience=20)

print(net.summary())
n_slices = LEN // 28

brks = np.array([LEN - (n_slices - i)*28 for i in range(n_slices+1)])

brks = brks[brks>=max(LAGS)+28]

print(LEN, C.shape, Z.shape)

print(brks)
C.min(), ys.min(), Z[:,66:].min()

net.fit_generator(DataGenerator((C, Z, ys), brks[:-1], batch_size=10_000), epochs=100, 

                  validation_data=(xv, yv), callbacks=[ckpt, reduce_lr, es])
nett = make_model(Z.shape[2])

nett.load_weights("w.h5")
pv = nett.predict(xv, batch_size=500, verbose=1)

pe = nett.predict(xe, batch_size=500, verbose=1)
nett.evaluate(xv, yv, batch_size=5000)
sv = scale[:,LEN-56:LEN-28]

se = scale[:,LEN-28:LEN]
k = np.random.randint(0, 42840)

#k = np.random.randint(0, 200)

print(ids[k, 0])

plt.plot(np.arange(28, 56), yv[k,:,0], label="future")

plt.plot(np.arange(28, 56), pv[k ,:, 3], label="q25")

plt.plot(np.arange(28, 56), pv[k ,:, 4], label="q50")

plt.plot(np.arange(28, 56), pv[k, :, 5], label="q75")

plt.plot(np.arange(0, 28), ys[k, -56:-28, 0], label="past")

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

#============#
sub = pd.DataFrame()

sub = sub.append(VALID + EVAL)

del VALID, EVAL, t1, t2
sub.head()
sub.to_csv("submission.csv", index=False)