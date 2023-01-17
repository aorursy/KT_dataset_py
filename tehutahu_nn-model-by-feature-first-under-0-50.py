from  datetime import datetime, timedelta

import gc

import numpy as np, pandas as pd

import lightgbm as lgb



from typing import Union

from tqdm.notebook import tqdm_notebook as tqdm

# import optuna.integration.
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns: #columns毎に処理

        col_type = df[col].dtypes

        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更

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
CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 

         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",

        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
pd.options.display.max_columns = 50
h = 28 

max_lags = 57

tr_last = 1913

fday = datetime(2016,4, 25)

fday
def create_dt(is_train = True, nrows = None, first_day = 1200):

    prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)

    for col, col_dtype in PRICE_DTYPES.items():

        if col_dtype == "category":

            prices[col] = prices[col].cat.codes.astype("int16")

            prices[col] -= prices[col].min()

            

    cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)

    cal["date"] = pd.to_datetime(cal["date"])

    for col, col_dtype in CAL_DTYPES.items():

        if col_dtype == "category":

            cal[col] = cal[col].cat.codes.astype("int16")

            cal[col] -= cal[col].min()

    

    start_day = max(1 if is_train  else tr_last-max_lags, first_day)

    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]

    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

    dtype = {numcol:"float32" for numcol in numcols} 

    dtype.update({col: "category" for col in catcols if col != "id"})

    dt = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv", 

                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)

    

    for col in catcols:

        if col != "id":

            dt[col] = dt[col].cat.codes.astype("int16")

            dt[col] -= dt[col].min()

    

    if not is_train:

        for day in range(tr_last+1, tr_last+ 28 +1):

            dt[f"d_{day}"] = np.nan

    

    dt = pd.melt(dt,

                  id_vars = catcols,

                  value_vars = [col for col in dt.columns if col.startswith("d_")],

                  var_name = "d",

                  value_name = "sales")

    

    dt = dt.merge(cal, on= "d", copy = False)

    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)

    

    return reduce_mem_usage(dt)
def create_fea(dt):

    lags = [7, 28]

    lag_cols = [f"lag_{lag}" for lag in lags ]

    for lag, lag_col in zip(lags, lag_cols):

        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)



    wins = [7, 28]

    for win in wins :

        for lag,lag_col in zip(lags, lag_cols):

            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())



    

    

    date_features = {

        

        "wday": "weekday",

        "week": "weekofyear",

        "month": "month",

        "quarter": "quarter",

        "year": "year",

        "mday": "day",

#         "ime": "is_month_end",

#         "ims": "is_month_start",

    }

    

#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)

    

    for date_feat_name, date_feat_func in date_features.items():

        if date_feat_name in dt.columns:

            dt[date_feat_name] = dt[date_feat_name].astype("int16")

        else:

            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
FIRST_DAY = 700 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !
%%time



df = create_dt(is_train=True, first_day= FIRST_DAY)

df.shape
df.head(10)
df.info()
%%time



create_fea(df)

df.shape
df.info()
df.head()
df.dropna(inplace = True)

df.shape
import tensorflow as tf

from keras.models import Model, load_model

from keras.layers import Input, Dropout, Dense, Embedding, concatenate, BatchNormalization, Flatten

from keras import backend as K

from keras.losses import mean_squared_error as mse_loss

from keras.optimizers import RMSprop, Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



def rmse(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))



def build_model(numericals):



    #Inputs

    nums = []

    for col in numericals:

        nums.append(Input(shape=[1], name=col))

    

    item_id = Input(shape=[1], name="item_id")

    dept_id = Input(shape=[1], name="dept_id")

    store_id = Input(shape=[1], name="store_id")

    cat_id = Input(shape=[1], name="cat_id")

    state_id = Input(shape=[1], name="state_id")

    event_name_1 = Input(shape=[1], name="event_name_1")

    event_name_2 = Input(shape=[1], name="event_name_2")

    event_type_1 = Input(shape=[1], name="event_type_1")

    event_type_2 = Input(shape=[1], name="event_type_2")

#     mday = Input(shape=[1], name="mday")

#     month = Input(shape=[1], name="month")

#     quarter = Input(shape=[1], name="quarter")

#     snap_CA = Input(shape=[1], name="snap_CA")

#     snap_TX = Input(shape=[1], name="snap_TX")

#     snap_WI = Input(shape=[1], name="snap_WI")

#     wday = Input(shape=[1], name="wday")

#     week = Input(shape=[1], name="week")

#     year = Input(shape=[1],name="year")

    

    #Embeddings layers

    emb_item_id = Embedding(3049, 10)(item_id)

    emb_dept_id = Embedding(7, 2)(dept_id)

    emb_store_id = Embedding(10, 2)(store_id)

    emb_cat_id = Embedding(3, 2)(cat_id)

    emb_state_id = Embedding(3, 2)(state_id)

    emb_event_name_1 = Embedding(31, 4)(event_name_1)

    emb_event_name_2 = Embedding(4, 2)(event_name_2)

    emb_event_type_1 = Embedding(5, 2)(event_type_1)

    emb_event_type_2 = Embedding(3, 2)(event_type_2)



    concat_emb = concatenate([

           Flatten() (emb_item_id)

         , Flatten() (emb_dept_id)

         , Flatten() (emb_store_id)

         , Flatten() (emb_cat_id)

         , Flatten() (emb_state_id)

         , Flatten() (emb_event_name_1)

         , Flatten() (emb_event_name_2)

         , Flatten() (emb_event_type_1)

         , Flatten() (emb_event_type_2)

    ])

    categ = Dense(128, activation='relu') (concat_emb)

    categ = Dropout(0.2686)(categ)

    categ = BatchNormalization()(categ)

    dateg = Dense(128, activation='relu') (categ)

    categ = Dropout(0.0051)(categ)

    

    #main layer

    main_l = concatenate([categ, *nums])

    main_l = Dense(32,activation='relu') (main_l)

    main_l = Dropout(0.0517)(main_l)

    main_l = BatchNormalization()(main_l)

    main_l = Dense(16,activation='relu') (main_l)

    main_l = Dropout(0.0216)(main_l)

    

    #output

    output = Dense(1) (main_l)



    model = Model(

        [

        item_id,

        dept_id,

        store_id, 

        cat_id, 

        state_id,

        event_name_1,

        event_name_2,

        event_type_1,

        event_type_2,

        *nums

        ], 

        output

    )



    model.compile(optimizer=Adam(lr=3e-4),

                  loss=mse_loss,

                  metrics=[rmse])

    model.summary()

    

    return model
def train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid,):

    early_stopping = EarlyStopping(monitor='val_rmse', mode='min', patience=5, verbose=1, restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_rmse', factor=0.1, patience=3, verbose=1, mode='min')

    model_checkpoint = ModelCheckpoint(f"model.h5", save_best_only=True, verbose=1, monitor='val_rmse', mode='min')



    hist = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,

                            validation_data=(X_v, y_valid), verbose=1,

                            callbacks=[early_stopping, reduce_lr, model_checkpoint])

    

    return keras_model, hist
def get_keras_data(df, feat_cols):

    X = {col: np.array(df[col]) for col in feat_cols}

    return X
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]

use_cols = df.columns[~df.columns.isin(useless_cols)]

cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]

num_cols = list(np.setdiff1d(use_cols, cat_feats))



np.random.seed(777)



fake_valid_inds = np.random.choice(df.index.values, 2_000_000, replace = False)

train_inds = np.setdiff1d(df.index.values, fake_valid_inds)

df_train = df.loc[train_inds]

df_val = df.loc[fake_valid_inds]



# valid_day = df["date"].max() - timedelta(days=28)

# df_train = df[df["date"] <= valid_day]

# df_val = df[df["date"] > valid_day]



X_train = df_train[use_cols]

y_train = df_train["sales"]

X_val = df_val[use_cols]

y_val = df_val["sales"]

X_train = get_keras_data(X_train, use_cols)

X_val = get_keras_data(X_val, use_cols)

del df_train, df_val, df; gc.collect()
batch_size = 1024

epochs = 32



model = build_model(num_cols)

model, hist = train_model(model, X_train, y_train, batch_size, epochs, X_val, y_val, )
import matplotlib.pyplot as plt



# https://keras.io/visualization/

def plot_history(history, filename='rmse.png'):

    # Plot training & validation accuracy values

    plt.plot(history.history['rmse'])

    plt.plot(history.history['val_rmse'])

    plt.title('Model RMSE')

    plt.ylabel('RMSE')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc='upper left')

    plt.savefig(filename)

    plt.show()

    plt.close()
plot_history(hist)
# %%time



# np.random.seed(777)



# fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)

# train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)

# train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], 

#                          categorical_feature=cat_feats, free_raw_data=False)

# fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],

#                               categorical_feature=cat_feats,

#                  free_raw_data=False)# This is a random sample, we're not gonna apply any time series train-test-split tricks here!
# del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()
# params = {

#         "objective" : "poisson",

#         "metric" :"rmse",

#         "force_row_wise" : True,

#         "learning_rate" : 0.075,

# #         "sub_feature" : 0.8,

#         "sub_row" : 0.75,

#         "bagging_freq" : 1,

#         "lambda_l2" : 0.1,

# #         "nthread" : 4

# #         "metric": ["rmse"],

#     'verbosity': 1,

#     'num_iterations' : 1200,

#     'num_leaves': 128,

#     "min_data_in_leaf": 100,

# }
# %%time



# m_lgb = lgb.train(params, train_data, valid_sets = [valid_data], verbose_eval=20) 
# m_lgb.save_model("model.lgb")
# %%time



# alphas = [1.028, 1.023, 1.018]

# weights = [1/len(alphas)]*len(alphas)

# sub = 0.



# for icount, (alpha, weight) in enumerate(zip(alphas, weights)):



#     te = create_dt(False)

#     cols = [f"F{i}" for i in range(1,29)]



#     for tdelta in range(0, 28):

#         day = fday + timedelta(days=tdelta)

#         print(tdelta, day)

#         tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()

#         create_fea(tst)

#         tst = tst.loc[tst.date == day , use_cols]

#         te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev







#     te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()

# #     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h), 

# #                                                                           "id"].str.replace("validation$", "evaluation")

#     te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]

#     te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()

#     te_sub.fillna(0., inplace = True)

#     te_sub.sort_values("id", inplace = True)

#     te_sub.reset_index(drop=True, inplace = True)

#     te_sub.to_csv(f"submission_{icount}.csv",index=False)

#     if icount == 0 :

#         sub = te_sub

#         sub[cols] *= weight

#     else:

#         sub[cols] += te_sub[cols]*weight

#     print(icount, alpha, weight)





# sub2 = sub.copy()

# sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")

# sub = pd.concat([sub, sub2], axis=0, sort=False)

# sub.to_csv("submission.csv",index=False)
# cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]

# useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]

# use_cols = df.columns[~df.columns.isin(useless_cols)]



# m_lgb = lgb.Booster(model_file="../input/m5-first-public-notebook-under-0-50/model.lgb")

# m_lgb
del X_train, y_train, X_val, y_val ; gc.collect()
%%time



sub = 0.



# fday = valid_day

te = create_dt(False)

cols = [f"F{i}" for i in range(1,29)]



for tdelta in range(0, 28):

    day = fday + timedelta(days=tdelta)

    print(tdelta, day)

    tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()

    create_fea(tst)

    tst = tst.loc[tst.date == day , use_cols]

    tst = get_keras_data(tst, use_cols)

    te.loc[te.date == day, "sales"] = model.predict(tst)





te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()

#     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h), 

#                                                                           "id"].str.replace("validation$", "evaluation")

te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]

te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()

te_sub.fillna(0., inplace = True)

te_sub.sort_values("id", inplace = True)

te_sub.reset_index(drop=True, inplace = True)

te_sub.to_csv(f"submission_.csv",index=False)

sub = te_sub



sub2 = sub.copy()

sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")

sub = pd.concat([sub, sub2], axis=0, sort=False)

sub.to_csv("submission.csv",index=False)
# sub = te_sub



# sub2 = sub.copy()

# sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")

# sub = pd.concat([sub, sub2], axis=0, sort=False)

# sub.to_csv("submission.csv",index=False)
sub.head(10)
# sub.id.nunique(), sub["id"].str.contains("validation$").sum()
sub.shape
## evaluation metric

## from https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 and edited to get scores at all levels

class WRMSSEEvaluator(object):



    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):

        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]

        train_target_columns = train_y.columns.tolist()

        weight_columns = train_y.iloc[:, -28:].columns.tolist()



        train_df['all_id'] = 0  # for lv1 aggregation



        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()

        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()



        if not all([c in valid_df.columns for c in id_columns]):

            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)



        self.train_df = train_df

        self.valid_df = valid_df

        self.calendar = calendar

        self.prices = prices



        self.weight_columns = weight_columns

        self.id_columns = id_columns

        self.valid_target_columns = valid_target_columns



        weight_df = self.get_weight_df()



        self.group_ids = (

            'all_id',

            'state_id',

            'store_id',

            'cat_id',

            'dept_id',

            ['state_id', 'cat_id'],

            ['state_id', 'dept_id'],

            ['store_id', 'cat_id'],

            ['store_id', 'dept_id'],

            'item_id',

            ['item_id', 'state_id'],

            ['item_id', 'store_id']

        )



        for i, group_id in enumerate(tqdm(self.group_ids)):

            train_y = train_df.groupby(group_id)[train_target_columns].sum()

            scale = []

            for _, row in train_y.iterrows():

                series = row.values[np.argmax(row.values != 0):]

                scale.append(((series[1:] - series[:-1]) ** 2).mean())

            setattr(self, f'lv{i + 1}_scale', np.array(scale))

            setattr(self, f'lv{i + 1}_train_df', train_y)

            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())



            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)

            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())



    def get_weight_df(self) -> pd.DataFrame:

        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()

        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])

        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})

        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)



        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])

        weight_df['value'] = weight_df['value'] * weight_df['sell_price']

        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']

        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)

        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)

        return weight_df



    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:

        valid_y = getattr(self, f'lv{lv}_valid_df')

        score = ((valid_y - valid_preds) ** 2).mean(axis=1)

        scale = getattr(self, f'lv{lv}_scale')

        return (score / scale).map(np.sqrt)



    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:

        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape



        if isinstance(valid_preds, np.ndarray):

            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)



        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)



        group_ids = []

        all_scores = []

        for i, group_id in enumerate(self.group_ids):

            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)

            weight = getattr(self, f'lv{i + 1}_weight')

            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)

            group_ids.append(group_id)

            all_scores.append(lv_scores.sum())



        return group_ids, all_scores

## public LB rank

def get_lb_rank(score):

    """

    Get rank on public LB as of 2020-05-31 23:59:59

    """

    df_lb = pd.read_csv("../input/m5-accuracy-final-public-lb/m5-forecasting-accuracy-publicleaderboard-rank.csv")



    return (df_lb.Score <= score).sum() + 1

## reading data

def make_evaluator():

    df_train_full = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")

    df_calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

    df_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

#     df_train = df_train_full.drop(df_train_full.columns[df_train_full.columns.str.startswith('d_')][:first_day-1], axis=1)[:-28]

    df_train = df_train_full.iloc[:, :-28]

    df_valid = df_train_full.iloc[:, -28:]



    df_train, df_valid, df_calendar, df_prices = [reduce_mem_usage(df) for df in [df_train, df_valid, df_calendar, df_prices]]

    evaluator = WRMSSEEvaluator(df_train, df_valid, df_calendar, df_prices)

    return evaluator
def evaluate_WRMSSEE(preds_valid):

    df_sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")

    df_sample_submission["order"] = range(df_sample_submission.shape[0])

    

    preds_valid = preds_valid[preds_valid.id.str.contains("validation")]

    preds_valid = preds_valid.merge(df_sample_submission[["id", "order"]], on = "id").sort_values("order").drop(["id", "order"], axis = 1)

    preds_valid.rename(columns = {

        "F1": "d_1914", "F2": "d_1915", "F3": "d_1916", "F4": "d_1917", "F5": "d_1918", "F6": "d_1919", "F7": "d_1920",

        "F8": "d_1921", "F9": "d_1922", "F10": "d_1923", "F11": "d_1924", "F12": "d_1925", "F13": "d_1926", "F14": "d_1927",

        "F15": "d_1928", "F16": "d_1929", "F17": "d_1930", "F18": "d_1931", "F19": "d_1932", "F20": "d_1933", "F21": "d_1934",

        "F22": "d_1935", "F23": "d_1936", "F24": "d_1937", "F25": "d_1938", "F26": "d_1939", "F27": "d_1940", "F28": "d_1941"

    }, inplace = True)

    

    evaluator = make_evaluator()

    groups, scores = evaluator.score(preds_valid)



    score_public_lb = np.mean(scores)

    score_public_rank = get_lb_rank(score_public_lb)



    for i in range(len(groups)):

        print(f"Score for group {groups[i]}: {round(scores[i], 5)}")



    print(f"\nPublic LB Score: {round(score_public_lb, 5)}")

    print(f"Public LB Rank: {score_public_rank}")
evaluate_WRMSSEE(sub)