!pip install torchtext==0.2.3

!pip install fastai==0.7.0



if 1==2:

    from fastai.imports import *

    from fastai.transforms import *

    from fastai.conv_learner import *

    from fastai.model import *

    from fastai.dataset import *

    from fastai.sgdr import *

    from fastai.plots import *

    from fastai.column_data import *

from fastai.structured import *
%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet, Ridge, LinearRegression, Lasso

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import RobustScaler

import seaborn as sea

import gc

import matplotlib.style as style 



style.use('seaborn-notebook') #sets the size of the charts

style.use("seaborn-pastel")

#style.use('ggplot')



import os

print(os.listdir("../input"))

PATH = "../input/"

PATH_TMP = "../../tmp/"

PATH_MODEL = "../../model/"

blnforoutlier = 0
def load_data():

    df_train = pd.read_csv(f'{PATH}train.csv')

    df_test = pd.read_csv(f'{PATH}test.csv')

    df_train.head(1)

    return df_train, df_test

def load_tran_data():

    df_history = pd.read_csv(f'{PATH}historical_transactions.csv')

    df_new = pd.read_csv(f'{PATH}new_merchant_transactions.csv')

    df_mer = pd.read_csv(f'{PATH}merchants.csv')

    df_history.head(1)

    return df_history, df_new, df_mer
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
def convert_merchant_id():

    global df_mer, df_history, df_new

    dic_merchant_id = {v:k for k,v in enumerate(list(df_mer["merchant_id"].unique()))}

    df_mer["merchant_id_SNo"] = df_mer["merchant_id"].map(dic_merchant_id)

    df_history["merchant_id_SNo"] = df_history["merchant_id"].map(dic_merchant_id)

    df_new["merchant_id_SNo"] = df_new["merchant_id"].map(dic_merchant_id)

    df_history.drop(["merchant_id"], axis=1,inplace=True)

    df_new.drop(["merchant_id"], axis=1,inplace=True)

    df_mer.drop(["merchant_id"], axis=1, inplace=True)
def convert_card_id():

    global df_history, df_new, df_train, df_test

    list_card_id = list(set(df_history["card_id"].unique()) | set(df_new["card_id"].unique()) | set(df_train["card_id"].unique()) | set(df_test["card_id"].unique()))

    dic_card_id = {v:k for k,v in enumerate(list_card_id)}

    df_new["card_id_SNo"] = df_new["card_id"].map(dic_card_id)

    df_history["card_id_SNo"] = df_history["card_id"].map(dic_card_id)

    df_train["card_id_SNo"] = df_train["card_id"].map(dic_card_id)

    df_test["card_id_SNo"] = df_test["card_id"].map(dic_card_id)



    df_history.drop('card_id',axis=1, inplace=True)

    df_new.drop('card_id',axis=1, inplace=True)

    df_train.drop('card_id',axis=1, inplace=True)
def check_duplicates(in_df, key_col, blnDrop):

    in_df["num_rec"] = 0

    df_grouped = in_df[["num_rec",key_col]].groupby([key_col]).count()

    df_grouped_duplicate = df_grouped[df_grouped["num_rec"] > 1]

    # Not sure what to do with these multiple recs. So will just take last index of each merchant_d

    df_duplicate = in_df[in_df[key_col].isin(list(df_grouped_duplicate.index.values))]

    print("No of duplicate records: ", df_duplicate.shape[0])

    if blnDrop == True:

        df_duplicate_max = in_df.loc[df_duplicate.index].reset_index()[[key_col,"index"]].groupby([key_col]).max()

        drop_index = list(set(df_duplicate.index.values) - set(df_duplicate_max["index"].values))

        in_df.drop(drop_index, inplace=True)

        del df_duplicate_max

    del df_grouped, df_grouped_duplicate, df_duplicate

    gc.collect()
def check_nulls(df, dfname):

    allsum = df.shape[0]

    for col in df.columns:

        nasum = df[col].isna().sum()

        if nasum > 0:

            print(dfname, col, df[col].dtype, nasum, np.round((nasum*100)/allsum),2)
def replace_null_with_most_freq_val(df, col):

    df["temp_freq"] = 0

    most_freq_val = df[[col,"temp_freq"]].groupby([col], as_index=False).count().sort_values(["temp_freq"]).tail(1)[col].values[0]

    null_col_index = list(df[df[col].isna()].index)

    print("Most freq val", most_freq_val, len(null_col_index))

    df.loc[null_col_index,col] = most_freq_val

    df.drop("temp_freq",axis=1, inplace=True)

    

def fix_missing_mer_for_group(in_df_mer, in_df_tran, group_cols):

    in_df_mer.sort_values("num_his_rec", inplace=True)

    df_mer_grouped = in_df_mer[["merchant_id_SNo"] + group_cols].groupby(group_cols, as_index=False).tail(1)



    df_his_null = in_df_tran[in_df_tran["merchant_id_SNo"].isna()]

    print("Null Recs", df_his_null.shape)

    df_merged = pd.merge(df_his_null.reset_index()[group_cols + ["index"]] , df_mer_grouped, how="inner", on=group_cols).set_index("index")

    print("Null recs rectified", df_merged.shape)

    in_df_tran.loc[list(df_merged.index), "merchant_id_SNo"] = df_merged.loc[list(df_merged.index), "merchant_id_SNo"]



    df_his_null = in_df_tran[in_df_tran["merchant_id_SNo"].isna()]

    print("Null recs after fix", df_his_null.shape)

    

    

def fix_missing_mer(in_df_tran, in_mer):

    in_df_tran["num_rec"] = 0

    df_tran_grouped = in_df_tran[in_df_tran.isna() == False][["num_rec","merchant_id_SNo"]].groupby("merchant_id_SNo").count()

    

    in_mer["num_his_rec"] = 0

    in_mer.set_index("merchant_id_SNo").loc[list(df_tran_grouped.index), "num_his_rec"] = df_tran_grouped.loc[list(df_tran_grouped.index), "num_rec"]

    in_mer["num_his_rec"].fillna(0, inplace=True)



    group_cols = ["subsector_id","merchant_category_id","category_1","category_2","city_id","state_id"]

    fix_missing_mer_for_group(in_mer, in_df_tran, group_cols)



    group_cols = ["subsector_id","merchant_category_id","category_1","category_2"] #,"city_id"] #,"state_id"]

    fix_missing_mer_for_group(in_mer, in_df_tran, group_cols)



    group_cols = ["subsector_id"] #,"merchant_category_id","category_1","category_2"] #,"city_id"] #,"state_id"]

    fix_missing_mer_for_group(in_mer, in_df_tran, group_cols)
def explore_label():

    global df_train

    df_train["outlier"] = 0

    df_train_outlier = df_train[(df_train["target"] < -10) | (df_train["target"] > 10)]

    df_train_outlier = df_train_outlier.loc[df_train_outlier.index]

    df_train.loc[df_train_outlier.index, "outlier"] = 1

    df_train_without_outlier = df_train.drop(df_train_outlier.index)

    

    print("Percentage outlier:", (df_train_outlier.shape[0]*100)/df_train.shape[0])

    print(df_train.shape, df_train_outlier.shape, df_train_without_outlier.shape)

    fig,ax = plt.subplots(nrows=1,ncols=3, figsize=(15,4))

    df_train[["target"]].plot(kind="hist", bins=100, ax=ax[0], title="All data");

    df_train_outlier["target"].plot(kind="hist", bins=100, ax=ax[1], title="Outlier data");

    df_train_without_outlier["target"].plot(kind="hist", bins=100, ax=ax[2], title="Data Without Outier");

    del df_train_without_outlier

    gc.collect()

    if blnforoutlier == 1:

        df_train["target"] = df_train["outlier"]

    #df_train.drop(["outlier"], axis=1, inplace=True)

    return df_train_outlier
def explore_cat_cols(df):

    global cat_cols

    fig,ax = plt.subplots(nrows=2,ncols=5 , figsize=(20,8))

    i = 0

    j = 0

    for col in cat_cols:

        df[col].value_counts().plot(kind="bar",ax=ax[i,j], title=col)

        j = j + 1

        if j == 5:

            j=0

            i = i + 1

    plt.tight_layout()

    

    

def explore_cont_data():

    global df_train

    global df_train_outlier

    fig,ax = plt.subplots(nrows=1,ncols=4, figsize=(20,4))

    df_train["first_active_monthElapsed"].plot(kind="hist", bins=100, ax=ax[0], title="All Data")

    df_train_outlier["first_active_monthElapsed"].plot(kind="hist", bins=100, ax=ax[1], title="Outlier Data")

    df_train.sort_values(["first_active_monthElapsed"]).plot(kind="scatter",x="first_active_monthElapsed", y="target", ax=ax[3], title="All Data")

    df_train_outlier.sort_values(["first_active_monthElapsed"]).plot(kind="scatter",x="first_active_monthElapsed", y="target", ax=ax[2], title="Outlier Data")
def get_X_Y(feat_cols):

    global df_train

    global df_test

    global label_col

    scaler = StandardScaler()

    X_all_raw = df_train[feat_cols].values.astype(np.float32)

    y_all = df_train[label_col].values

    X_test_raw = df_test[feat_cols].values.astype(np.float32)

    outlier = df_train["outlier"].values

    

    X_all = scaler.fit_transform(X_all_raw)

    X_test = scaler.transform(X_test_raw)

    X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, stratify=outlier)

    return X_train, X_valid, y_train, y_valid, X_test, X_all, y_all

def get_lgbm_pred():

    global X_all, y_all, X_valid, y_valid, feature_cols, X_test

    train_data = lgb.Dataset(X_all, label=y_all, free_raw_data=False)

    test_data = lgb.Dataset(X_valid, label=y_valid,  free_raw_data=False)

    params_1 = {

            'task': 'train',

            'boosting_type': 'gbdt',

            'objective': "regression",

            'verbose': 1,

            'max_depth':7,

            'num_leaves':70,

            'learning_rate':0.01,

             

        }

    if blnforoutlier == 1:

        objective = "binary"

    else:

        objective = "regression"

    params ={

                'task': 'train',

                'boosting': 'goss',

                'objective': objective,

                'metric': 'rmse',

                'learning_rate': 0.0001,

                'subsample': 0.9855232997390695,

                'max_depth': 7,

                'top_rate': 0.9064148448434349,

                'num_leaves': 63,

                'min_child_weight': 41.9612869171337,

                'other_rate': 0.0721768246018207,

                'reg_alpha': 9.677537745007898,

                'colsample_bytree': 0.5665320670155495,

                'min_split_gain': 9.820197773625843,

                'reg_lambda': 8.2532317400459,

                'min_data_in_leaf': 21,

                'verbose': -1#,

               # 'seed':int(2**n_fold),

               # 'bagging_seed':int(2**n_fold),

              #  'drop_seed':int(2**n_fold)

                }



    gbm_1 = lgb.train(params,

            train_data,

            valid_sets=test_data,

            num_boost_round=50000,

            early_stopping_rounds= 200,

            feature_name=feature_cols,

            categorical_feature='auto' #cat_cols

            )



    pred_valid = list(gbm_1.predict(X_valid))

    valmse = mean_squared_error(pred_valid, y_valid)

    print(valmse)

    pred_test = list(gbm_1.predict(X_test))

    return pred_test, valmse
def get_keras_pred(num_epoch=20):

    global X_all, y_all, X_test, X_valid, y_valid

    from keras.models import Sequential

    from keras.layers import Dense, Dropout

    from keras.wrappers.scikit_learn import KerasRegressor

    import keras



    model = Sequential()

    num_features = len(feature_cols)

    model.add(Dropout(0.2, input_shape=(num_features,) ))

    model.add(Dense(num_features*2,  kernel_initializer='normal', activation='relu'))

    model.add(Dropout(0.2, input_shape=(num_features,) ))

    model.add(Dense(num_features, kernel_initializer='normal', activation='relu'))

    if blnforoutlier == 1:

        model.add(Dense(1, kernel_initializer='normal'), activation="softmax")

        opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss='categorical_crossentropy', optimizer=opt)

    else:

        model.add(Dense(1, kernel_initializer='normal'))

        opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss='mean_squared_error', optimizer=opt)

    

    model.fit(X_all, y_all, epochs=1000, batch_size=128)

    scores = model.evaluate(X_all, y_all)

    print("\n%s: %.2f" % (model.metrics_names[0], scores))

    pred_valid = list(model.predict(X_valid))

    valmse = mean_squared_error(pred_valid, y_valid)

    print(valmse)

    pred_test = model.predict(X_test)

    return pred_test, valmse, model
def get_sklearn_model(regr, X_train, X_valid, y_train, y_valid, X_test): 

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_valid)

    print(mean_squared_error(y_valid, y_pred))

    return list(regr.predict(X_test))





def get_stratified_prediction(in_regr, in_X_all, in_y_all, in_outlier_all, in_X_test, in_feat_col):

    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=2333)

    y_pred = np.zeros(in_X_all.shape[0])

    y_pred_test = np.zeros(in_X_test.shape[0])



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(in_X_all,in_outlier_all)):

        cur_X_train = in_X_all[trn_idx]

        cur_y_train = in_y_all[trn_idx].astype(np.float64)

        cur_X_val = in_X_all[val_idx]

        in_regr.fit(cur_X_train, cur_y_train)

        y_pred[val_idx] = in_regr.predict(cur_X_val)

        cur_pred = in_regr.predict(in_X_test)

        cur_pred = cur_pred/folds.n_splits

        y_pred_test += cur_pred



    print(mean_squared_error(in_y_all, y_pred))

    feature_importance_df = pd.DataFrame({"col":in_feat_col, "coef":in_regr.coef_})

    feature_importance_df.sort_values("coef", inplace=True)

    fig,ax = plt.subplots(nrows=1,ncols=2 , figsize=(20,8))

    feature_importance_df.head(20).plot(kind="barh",x="col", y="coef", ax=ax[0])

    feature_importance_df.tail(20).plot(kind="barh",x="col", y="coef", ax=ax[1])

    plt.tight_layout()

    return y_pred_test , feature_importance_df





def get_fast_ai_pred():

    global df_train

    global df_test

    global cat_cols

    df_train.reset_index(inplace=True)

    all_idx = list(range(df_train.shape[0]))

    train_idx, val_idx = train_test_split(all_idx)

    for col in cat_cols:

        df_train[col] = df_train[col].astype('category').cat.as_ordered()

    apply_cats(df_test, df_train)

    df, y, nas, mapper = proc_df(df_train[feature_cols+["target"]], 'target', do_scale=True)

    df_test.reset_index(inplace=True)

    df_test["target"] = 0.0

    df_test_fi, _, nas, mapper = proc_df(df_test[feature_cols+["target"]], 'target', do_scale=True,

                                      mapper=mapper, na_dict=nas)

    md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_cols, bs=128,

                                           test_df=df_test_fi)

    categorical_col_data = [(c, len(df_train[c].cat.categories)+1) for c in cat_cols]

    embd_sz = [(c, min(50, (c+1)//2)) for _,c in categorical_col_data]

    m = md.get_learner(embd_sz, len(df.columns)-len(cat_cols),

                   0.04, 1, [1000,500], [0.001,0.01], tmp_name=PATH_TMP, models_name=PATH_MODEL)

    m.lr_find()

    m.sched.plot_lr()

    return m
#for trend_col in ["authorized_flag", "category_1", "city_id"]: #,"category_3","merchant_category_id","category_2","state_id","subsector_id","merchant_id_SNo"]:

#for trend_col in ["category_3","merchant_category_id","category_2"]: #,"state_id","subsector_id","merchant_id_SNo"]:

def gen_feature_for_history_trend_col(trend_col_list, feature_cols,in_df,  prefix):

    global df_test, df_train, key_col

    org_feature_cols = feature_cols

    for trend_col in trend_col_list:

        print(trend_col)

        dic_agg = {"purchase_amount": ["sum","count"],"installments":["sum"],"month_lag":["nunique","max"]}

        df1 = in_df.groupby([key_col,trend_col]).agg(dic_agg)

        df1.columns = [prefix + "_" + trend_col + "_" + col[0] + "_" + col[1] for col in df1.columns]

        dic_agg2 = {}

        for col in df1.columns:

            dic_agg2[col] = ["min","max","mean","std","skew"]

        df1.reset_index(inplace=True)

        df1.set_index(key_col, inplace=True)

        df2 = df1.groupby(df1.index).agg(dic_agg2)

        df2.columns = [col[0] + "_" + col[1] for col in df2.columns]

        tran_index = set(df2.index.values) & set(df_train.index.values)

        for col in df2.columns:

            df_train[col] = 0

            df_train.loc[tran_index, col] = df2.loc[tran_index,col]

        test_index = set(df2.index.values) & set(df_test.index.values)

        for col in df2.columns:

            df_test[col] = 0

            df_test.loc[test_index, col] = df2.loc[test_index,col]

        

        all_cols = list(df_train.columns)

        all_cols.remove("outlier")

        cr = df_train[all_cols].corr()

        cr1 = cr["target"]

        feature_cols = list(cr1 [ (cr1>=0.03) | (cr1 <= -0.03)].index)

        feature_cols.remove("target")

        for col in cr.columns:

            crcol = cr[col]

            highcrcol = list(crcol[crcol>0.65].index)

            if col in highcrcol:

                highcrcol.remove(col)

                for col1 in highcrcol:

                    if col1 != "target":

                        cor1 = cr.loc[col,"target"]

                        cor2 = cr.loc[col1,"target"]

                        if cor1 < cor2:

                            if col in feature_cols:

                                feature_cols.remove(col)

                        else:

                            if col1 in feature_cols:

                                feature_cols.remove(col1)

                

        for col in feature_cols:

            df_train[col].loc[~np.isfinite(df_train[col])] = 0

            df_test[col].loc[~np.isfinite(df_test[col])] = 0

            df_train[col] = df_train[col].fillna(0)

            df_test[col] = df_test[col].fillna(0) 



        cr1.sort_values()

        for col in df_train[all_cols].columns:

            if (col != "target") & (col not in feature_cols):

                df_train.drop(col, axis=1, inplace=True)

                if col in df_test.columns:

                    df_test.drop(col, axis=1, inplace=True)

    return feature_cols, list(set(feature_cols) - set(org_feature_cols))
df_train, df_test = load_data()

df_history, df_new, df_mer = load_tran_data()



reduce_mem_usage(df_train)

reduce_mem_usage(df_test)

reduce_mem_usage(df_history)

reduce_mem_usage(df_new)

reduce_mem_usage(df_mer)



convert_merchant_id()

convert_card_id()



check_duplicates(df_mer, "merchant_id_SNo", True)
print(df_history.shape)

print(df_mer.shape)

check_nulls(df_train, "train")

check_nulls(df_test, "test")

check_nulls(df_history, "history")

check_nulls(df_new, "new")

check_nulls(df_mer, "mer")
replace_null_with_most_freq_val(df_test, "first_active_month")



replace_null_with_most_freq_val(df_history, "category_3")

replace_null_with_most_freq_val(df_new, "category_3")



replace_null_with_most_freq_val(df_history, "category_2")

replace_null_with_most_freq_val(df_new, "category_2")

replace_null_with_most_freq_val(df_mer, "category_2")



null_lag3_index = list(df_mer[df_mer["avg_sales_lag3"].isna()].index)

df_mer.loc[null_lag3_index,"avg_sales_lag3"] = df_mer["avg_sales_lag3"].median()



null_lag6_index = list(df_mer[df_mer["avg_sales_lag6"].isna()].index)

df_mer.loc[null_lag6_index,"avg_sales_lag6"] = df_mer["avg_sales_lag6"].median()



null_lag12_index = list(df_mer[df_mer["avg_sales_lag12"].isna()].index)

df_mer.loc[null_lag12_index,"avg_sales_lag12"] = df_mer["avg_sales_lag12"].median()



#fix_missing_mer(df_history, df_mer)

#fix_missing_mer(df_new, df_mer)
df_train_outlier = explore_label()
add_datepart(df_train, "first_active_month", drop=True)

add_datepart(df_test, "first_active_month", drop=True)
label_col = "target"
cr = df_train.corr()

cr1 = cr["target"]

feature_cols = list(cr1 [ (cr1>=0.03) | (cr1 <= -0.03)].index)

feature_cols.remove("target")

feature_cols.remove("outlier")

for col in feature_cols:

    df_train[col] = df_train[col].fillna(0)

    df_test[col] = df_test[col].fillna(0)



cr1.sort_values()

for col in df_train.columns:

    if (col != "target") & (col != "outlier") & (col != "card_id_SNo") & (col not in feature_cols):

        df_train.drop(col, axis=1, inplace=True)

        if col in df_test.columns:

            df_test.drop(col, axis=1, inplace=True)
feature_cols
key_col = "card_id_SNo"

df_train.set_index(key_col, inplace=True)

df_test.set_index(key_col, inplace=True)
dflist = [df_history, df_new]

prefix_list = ["his","new"]

for i in range(len(dflist)):

    in_df = dflist[i]

    prefix = prefix_list[i]

    feature_cols, new_features = gen_feature_for_history_trend_col(["authorized_flag"], feature_cols, in_df, prefix)

    if 1==1:

        feature_cols, new_features = gen_feature_for_history_trend_col(["state_id","subsector_id","merchant_id_SNo"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["category_1"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["city_id"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["category_3"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["merchant_category_id"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["category_2"], feature_cols, in_df, prefix)



        in_df.drop(["state_id","subsector_id","authorized_flag","category_1","city_id","category_3","merchant_category_id","category_2"], axis=1, inplace=True)

        add_datepart(in_df, "purchase_date",drop=True)



        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Month"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Week"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Day"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Dayofweek"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Dayofyear"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Is_month_end"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Is_month_start"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Is_quarter_end"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Is_quarter_start"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Is_year_end"], feature_cols, in_df, prefix)

        feature_cols, new_features = gen_feature_for_history_trend_col(["purchase_Is_year_start"], feature_cols, in_df, prefix)



        in_df.drop(["purchase_Month","purchase_Week","purchase_Day","purchase_Dayofweek","purchase_Dayofyear",

                    "purchase_Is_month_end","purchase_Is_month_start","purchase_Is_quarter_end","purchase_Is_quarter_start",

                     "purchase_Is_year_end","purchase_Is_year_start"], axis=1, inplace=True)
df_his_mer = pd.merge(df_history[["card_id_SNo","merchant_id_SNo","purchase_amount","installments","month_lag"]], df_mer, how="inner", on="merchant_id_SNo")

col_list = df_his_mer.columns

col_list = list(set(col_list) - set(["card_id_SNo","merchant_id_SNo","purchase_amount","installments","month_lag"]))

for col in col_list:

    numval = len(list(df_his_mer[col].unique()))

    if numval <= 10:

        feature_cols, new_features = gen_feature_for_history_trend_col([col], feature_cols, df_his_mer, "his_mer")
df_new_mer = pd.merge(df_new[["card_id_SNo","merchant_id_SNo","purchase_amount","installments","month_lag"]], df_mer, how="inner", on="merchant_id_SNo")

col_list = df_new_mer.columns

col_list = list(set(col_list) - set(["card_id_SNo","merchant_id_SNo","purchase_amount","installments","month_lag"]))

for col in col_list:

    numval = len(list(df_his_mer[col].unique()))

    if numval <= 10:

        feature_cols, new_features = gen_feature_for_history_trend_col([col], feature_cols, df_new_mer, "new_mer")
X_train, X_valid, y_train, y_valid, X_test, X_all, y_all = get_X_Y(feature_cols)
pred_test2, valmse, model = get_keras_pred()
print("Loss", valmse)
pred_test1, valmse = get_lgbm_pred()
print("Loss", valmse)
fig,ax = plt.subplots(figsize=(12,12))

ax = sns.heatmap(df_train[["target"]  + feature_cols].corr(), ax=ax)
if 1==2:

    regr = LinearRegression()

    pred_test = get_sklearn_model(regr, X_train, X_valid, y_train, y_valid, X_test)

    #This scores 3.924
if 1==2:

    regr1 = LinearRegression()

    pred_test, feature_df = get_stratified_prediction(regr1, X_all_dummy, y_all_dummy, outlier_all_dummy, X_test_dummy, feature_cols_with_dummy)

    #This also scores 3.924
if 1==2:

    regr = Lasso(alpha=0.005, max_iter=1000)

    pred_test, feature_df = get_stratified_prediction(regr, X_all_dummy, y_all_dummy, outlier_all_dummy, X_test_dummy, feature_cols_with_dummy)

    #This also scores 3.924
if 1==2: #fast.ai is giving error

    m = get_fast_ai_pred()

    m.sched.plot()

    plt.tight_layout()

    plt.axvline(x=1.8e-2, color="red");

    m.fit(1e-2, 3) #, cycle_len=1, cycle_mult=2)

    pred_test=m.predict(True)
import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1, scoring="neg_mean_squared_error").fit(X_train,y_train)

eli5.show_weights(perm, feature_names = feature_cols, top=50)
#print("Loss", valmse)
df_test["target"] = list(pred_test1)

df_test[["card_id","target"]].to_csv("submission1.csv", index=False)

from IPython.display import FileLink

FileLink('submission1.csv')
df_test["target"] = list(pred_test2.flatten())

df_test[["card_id","target"]].to_csv("submission2.csv", index=False)

from IPython.display import FileLink

FileLink('submission1.csv')