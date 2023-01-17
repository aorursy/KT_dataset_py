import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.manifold import TSNE

import random

import tensorflow as tf

from tensorflow.keras.layers import *

!pip install tensorflow_addons==0.9.1

from tensorflow.keras.callbacks import Callback, LearningRateScheduler

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K

from tensorflow.keras import losses, models, optimizers

import tensorflow_addons as tfa

from sklearn.metrics import f1_score

from sklearn.model_selection import GroupKFold

import gc

import warnings

import os

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')
original_train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

print("train_data_shape : {}".format(original_train.shape))



plt.figure(figsize=(15,5), dpi=100)

plt.title("train_with_drift")

plt.plot(original_train["time"],original_train["signal"])

plt.ylabel("signal")

plt.xlabel("time")

plt.show()





train_clean = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

plt.figure(figsize=(15,5), dpi=100)

plt.title("train_without_drift")

plt.plot(train_clean["time"],train_clean["signal"])

plt.ylabel("signal")

plt.xlabel("time")

plt.show()



original_test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

print("test_data_shape : {}".format(original_test.shape))



plt.figure(figsize=(15,5), dpi=100)

plt.title("test_with_drift")

plt.plot(original_test["time"],original_test["signal"])

plt.ylabel("signal")

plt.xlabel("time")

plt.show()





test_clean = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

plt.figure(figsize=(15,5), dpi=100)

plt.title("test_without_drift")

plt.plot(test_clean["time"],test_clean["signal"])

plt.ylabel("signal")

plt.xlabel("time")

plt.show()
# ①変なnoiseはsiganlが0以上なのにopen_channelsが0でおかしいので改ざん

plt.figure(figsize=(15,5), dpi=100)

plt.title("check_train_anomary_signals")

fo = train_clean[train_clean["time"]<100]

plt.plot(fo["time"],fo["signal"])

plt.ylabel("signal")

plt.xlabel("time")

plt.show()



fig = plt.figure(figsize=(10,4), dpi=72)

tr_1 = fo[fo["open_channels"]==0]

ax1 = fig.add_subplot(1,2,1)

ax1.plot(tr_1["time"],tr_1["signal"],color="blue")

ax1.set_yticks(np.linspace(-5,5,5))

ax1.grid(axis = "x", color = "black",linestyle = "--", linewidth = 0.5)

ax1.grid(axis = "y", color = "black",linestyle = "--", linewidth = 0.5)

ax1.set_title("open_channel_0 (train_clean)", fontsize = 16)



tr_2 = fo[fo["open_channels"]==1]

ax2 = fig.add_subplot(1,2,2)

ax2.plot(tr_2["time"],tr_2["signal"],color="red")

ax2.set_yticks(np.linspace(-5,5,5))

ax2.grid(axis = "x", color = "black",linestyle = "--", linewidth = 0.5)

ax2.grid(axis = "y", color = "black",linestyle = "--", linewidth = 0.5)

ax2.set_title("open_channel_1 (train_clean)", fontsize = 16)

fig.tight_layout()

fig.legend()

plt.show()
for i in [478587, 478609, 478610, 599999]:

    train_clean.at[i,"signal"]=-2.7258763313293457

    

plt.figure(figsize=(15,5), dpi=100)

plt.title("process_A_done")

plt.plot(train_clean["time"],train_clean["signal"])

plt.ylabel("signal")

plt.xlabel("time")

plt.show()
fo = train_clean[(train_clean["time"]>=200)&(train_clean["time"]<300)]

color_array = ["blue","green","red","orangered",'sienna',"cyan","magenta","yellow","black","gray","lawngreen"]

for i in range(0,11):

    plt.scatter(fo[fo["open_channels"]==i]["time"],fo[fo["open_channels"]==i]["signal"],color=color_array[i],label="group_{}".format(str(i)))

plt.ylabel("signal")

plt.xlabel("time")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)

plt.show()



fo_normal = train_clean[(train_clean["time"]>=250)&(train_clean["time"]<300)]

fo_abnormal = train_clean[(train_clean["time"]>=200)&(train_clean["time"]<250)]

range_name = [i for i in range(11)]

range_box_normal = []

range_box_abnormal = []

for i in range(11):

    ma = round(np.max(fo_normal[fo_normal["open_channels"]==i]["signal"]),3)

    mi = round(np.min(fo_normal[fo_normal["open_channels"]==i]["signal"]),3)

    range_box_normal.append(str(mi)+" ~ "+str(ma))

    

    ma = round(np.max(fo_abnormal[fo_abnormal["open_channels"]==i]["signal"]),3)

    mi = round(np.min(fo_abnormal[fo_abnormal["open_channels"]==i]["signal"]),3)

    range_box_abnormal.append(str(mi)+" ~ "+str(ma))



range_df= pd.DataFrame()

range_df["Open_Channels"] = range_name

range_df["Signal_Range_Normal"] = range_box_normal

range_df["Signal_Range_Abnormal"] = range_box_abnormal

display(range_df)
inde = train_clean[(train_clean["time"]<=250)&(train_clean["time"]>200)].index

train_clean.loc[inde,"signal"] = train_clean.loc[inde,"signal"] + 2.7226785



inde = train_clean[(train_clean["time"]<=500)&(train_clean["time"]>450)].index

train_clean.loc[inde,"signal"] = train_clean.loc[inde,"signal"] + 2.7226785



inde = test_clean[(test_clean["time"]<=560)&(test_clean["time"]>550)].index

test_clean.loc[inde,"signal"] = test_clean.loc[inde,"signal"] + 2.7226785



inde = test_clean[(test_clean["time"]<=580)&(test_clean["time"]>570)].index

test_clean.loc[inde,"signal"] = test_clean.loc[inde,"signal"] + 2.7226785



plt.figure(figsize=(15,5), dpi=100)

plt.title("process_B_done")

plt.plot(train_clean["time"],train_clean["signal"])

plt.ylabel("signal")

plt.xlabel("time")

plt.show()
def create_signal_mod(train):

    left = 3641000

    right = 3829000

    thresh_dict = {

        3: [0.1, 2.0],

        2: [-1.1, 0.7],

        1: [-2.3, -0.6],

        0: [-3.8, -2],

    }

    

    train['signal_mod'] = train['signal'].values

    for ch in train[train['batch']==7]['open_channels'].unique():

        idxs_noisy = (train['open_channels']==ch) & (left<train.index) & (train.index<right)

        idxs_not_noisy = (train['open_channels']==ch) & ~idxs_noisy

        mean = train[idxs_not_noisy]['signal'].mean()



        idxs_outlier = idxs_noisy & (thresh_dict[ch][1]<train['signal'].values)

        train['signal_mod'][idxs_outlier]  = mean

        idxs_outlier = idxs_noisy & (train['signal'].values<thresh_dict[ch][0])

        train['signal_mod'][idxs_outlier]  = mean

    train["signal"] = train["signal_mod"]

    train = train.drop(["signal_mod","batch"],axis=1)

    return train



batch_list = []

for n in range(10):

    batchs = np.ones(500000)*n

    batch_list.append(batchs.astype(int))

batch_list = np.hstack(batch_list)

train_clean['batch'] = batch_list

train_clean = create_signal_mod(train_clean)



plt.figure(figsize=(15,5), dpi=100)

plt.title("process_C_done")

plt.plot(train_clean["time"],train_clean["signal"])

plt.ylabel("signal")

plt.xlabel("time")

plt.show()
def reduce_mem_usage(df: pd.DataFrame,

                     verbose: bool = True) -> pd.DataFrame:

    

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2



    for col in df.columns:

        col_type = df[col].dtypes



        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()



            if str(col_type)[:3] == 'int':



                if (c_min > np.iinfo(np.int32).min

                      and c_max < np.iinfo(np.int32).max):

                    df[col] = df[col].astype(np.int32)

                elif (c_min > np.iinfo(np.int64).min

                      and c_max < np.iinfo(np.int64).max):

                    df[col] = df[col].astype(np.int64)

            else:

                if (c_min > np.finfo(np.float16).min

                        and c_max < np.finfo(np.float16).max):

                    df[col] = df[col].astype(np.float16)

                elif (c_min > np.finfo(np.float32).min

                      and c_max < np.finfo(np.float32).max):

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    reduction = (start_mem - end_mem) / start_mem



    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'

    if verbose:

        print(msg)



    return df



train_clean = reduce_mem_usage(train_clean)

test_clean = reduce_mem_usage(test_clean)
def data_make(train,test):

    split_batch = 4000

    train['group'] = train.groupby(train.index//split_batch, sort=False)['signal'].agg(['ngroup']).values

    train['group'] = train['group'].astype(np.uint16)

    test['group'] = test.groupby(test.index//split_batch, sort=False)['signal'].agg(['ngroup']).values

    test['group'] = test['group'].astype(np.uint16)



    train_data = pd.DataFrame()

    

    tr_num = len(train["group"].value_counts())

    for gro in range(0,tr_num):

        try:

            name = str(gro)

            train_data[name] = train[train.group==gro]["signal"].values

        except:

            pass



    train_data = train_data.T

    train_data.columns = ["time_"+str(i) for i in range(0,split_batch)]



    te_num = len(test["group"].value_counts())

    test_data = pd.DataFrame()

    for gro in range(0,te_num):

        name = str(gro)

        test_data[name] = test[test.group==gro]["signal"].values



    test_data = test_data.T 

    test_data.columns = ["time_"+str(i) for i in range(0,split_batch)]

    

    train_data["index"] = train_data.index

    test_data["index"] = test_data.index

    return train_data,test_data



# グラフの可視化

def make_graph(data,data_length,boolen):

    tr_1 = data[:data_length, 0]

    tr_2 = data[:data_length, 1]

    te_1 = data[data_length:, 0]

    te_2 = data[data_length:, 1]

    if boolen:

        fig = plt.figure(figsize=(10,4), dpi=72)

        ax1 = fig.add_subplot(1,2,1)

        #ax2 = fig.add_subplot(1,2,3)



        ax1.scatter(tr_1,tr_2,color="blue",label="train")

        ax1.set_xticks(np.linspace(min(tr_1)*1.3,max(tr_1)*1.3, 5))

        ax1.set_yticks(np.linspace(min(tr_2)*1.3,max(tr_2)*1.3, 5))

        ax1.grid(axis = "x", color = "black",linestyle = "--", linewidth = 0.5)

        ax1.grid(axis = "y", color = "black",linestyle = "--", linewidth = 0.5)

        ax1.set_title("train_distribution", fontsize = 16)



    ax2 = fig.add_subplot(1,2,2)

    ax2.scatter(te_1,te_2,color="red",label="test")

    ax2.set_xticks(np.linspace(min(tr_1)*1.3,max(tr_1)*1.3, 5))

    ax2.set_yticks(np.linspace(min(tr_2)*1.3,max(tr_2)*1.3, 5))

    ax2.grid(axis = "x", color = "black",linestyle = "--", linewidth = 0.5)

    ax2.grid(axis = "y", color = "black",linestyle = "--", linewidth = 0.5)

    ax2.set_title("test_distribution", fontsize = 16)

    fig.tight_layout()

    fig.legend()

    plt.show()



    fig = plt.figure(figsize=(5.1,4), dpi=72)

    ax3 = fig.add_subplot(1,1,1)

    ax3.scatter(tr_1,tr_2,color="blue",label="train")

    ax3.scatter(te_1,te_2,color="red",label="test")

    ax3.set_xticks(np.linspace(min(tr_1)*1.3,max(tr_1)*1.3, 5))

    ax3.set_yticks(np.linspace(min(tr_2)*1.3,max(tr_2)*1.3, 5))

    ax3.grid(axis = "x", color = "black",linestyle = "--", linewidth = 0.5)

    ax3.grid(axis = "y", color = "black",linestyle = "--", linewidth = 0.5)

    ax3.set_title("train_test_distribution", fontsize = 16)

    fig.tight_layout()

    fig.legend()

    plt.show()
train_tsne,test_tsne = data_make(train_clean,test_clean)

data = pd.concat([train_tsne,test_tsne],axis=0).drop("index",axis=1)

tsne_data = TSNE(n_components=2,perplexity=30,learning_rate=300,random_state=1003).fit_transform(data)

make_graph(tsne_data,train_tsne.shape[0],True)
def make_feature(train,test):

    #4000でデータをバッチ化する

    train['group'] = train.groupby(train.index//4000, sort=False)['signal'].agg(['ngroup']).values

    train['group'] = train['group'].astype(np.uint16)

    test['group'] = test.groupby(test.index//4000, sort=False)['signal'].agg(['ngroup']).values

    test['group'] = test['group'].astype(np.uint16)

    

    train_proba = pd.read_csv('/kaggle/input/cleandataajustedproba/train_proba.csv')

    test_proba = pd.read_csv('/kaggle/input/cleandataajustedproba/test_proba.csv')



    train = pd.concat([train, train_proba], axis=1)

    test = pd.concat([test, test_proba], axis=1)

    

    #二乗要素追加

    #※open_channelsが0の時、signalが代替0になるように調節する

    train['signal**2'] = (train['signal']+2.694801092147827) ** 2

    test['signal**2'] = (test['signal']+2.694801092147827) ** 2



    #正規化

    mi = np.min(pd.concat([train["signal"],test["signal"]],axis=0))

    ma = np.max(pd.concat([train["signal"],test["signal"]],axis=0))

    train["signal"] = (train["signal"]-mi)/(ma - mi)

    test["signal"] = (test["signal"]-mi)/(ma - mi)

    mi = np.min(pd.concat([train["signal**2"],test["signal**2"]],axis=0))

    ma = np.max(pd.concat([train["signal**2"],test["signal**2"]],axis=0))

    train["signal**2"] = (train["signal**2"]-mi)/(ma - mi)

    test["signal**2"] = (test["signal**2"]-mi)/(ma - mi)

    

    

    #欠損値の補完

    def feature_selection(train, test):

        features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]

        train = train.replace([np.inf, -np.inf], np.nan)

        test = test.replace([np.inf, -np.inf], np.nan)

        for feature in features:

            feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()

            train[feature] = train[feature].fillna(feature_mean)

            test[feature] = test[feature].fillna(feature_mean)

        return train, test, features



    train,test,features = feature_selection(train,test)

    

    train = reduce_mem_usage(train)

    test = reduce_mem_usage(test)

    

    return train,test,features
train,test,features = make_feature(train_clean,test_clean)
#設定



EPOCHS = 1

NNBATCHSIZE = 16

GROUP_BATCH_SIZE = 4000

SEED = 321

LR = 0.0015

SPLITS = 2



def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)
random.seed(321)

np.random.seed(321)

os.environ['PYTHONHASHSEED'] = str(321)

tf.random.set_seed(321)

K.clear_session()

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)

tf.compat.v1.keras.backend.set_session(sess)

oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)

preds_ = np.zeros((len(test), 11))

target = ['open_channels']

ttt = train["open_channels"]

group = train['group']

#もちろんGfold

kf = GroupKFold(n_splits=SPLITS)



# train,test * 5分割

splits = [x for x in kf.split(train, train[target], group)]



new_splits = []

for sp in splits:

    # sp -> (train,test)

    new_split = []

    # trainで分けられたユニークなグループ

    new_split.append(np.unique(group[sp[0]]))

    # testで分けられたユニークなグループ

    new_split.append(np.unique(group[sp[1]]))

    # testで分けられたグループ

    new_split.append(sp[1])    

    new_splits.append(new_split)



tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)

tr.columns = ['target_'+str(i) for i in range(11)] + ['group']

target_cols = ['target_'+str(i) for i in range(11)]

train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)

#train : (Ntrain,4000,features)

train = np.array(list(train.groupby('group').apply(lambda x: x[features].values)))

#test : (Ntest,4000,features)

test = np.array(list(test.groupby('group').apply(lambda x: x[features].values)))
def Classifier(shape):

    

    def cbr(x, out_layer, kernel, stride, dilation):

        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        return x

    

    def wave_block(x, filters, kernel_size, n):

        # https://qiita.com/MasaEguchi/items/cd5f7e9735a120f27e2a

        # 層が深くなるにつれ、畳み込みのノードを離す（指数的に）

        dilation_rates = [2**i for i in range(n)]

        #Causal Conv

        x = Conv1D(filters=filters,kernel_size=1,padding = 'same')(x)

        res_x = x

        for dilation_rate in dilation_rates:

            tanh_out = Conv1D(filters = filters,

                              kernel_size = kernel_size,

                              padding = 'same', 

                              activation = 'tanh', 

                              dilation_rate = dilation_rate)(x)

            sigm_out = Conv1D(filters = filters,

                              kernel_size = kernel_size,

                              padding = 'same',

                              activation = 'sigmoid', 

                              dilation_rate = dilation_rate)(x)

            x = Multiply()([tanh_out, sigm_out])

            x = Conv1D(filters = filters,

                       kernel_size = 1,

                       padding = 'same')(x)

            res_x = Add()([res_x, x])

        return res_x

    

    inp = Input(shape = (shape_))

    x = wave_block(inp, 16, 3, 12)

    x = wave_block(x, 32, 3, 8)

    x = wave_block(x, 64, 3, 4)

    x = wave_block(x, 128, 3, 1)



    out = Dense(11, activation = 'softmax', name = 'out')(x)

    

    model = models.Model(inputs = inp, outputs = out)

    

    opt = Adam(lr = LR)

    opt = tfa.optimizers.SWA(opt)

    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])

    return model



def lr_schedule(epoch):

    if epoch < 30:

        lr = LR

    elif epoch < 40:

        lr = LR / 3

    elif epoch < 50:

        lr = LR / 5

    elif epoch < 60:

        lr = LR / 7

    elif epoch < 70:

        lr = LR / 9

    elif epoch < 80:

        lr = LR / 11

    elif epoch < 90:

        lr = LR / 13

    else:

        lr = LR / 100

    return lr



class MacroF1(Callback):

    def __init__(self, model, inputs, targets):

        self.model = model

        self.inputs = inputs

        self.targets = np.argmax(targets, axis = 2).reshape(-1)

        

    def on_epoch_end(self, epoch, logs):

        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)

        score = f1_score(self.targets, pred, average = 'macro')

        print(f'F1 Macro Score: {score:.5f}')
for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):

    train_x, train_y = train[tr_idx], train_tr[tr_idx]

    valid_x, valid_y = train[val_idx], train_tr[val_idx]

    gc.collect()

    shape_ = (None,len(features))#19:特徴量

    model = Classifier(shape_)

    #学習率を変化させる

    cb_lr_schedule = LearningRateScheduler(lr_schedule)

    hist = model.fit(train_x,train_y,

          epochs=EPOCHS,

          callbacks=[cb_lr_schedule, MacroF1(model, valid_x, valid_y)], # adding custom evaluation metric for each epoch

          batch_size=16,verbose = 2,

          validation_data = (valid_x,valid_y))

    preds_f = model.predict(valid_x)

    preds_f = preds_f.reshape(-1, preds_f.shape[-1])

    oof_[val_orig_idx,:] += preds_f

    te_preds = model.predict(test)

    te_preds = te_preds.reshape(-1, te_preds.shape[-1])

    preds_ += te_preds / SPLITS
f1_score_ = f1_score(ttt,  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)

print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')
compare_train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

compare_train["prediction"] = np.argmax(oof_,axis=1)



# 上記載のgroupを特徴量に入れる

compare_train["g"] = 0

num = 100000

compare_train.loc[0:10*num,"g"] = 0

compare_train.loc[10*num:15*num,"g"] = 3

compare_train.loc[15*num:20*num,"g"] = 1

compare_train.loc[20*num:25*num,"g"] = 4

compare_train.loc[25*num:30*num,"g"] = 2

compare_train.loc[30*num:35*num,"g"] = 3

compare_train.loc[35*num:40*num,"g"] = 1

compare_train.loc[40*num:45*num,"g"] = 2

compare_train.loc[45*num:50*num,"g"] = 4



for i in range(0,5):

    fo = compare_train[compare_train["g"]==i]

    score = f1_score(fo["open_channels"], fo["prediction"], average = 'micro')

    print("グループ{} : ".format(str(i)) + str(score))