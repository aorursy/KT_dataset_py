! pip install pyunpack PyWavelets pyts fire nvidia-ml-py3
%reload_ext autoreload

%autoreload 2

%matplotlib inline
! git clone https://github.com/timeseriesAI/timeseriesAI1  
! ln -s timeseriesAI1/fastai_timeseries .

! ln -s timeseriesAI1/torchtimeseries .
import fastai, os

from fastai_timeseries import *

from torchtimeseries.models import *

from fastai.callbacks import *

import random



path = Path('/kaggle/input/data-without-drift')



print('fastai :', fastai.__version__)

print('torch  :', torch.__version__)

print('device :', device)
#plotting fn from https://www.kaggle.com/miklgr500/ghost-drift-and-outliers

import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use('dark_background')
# https://www.kaggle.com/miklgr500/ghost-drift-and-outliers



def plot_open_channels_signal(df: pd.DataFrame, vline=[]):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    

    ax.plot(df.signal, df.open_channels, '.', color='fuchsia', alpha=0.25)

    for x in vline:

        ax.axvline(x, alpha=0.75, color='tomato')

    ax.set_xlabel('Signal')

    ax.set_ylabel('Open Channels')

    plt.show()

    

    

def plot_data(df: pd.DataFrame):

    if 'open_channels' in df.columns:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 16))

    

        ax2.plot(df.time, df.open_channels, color='royalblue', alpha=0.75)

        ax2.set_xlabel('time')

        ax2.set_ylabel('Open Channels')

    else:

        fig, ax1 = plt.subplots(1, 1, figsize=(24, 8))

    

    ax1.plot(df.time, df.signal, color='royalblue', alpha=0.75)

    ax1.set_xlabel('time')

    ax1.set_ylabel('Signal')

    plt.show()

file_tr = 'train_clean.csv'

file_tst = 'test_clean.csv'



df_train_all=pd.read_csv(path/file_tr,  ) 

df_test_all=pd.read_csv(path/file_tst,  )

#df_valid=pd.read_csv(path/file_val,  )
def remove_bad_signal(data):

    # https://www.kaggle.com/hirayukis/lightgbm-keras-and-4-kfold?scriptVersionId=32154310

    # read data

    #data = pd.read_csv('../input/data-without-drift/train_clean.csv')

    data.iloc[478587:478588, [1]] = -2  #reset spike siugnals x2

    data.iloc[478609:478610, [1]] = -2

    data_ = data[3500000:3642922].append(data[3822754:4000000])  # cut off error signal from DF

    data = data[:3500000].append(data[4000000:]).reset_index().append(data_, ignore_index=True)

    return data

    #data.head()

    #data[["signal", "open_channels"]].plot(figsize=(19,5), alpha=0.7)

plot_data(df_train_all)
plot_data(df_test_all)
#500k samples per group

gp_size = 500_000



for df in [df_train_all, df_test_all]:

  batches = df.shape[0] // gp_size

  df['batch'] = 0

  for i in range(batches):

        idx = np.arange(i*gp_size, (i+1)*gp_size)

        df.loc[idx, 'batch'] = i 
df_train_all = remove_bad_signal(df_train_all)  #remove bad signals from set 0 & 7
plot_data(df_train_all)
def get_db(train_list, valid_list, test_list,  scale_type, bs=1024):



  # selecting rows based on condition 

  df_train = df_train_all.loc[df_train_all['batch'].isin(train_list)] 

  df_valid = df_train_all.loc[df_train_all['batch'].isin(valid_list)] 



  df_test = df_test_all.loc[df_test_all['batch'].isin(test_list)]



  # split_by_df

  df_train['is_valid']=False

  df_valid['is_valid']=True

  df_combine = pd.concat([df_train, df_valid], axis=0, sort=False)



  offset = random.randint(0, 20000)

  train_size=df_train.shape[0]

  valid_size=df_valid.shape[0]

  test_size=df_test.shape[0]

  train_idx = 0  #offset + train_size

  valid_idx = 0  #valid_size + train_idx

  test_idx = 0  #test_size + valid_idx

  print ('training set= ',train_size, 'train index=', train_idx)

  print ('valid set= ',valid_size, 'valid index=', valid_idx)

  print('test set =', test_size, 'test index=', test_idx)

  print ('total length', test_size+train_size+valid_size)

  print ('Dataset= ', df_train.shape[0] )



  bs = bs                           # ✳️ orig 1024

  #seed = 8888                        # ✳️

  scale_type = scale_type          # ✳️ 

  scale_by_channel = True            # ✳️ 

  scale_by_sample  = False           # ✳️ 

  scale_range = (-1, 1)              # ✳️ 



  db = (TimeSeriesList.from_df(df_combine, '.', cols=["signal"],)  # feat='feat')

      #.split_by_idx(list(range(train_size, train_size+valid_size)) )

      .split_from_df(col='is_valid')

      .label_from_df(cols='open_channels', label_cls=CategoryList)

      .add_test(TimeSeriesList.from_df(df_test, '.', cols=["signal"]) )

      .databunch(bs=bs,  val_bs=bs,  num_workers=cpus,  device=device)

      .scale(scale_type=scale_type, scale_by_channel=scale_by_channel, 

             scale_by_sample=scale_by_sample,scale_range=scale_range)

     )

  return db, df_test
def main(

        epochs: 10,

        bs:    1024,

        runs:  1, 

        train_list: [],

        valid_list: [],

        test_list: [],

        scale_type: 'normalize', 

        ):





    global df_result, learn



    bs = bs                           # ✳️ orig 1024

    scale_type = scale_type          # ✳️ 



    

    # ResCNN, FCN, InceptionTime, ResNet

    arch = InceptionTime                     # ✳️   

    arch_kwargs = dict()           # 

   

    

    db, df_result = get_db(train_list, valid_list, test_list, scale_type, bs)

    print('# class= ', db.c, 'features= ', db.features)



    epochs = epochs         # ✳️ orig 100

    max_lr = 1e-2        # ✳️ orig 1e-2

    warmup = True       # ✳️ orig False

    pct_start = .7       # ✳️

    metrics = [accuracy] # ✳️

    wd = 1e-2



    

    for run in range(runs):

        print(f'Run: {run}')

        model = arch(db.features, db.c, **arch_kwargs).to(device)

        learn = Learner(db, model, opt_func=Ranger)



        learn.metrics = metrics

        learn.fit_fc(epochs, max_lr,  callbacks=[OverSamplingCallback(learn), SaveModelCallback(learn, monitor='accuracy')] ) 



        preds, tgt = learn.get_preds(ds_type=DatasetType.Test) # ds_type=DatasetType.Valid

        test_preds = preds.argmax(-1).view(-1).numpy()

        df_result[f'Run_{run}'] = test_preds



    
#trg set 0 & 7 have spikes

# 10 channel model

runs = 1

epochs = 3



kwargs = ( {'epochs': epochs, 'bs': 1024, 'runs': runs, 'train_list': [4], 'valid_list': [9], 'test_list': [1], 'scale_type': 'normalize' })  # 10 chan

main(**kwargs)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
df_mod1 = df_result.copy()
#trg set 0 & 7 have spikes

#5 channel model



kwargs = ( {'epochs': epochs, 'bs': 1024, 'runs': runs, 'train_list': [5], 'valid_list': [8], 'test_list': [0], 'scale_type': 'normalize' })  #up to 5 chan

main(**kwargs)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
df_mod2 = df_result.copy()
#trg set 0 & 7 have spikes

#3 channel model



kwargs = ( {'epochs': epochs, 'bs': 1024, 'runs': runs, 'train_list': [3], 'valid_list': [7], 'test_list': [2,3], 'scale_type': 'normalize' })  

main(**kwargs)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
df_mod3 = df_result.copy()
df_model = pd.concat([df_mod1, df_mod2, df_mod3], axis=0).sort_values(['time'])
if runs == 1 :

  df_vote = df_model[['Run_0']]

elif runs == 3 :

  df_vote = df_model[['Run_0', 'Run_1', 'Run_2']]

elif runs == 5 :  

  df_vote = df_model[['Run_0', 'Run_1', 'Run_2', 'Run_3', 'Run_4']]

else :

  print ("Error ! runs INCORRECT ! ", runs)

  
#use numba to run mode 4x faster !!

import numba

from numba import jit

from scipy import stats



# numba likes loop, np array & broadcasting



@jit

def mode_numba(df):  

    x = df.to_numpy()

    a = np.zeros(shape=x.shape[0])

    for i in range(x.shape[0]):

      a[i] = np.asscalar(stats.mode(x[i, :])[0] ) # index 0 gives class, index 1 gives freq

    

    return a.astype(int)
if runs == 1:

  df_model['vote'] = df_model['Run_0']

else :

  df_model['vote'] = mode_numba(df_vote) 
df_model[df_model['batch']==1]
path2 = Path('/kaggle/input/liverpool-ion-switching')

df_subm = pd.read_csv(path2/"sample_submission.csv")

df_subm['open_channels'] = df_model.vote.values

df_subm.to_csv("submissions.csv", float_format='%.4f', index=False)
df_model.vote.value_counts()
df_subm