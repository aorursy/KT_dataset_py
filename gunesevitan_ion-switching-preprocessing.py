import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression



from pykalman import KalmanFilter

from scipy import signal
df_train = pd.read_csv('../input/liverpool-ion-switching/train.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels': np.uint8})

df_test = pd.read_csv('../input/liverpool-ion-switching/test.csv', dtype={'time': np.float32, 'signal': np.float32})



print('Training Set Shape = {}'.format(df_train.shape))

print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))

print('Training Set Batches = {}'.format(int(len(df_train) / 500000)))

print('Test Set Shape = {}'.format(df_test.shape))

print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))

print('Test Set Batches = {}'.format(int(len(df_test) / 500000)))
BATCH_SIZE = 500000



for i in range(10):

    df_train.loc[i * BATCH_SIZE:((i + 1) * BATCH_SIZE) - 1, 'batch'] = i

    

for i in range(4):

    df_test.loc[i * BATCH_SIZE:((i + 1) * BATCH_SIZE) - 1, 'batch'] = i

    

df_train['batch'] = df_train['batch'].astype(np.uint8)

df_test['batch'] = df_test['batch'].astype(np.uint8)
fig = plt.figure(figsize=(15, 7))

sns.barplot(x=df_train['open_channels'].value_counts().index, y=df_train['open_channels'].value_counts().values)



plt.xlabel('Open Ion Channels', size=15, labelpad=20)

plt.ylabel('Value Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)

plt.title('Open Ion Channels Value Counts in Training Set', size=15)



plt.show()
training_batches = df_train.groupby('batch')



fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(20, 20), dpi=100)

for i, batch in training_batches:

    ax = plt.subplot(5, 2, i + 1) 

    

    sns.barplot(x=batch['open_channels'].value_counts().index, y=batch['open_channels'].value_counts().values)

    

    plt.xlabel('')

    plt.ylabel('')

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    ax.set_title(f'Batch {i} ({batch.index.min()}-{batch.index.max()}) Target Distribution', size=15)

    

plt.tight_layout()

plt.show()
def remove_ramp(X, constant=3):

    r = np.arange(len(X))        

    return X - ((r * constant) / len(X))
def remove_sine(X, constant=5):

    s = np.arange(len(X))        

    return X - (constant * (np.sin(np.pi * s / len(X))))
df_train['signal_processed'] = df_train['signal'].copy()

df_test['signal_processed'] = df_test['signal'].copy()
def report_training_batch(df, feature, batch):

    

    print(f'Training Batch {batch} - Unique Open Channel Values = {df[df["batch"] == batch]["open_channels"].unique()}')

    signal_openchannel_corr = np.corrcoef(df[df['batch'] == batch][feature], df[df['batch'] == batch]['open_channels'])[0][1]

    print(f'Training Batch {batch} - Correlation between Signal and Open Channels = {signal_openchannel_corr:.4}')

    print(f'Training Batch {batch} - Signal Mean = {df[df["batch"] == batch][feature].mean():.4} and Open Channels Mean = {df[df["batch"] == batch]["open_channels"].mean():.4}')

    print(f'Training Batch {batch} - Signal Std = {df[df["batch"] == batch][feature].std():.4} and Open Channels Std = {df[df["batch"] == batch]["open_channels"].std():.4}')

    print(f'Training Batch {batch} - Open Channels Range:')

    for value in df[df['batch'] == batch]['open_channels'].unique():

        print(f'                   Open Channels {value} - Min = {df.query("batch == @batch and open_channels == @value")[feature].min():.6} and Max = {df.query("batch == @batch and open_channels == @value")[feature].max():.6}')

    

    fig = plt.figure(figsize=(16, 6), dpi=100)

    df[df['batch'] == batch].set_index('time')[feature].plot(label='Signal')

    df[df['batch'] == batch].set_index('time')['open_channels'].plot(label='Open Channels')

        

    plt.xlabel('Time', size=15)

    plt.tick_params(axis='x', labelsize=12)

    plt.tick_params(axis='y', labelsize=12)

    plt.legend()

    title = f'Training Batch {batch} ({df[df["batch"] == batch].index.min()}-{df[df["batch"] == batch].index.max()})'

    plt.title(title, size=15)

    

    plt.show()

    

def report_test_batch(df, feature, batch):

    

    print(f'Test Batch {batch} - Signal Mean = {df[df["batch"] == batch][feature].mean():.4}')

    print(f'Test Batch {batch} - Signal Std = {df[df["batch"] == batch][feature].std():.4}')

    

    fig = plt.figure(figsize=(16, 6), dpi=100)

    df[df['batch'] == batch].set_index('time')[feature].plot(label='Signal')

        

    plt.xlabel('Time', size=15)

    plt.tick_params(axis='x', labelsize=12)

    plt.tick_params(axis='y', labelsize=12)

    plt.legend()

    title = f'Test Batch {batch} ({df[df["batch"] == batch].index.min()}-{df[df["batch"] == batch].index.max()})'

    plt.title(title, size=15)

    

    plt.show()

    
report_training_batch(df_train, 'signal', 0)
outlier_idx = df_train.query('signal > 0 and batch == 0').index

batch0_target0_mean = df_train.drop(outlier_idx).query('batch == 0 and open_channels == 0')['signal'].mean()

df_train.loc[outlier_idx, 'signal_processed'] = batch0_target0_mean
report_training_batch(df_train, 'signal_processed', 0)
report_training_batch(df_train, 'signal', 1)
batch1_slice = df_train.loc[df_train.query('time <= 60.0000 and batch == 1').index, 'signal']

df_train.loc[df_train.query('time <= 60.0000 and batch == 1').index, 'signal_processed'] = remove_ramp(batch1_slice)
report_training_batch(df_train, 'signal_processed', 1)
report_training_batch(df_train, 'signal', 2)
report_training_batch(df_train.loc[df_train.query('batch == 2 and 143 < time <= 144').index], 'signal', 2)
report_training_batch(df_train, 'signal', 3)
report_training_batch(df_train.loc[df_train.query('batch == 3 and 190 < time < 191').index], 'signal', 3)
report_training_batch(df_train, 'signal', 4)
report_training_batch(df_train.loc[df_train.query('batch == 4 and 239 < time <= 240').index], 'signal', 4)
report_training_batch(df_train, 'signal', 5)
report_training_batch(df_train.loc[df_train.query('batch == 5 and 295 < time <= 296').index], 'signal', 5)
report_training_batch(df_train, 'signal', 6)
batch6 = df_train.loc[df_train.query('batch == 6').index]['signal']

df_train.loc[df_train.query('batch == 6').index, 'signal_processed'] = remove_sine(batch6)
report_training_batch(df_train, 'signal_processed', 6)
report_training_batch(df_train, 'signal', 7)
batch7 = df_train.loc[df_train.query('batch == 7').index]['signal']

df_train.loc[df_train.query('batch == 7').index, 'signal_processed'] = remove_sine(batch7)
report_training_batch(df_train, 'signal_processed', 7)
df_train['is_filtered'] = 0

df_train['is_filtered'] = df_train['is_filtered'].astype(np.uint8)

batch7_outlier_idx = pd.Int64Index(range(3641000, 3829000))

df_train.loc[batch7_outlier_idx, 'is_filtered'] = 1
report_training_batch(df_train.drop(batch7_outlier_idx), 'signal_processed', 7)
open_channels0_mean = df_train[((df_train['batch'] == 3) | (df_train['batch'] == 7)) & (df_train['is_filtered'] == 0) & (df_train['open_channels'] == 0)]['signal_processed'].mean()

open_channels1_mean = df_train[((df_train['batch'] == 3) | (df_train['batch'] == 7)) & (df_train['is_filtered'] == 0) & (df_train['open_channels'] == 1)]['signal_processed'].mean()

open_channels2_mean = df_train[((df_train['batch'] == 3) | (df_train['batch'] == 7)) & (df_train['is_filtered'] == 0) & (df_train['open_channels'] == 2)]['signal_processed'].mean()

open_channels3_mean = df_train[((df_train['batch'] == 3) | (df_train['batch'] == 7)) & (df_train['is_filtered'] == 0) & (df_train['open_channels'] == 3)]['signal_processed'].mean()



df_train.loc[(df_train['is_filtered'] == 1) & (df_train['open_channels'] == 0), 'signal_processed'] = open_channels0_mean

df_train.loc[(df_train['is_filtered'] == 1) & (df_train['open_channels'] == 1), 'signal_processed'] = open_channels1_mean

df_train.loc[(df_train['is_filtered'] == 1) & (df_train['open_channels'] == 2), 'signal_processed'] = open_channels2_mean

df_train.loc[(df_train['is_filtered'] == 1) & (df_train['open_channels'] == 3), 'signal_processed'] = open_channels3_mean



batch7_filtered_part = df_train.loc[df_train['is_filtered'] == 1, 'signal_processed']

df_train.loc[df_train['is_filtered'] == 1, 'signal_processed'] = batch7_filtered_part + np.random.normal(0, 0.3, size=len(batch7_filtered_part)) 
report_training_batch(df_train, 'signal_processed', 7)
report_training_batch(df_train, 'signal', 8)
batch8 = df_train.loc[df_train.query('batch == 8').index]['signal']

df_train.loc[df_train.query('batch == 8').index, 'signal_processed'] = remove_sine(batch8)
report_training_batch(df_train, 'signal_processed', 8)
report_training_batch(df_train.loc[df_train.query('batch == 8 and 430 < time <= 431').index], 'signal_processed', 8)
report_training_batch(df_train, 'signal', 9)
batch9 = df_train.loc[df_train.query('batch == 9').index]['signal']

df_train.loc[df_train.query('batch == 9').index, 'signal_processed'] = remove_sine(batch9)
report_training_batch(df_train, 'signal_processed', 9)
report_training_batch(df_train.loc[df_train.query('batch == 9 and 470 < time <= 471').index], 'signal_processed', 9)
report_test_batch(df_test, 'signal', 0)
batch0_1 = df_test.loc[:100000 - 1, 'signal']

df_test.loc[:100000 - 1, 'signal_processed'] = remove_ramp(batch0_1)
report_test_batch(df_test.loc[:100000 - 1], 'signal_processed', 0)
batch0_2 = df_test.loc[100000:200000 - 1, 'signal']

df_test.loc[100000:200000 - 1, 'signal_processed'] = remove_ramp(batch0_2)
report_test_batch(df_test.loc[100000:200000 - 1], 'signal_processed', 0)
report_test_batch(df_test.loc[200000:300000 - 1], 'signal', 0)
report_test_batch(df_test.loc[300000:400000 - 1], 'signal', 0)
batch0_5 = df_test.loc[400000:500000 - 1, 'signal']

df_test.loc[400000:500000 - 1, 'signal_processed'] = remove_ramp(batch0_5)
report_test_batch(df_test.loc[400000:500000 - 1], 'signal_processed', 0)
report_test_batch(df_test, 'signal_processed', 0)
report_test_batch(df_test, 'signal', 1)
report_test_batch(df_test.loc[500000:600000 - 1], 'signal_processed', 1)
batch1_2 = df_test.loc[600000:700000 - 1, 'signal']

df_test.loc[600000:700000 - 1, 'signal_processed'] = remove_ramp(batch1_2)
report_test_batch(df_test.loc[600000:700000 - 1], 'signal_processed', 1)
batch1_3 = df_test.loc[700000:800000 - 1, 'signal']

df_test.loc[700000:800000 - 1, 'signal_processed'] = remove_ramp(batch1_3)
report_test_batch(df_test.loc[700000:800000 - 1], 'signal_processed', 1)
batch1_4 = df_test.loc[800000:900000 - 1, 'signal']

df_test.loc[800000:900000 - 1, 'signal_processed'] = remove_ramp(batch1_4)
report_test_batch(df_test.loc[800000:900000 - 1], 'signal_processed', 1)
report_test_batch(df_test.loc[900000:1000000 - 1], 'signal_processed', 1)
report_test_batch(df_test, 'signal_processed', 1)
report_test_batch(df_test, 'signal', 2)
batch2 = df_test.loc[df_test.query('batch == 2').index]['signal']

df_test.loc[df_test.query('batch == 2').index, 'signal_processed'] = remove_sine(batch2)
report_test_batch(df_test, 'signal_processed', 2)
report_test_batch(df_test.loc[df_test.query('batch == 2 and 624 < time <= 625').index], 'signal_processed', 2)
report_test_batch(df_test, 'signal', 3)
report_test_batch(df_test.loc[df_test.query('batch == 3 and 681 < time <= 682').index], 'signal_processed', 3)
# model 0

model0_trn_idx = df_train.query('batch == 0 or batch == 1').index

model0_tst_idx = df_test.query('batch == 0 and (500 < time <= 510)').index



df_test.loc[model0_tst_idx, 'model'] = 0

df_train.loc[model0_trn_idx, 'model'] = 0



# model 1

model1_trn_idx = df_train.query('batch == 2 or batch == 6').index

model1_tst_idx = df_test.query('batch == 0 and (540 < time <= 550)').index



df_train.loc[model1_trn_idx, 'model'] = 1

df_test.loc[model1_tst_idx, 'model'] = 1



# model 1.5

model15_tst_idx = df_test.query('(batch == 0 and (530 < time <= 540)) or (batch == 1 and (580 < time <= 590)) or batch == 2 or batch == 3').index

df_test.loc[model15_tst_idx, 'model'] = 1.5



# model 2

model2_trn_idx = df_train.query('batch == 3 or batch == 7').index

model2_tst_idx = df_test.query('(batch == 0 and (510 < time <= 520)) or (batch == 1 and (590 < time <= 600))').index



df_train.loc[model2_trn_idx, 'model'] = 2

df_test.loc[model2_tst_idx, 'model'] = 2



# model 3

model3_trn_idx = df_train.query('batch == 5 or batch == 8').index

model3_tst_idx = df_test.query('(batch == 0 and (520 < time <= 530)) or (batch == 1 and (560 < time <= 570))').index



df_train.loc[model3_trn_idx, 'model'] = 3

df_test.loc[model3_tst_idx, 'model'] = 3



# model 4

model4_trn_idx = df_train.query('batch == 4 or batch == 9').index

model4_tst_idx = df_test.query('(batch == 1 and (550 < time <= 560)) or (batch == 1 and (570 < time <= 580))').index



df_train.loc[model4_trn_idx, 'model'] = 4

df_test.loc[model4_tst_idx, 'model'] = 4



for model in [0, 1, 1.5, 2, 3, 4]:

    print(f'\n---------- Model {model} ----------\n')

    for batch in df_train[df_train['model'] == model]['batch'].unique():

        model_signal_mean = df_train[(df_train['model'] == model) & (df_train['batch'] == batch)]['signal_processed'].mean()

        model_signal_std = df_train[(df_train['model'] == model) & (df_train['batch'] == batch)]['signal_processed'].std()

        model_signal_min = df_train[(df_train['model'] == model) & (df_train['batch'] == batch)]['signal_processed'].min()

        model_signal_max = df_train[(df_train['model'] == model) & (df_train['batch'] == batch)]['signal_processed'].max()

        print(f'Training Set Model {model} Batch {batch} signal_processed mean = {model_signal_mean:.4}, std = {model_signal_std:.4}, range = {model_signal_min:.4} - {model_signal_max:.4}')



    for batch in df_test[df_test['model'] == model]['batch'].unique():

        model_signal_mean = df_test[(df_test['model'] == model) & (df_test['batch'] == batch)]['signal_processed'].mean()

        model_signal_std = df_test[(df_test['model'] == model) & (df_test['batch'] == batch)]['signal_processed'].std()

        model_signal_min = df_test[(df_test['model'] == model) & (df_test['batch'] == batch)]['signal_processed'].min()

        model_signal_max = df_test[(df_test['model'] == model) & (df_test['batch'] == batch)]['signal_processed'].max()

        print(f'Test Set Model {model} Batch {batch} signal_processed mean = {model_signal_mean:.4}, std = {model_signal_std:.4}, range = {model_signal_min:.4} - {model_signal_max:.4}')

        

print('\n---------- Training Set Model Value Counts ----------\n')

print(df_train['model'].value_counts())

print('\n---------- Test Set Model Value Counts ----------\n')

print(df_test['model'].value_counts())
fig, axes = plt.subplots(nrows=2, figsize=(20, 14), dpi=100)



df_train.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[0])

for batch in np.arange(0, 550, 50):

    axes[0].axvline(batch, color='r', linestyle='--', lw=2)

    

df_test.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[1])



for batch in np.arange(500, 600, 10):

    axes[1].axvline(batch, color='r', linestyle='--', lw=2)

for batch in np.arange(600, 700, 50):

    axes[1].axvline(batch, color='r', linestyle='--', lw=2)

    

axes[1].axvline(560, color='y', linestyle='dotted', lw=8)



for i in range(2):    

    for batch in np.arange(0, 550, 50):

        axes[i].axvline(batch, color='r', linestyle='--', lw=2)

        

    axes[i].set_xlabel('')

    axes[i].tick_params(axis='x', labelsize=15)

    axes[i].tick_params(axis='y', labelsize=15)

    axes[i].legend()

    

axes[0].set_title('Training Set Batches', size=18, pad=18)

axes[1].set_title('Public/Private Test Set Batches and Sub-batches', size=18, pad=18)



plt.show()
SHIFT_CONSTANT = np.exp(1)



df_train.loc[df_train['model'] == 4, 'signal_processed'] += SHIFT_CONSTANT

df_test.loc[df_test['model'] == 4, 'signal_processed'] += SHIFT_CONSTANT
fig, axes = plt.subplots(nrows=2, figsize=(20, 14), dpi=100)



df_train.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[0])

for batch in np.arange(0, 550, 50):

    axes[0].axvline(batch, color='r', linestyle='--', lw=2)

    

df_test.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[1])



for batch in np.arange(500, 600, 10):

    axes[1].axvline(batch, color='r', linestyle='--', lw=2)

for batch in np.arange(600, 700, 50):

    axes[1].axvline(batch, color='r', linestyle='--', lw=2)

    

axes[1].axvline(560, color='y', linestyle='dotted', lw=8)



for i in range(2):    

    for batch in np.arange(0, 550, 50):

        axes[i].axvline(batch, color='r', linestyle='--', lw=2)

        

    axes[i].set_xlabel('')

    axes[i].tick_params(axis='x', labelsize=15)

    axes[i].tick_params(axis='y', labelsize=15)

    axes[i].legend()

    

axes[0].set_title('Training Set Batches without Ghost Drift', size=18, pad=18)

axes[1].set_title('Public/Private Test Set Batches and Sub-batches without Ghost Drift', size=18, pad=18)



plt.show()
df_train['signal_processed_denoised'] = df_train['signal_processed'].copy(deep=True)

df_test['signal_processed_denoised'] = df_test['signal_processed'].copy(deep=True)



# Clean parts of training set signal

signalA = df_train[df_train['batch'] != 7]['signal_processed_denoised'].values

channelsA = df_train[df_train['batch'] != 7]['open_channels'].values



# Replacing a hidden outlier

signal1 = signalA[:1000000]

channels1 = channelsA[:1000000]

median = np.median(signal1[channels1 == 0])

condition = (signal1 > -1) & (channels1 == 0)

signal1[condition] = median

signalA[:1000000] = signal1



# Batch 7 first clean part and second clean part separated

signalB_good1 = df_train.loc[3_500_000:3_642_932 - 1]['signal_processed_denoised'].values

signalB_good2 = df_train.loc[3_822_753 + 1:4_000_000 - 1]['signal_processed_denoised'].values

channelsB_good1 = df_train.loc[3_500_000:3_642_932 - 1]['open_channels'].values

channelsB_good2 = df_train.loc[3_822_753 + 1:4_000_000 - 1]['open_channels'].values



# Test set signal and Bidirectional Viterbi predictions

signalC = df_test['signal_processed_denoised'].values

channelsC = pd.read_csv('../input/ion-switching-0945-predictions/gbdt_blend_submission.csv')['open_channels'].astype(np.uint8)
label = np.arange(len(signalA))

channel_list = np.arange(11)

n_list = np.empty(11)

mean_list = np.empty(11)

std_list = np.empty(11)

stderr_list = np.empty(11)



fig, axes = plt.subplots(ncols=2, figsize=(25, 8), dpi=100)



for i in range(11):

    x = label[channelsA == i]

    y = signalA[channelsA == i]

    n_list[i] = np.size(y)

    mean_list[i] = np.mean(y)

    std_list[i] = np.std(y)

    

    axes[0].plot(x, y, '.', markersize=0.5, alpha=0.02)    

    axes[0].tick_params(axis='x', labelsize=15)

    axes[0].tick_params(axis='y', labelsize=15)

    axes[0].set_title('Training Set Signal Processed Open Channels', size=18, pad=18)

    

stderr_list = std_list / np.sqrt(n_list)

sample_weight = 1 / stderr_list

channel_list = channel_list.reshape(-1, 1)



lr = LinearRegression()

lr.fit(channel_list, mean_list, sample_weight=sample_weight)

mean_predictA = lr.predict(channel_list)



x = np.linspace(-0.5, 10.5, 5)

y = lr.predict(x.reshape(-1, 1))

axes[1].plot(x, y, label='Predicted Means')

axes[1].plot(channel_list, mean_list, '.', markersize=8, label='Original Means')

axes[1].legend()



axes[1].tick_params(axis='x', labelsize=15)

axes[1].tick_params(axis='y', labelsize=15)

axes[1].set_title('Training Set Signal Processed Means', size=18, pad=18)



print('Predicted Means of signalA (Training Clean Signal):\n', mean_predictA)

plt.show()
def remove_target_mean(signal, target, means):

    signal_out = signal.copy()

    for i in range(11):

        signal_out[target == i] -= means[i]

    return signal_out



sig_A = remove_target_mean(signalA, channelsA, mean_predictA)

sig_B1 = remove_target_mean(signalB_good1, channelsB_good1, mean_predictA)

sig_B2 = remove_target_mean(signalB_good2, channelsB_good2, mean_predictA)

sig_C = remove_target_mean(signalC, channelsC, mean_predictA)
fig, axes = plt.subplots(nrows=3, figsize=(25, 20), dpi=100)

axes[0].plot(sig_A, linewidth=1)

axes[1].plot(np.hstack((sig_B1, sig_B2)), linewidth=1)

axes[2].plot(sig_C, linewidth=1)



for i in range(3):

    axes[i].tick_params(axis='x', labelsize=12)

    axes[i].tick_params(axis='y', labelsize=12)

axes[0].set_title('Training Set (Without Batch 7) Signal Processed After Means are Subtracted', size=18, pad=18)

axes[1].set_title('Training Set Batch 7 (Without Noisy Part) Signal Processed After Means are Subtracted', size=18, pad=18)

axes[2].set_title('Test Set Signal Processed After Means are Subtracted', size=18, pad=18)



plt.show()
def bandstop(x, samplerate=1000000, fp=np.array([4925, 5075]), fs=np.array([4800, 5200])):    

    fn = samplerate / 2

    wp = fp / fn

    ws = fs / fn

    gpass = 1

    gstop = 10.0



    N, Wn = signal.buttord(wp, ws, gpass, gstop)

    b, a = signal.butter(N, Wn, 'bandstop')

    y = signal.filtfilt(b, a, x)

    return y



def bandpass(x, samplerate=1000000, fp=np.array([4925, 5075]), fs=np.array([4800, 5200])):

    fn = samplerate / 2

    wp = fp / fn

    ws = fs / fn

    gpass = 1

    gstop = 10.0



    N, Wn = signal.buttord(wp, ws, gpass, gstop)

    b, a = signal.butter(N, Wn, "bandpass")

    y = signal.filtfilt(b, a, x)

    return y
train_normalized_signals = np.split(sig_A, 9)

train_original_signals = np.split(signalA, 9)

train_filtered_signals = []

train_supervised_noise = []



# Denoising training batches except Batch 7

for batch, original_signal in enumerate(train_original_signals):

    

    normalized_signal = train_normalized_signals[batch]    

    filtered_signal = bandstop(normalized_signal)

    noise = bandpass(normalized_signal)

    

    if batch >= 7:

        batch += 1

    

    plt.figure(figsize=(25, 5))

    plt.title(f'Open Channels Denormalized - Training Batch {batch}', size=18, pad=18)

    plt.tick_params(axis='x', labelsize=12)

    plt.tick_params(axis='y', labelsize=12)

    plt.plot(normalized_signal, label='Original Signal', linewidth=0.5, alpha=0.5)

    plt.plot(filtered_signal, label = 'Filtered Signal', linewidth=0.5, alpha=0.5)

    plt.show()

    

    clean_signal = original_signal - noise

    plt.figure(figsize=(25, 5))

    plt.title(f'Signal Space - Training Batch {batch}', size=18, pad=18)

    plt.tick_params(axis='x', labelsize=12)

    plt.tick_params(axis='y', labelsize=12)

    plt.plot(original_signal, linewidth=0.5, alpha=0.5)

    plt.plot(clean_signal, linewidth=0.5, alpha=0.5)

    plt.show()



    plt.figure(figsize=(25, 5))

    plt.title(f'Signal Space - Training Batch {batch}', size=18, pad=18)

    plt.tick_params(axis='x', labelsize=12)

    plt.tick_params(axis='y', labelsize=12)

    plt.plot(original_signal, linewidth=0.5, alpha=0.5)

    plt.twinx()

    plt.plot(filtered_signal, linewidth=0.5, alpha=0.5, c='orange')

    plt.show()

   

    train_filtered_signals.append(clean_signal)

    train_supervised_noise.append(noise)

    

# Denoising Batch 7 Part 1

batch7_filtered_signal1 = bandstop(sig_B1)

batch7_noise1 = bandpass(sig_B1)



plt.figure(figsize=(25, 5))

plt.title(f'Open Channels Denormalized - Training Batch 7 Part 1', size=18, pad=18)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.plot(sig_B1, label='Batch 7 Normalized Signal Part 1', linewidth=0.5, alpha=0.5)

plt.plot(batch7_filtered_signal1, label = 'Batch 7 Filtered Signal Part 1', linewidth=0.5, alpha=0.5)

plt.show()

    

batch7_clean_signal1 = signalB_good1 - batch7_noise1

plt.figure(figsize=(25, 5))

plt.title(f'Batch 7 Signal Space - Part 1', size=18, pad=18)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.plot(signalB_good1, linewidth=0.5, alpha=0.5)

plt.plot(batch7_filtered_signal1, linewidth=0.5, alpha=0.5)

plt.show()



plt.figure(figsize=(25, 5))

plt.title(f'Batch 7 Signal Space - Part 1', size=18, pad=18)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.plot(signalB_good1, linewidth=0.5, alpha=0.5)

plt.twinx()

plt.plot(batch7_filtered_signal1, linewidth=0.5, alpha=0.5, c='orange')

plt.show()



# Denoising Batch 7 Part 2

batch7_filtered_signal2 = bandstop(sig_B2)

batch7_noise2 = bandpass(sig_B2)



plt.figure(figsize=(25, 5))

plt.title(f'Open Channels Denormalized - Training Batch 7 Part 2', size=18, pad=18)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.plot(sig_B2, label='Batch 7 Normalized Signal Part 2', linewidth=0.5, alpha=0.5)

plt.plot(batch7_filtered_signal2, label = 'Batch 7 Filtered Signal Part 2', linewidth=0.5, alpha=0.5)

plt.show()

    

batch7_clean_signal2 = signalB_good2 - batch7_noise2

plt.figure(figsize=(25, 5))

plt.title(f'Batch 7 Signal Space - Part 2', size=18, pad=18)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.plot(signalB_good2, linewidth=0.5, alpha=0.5)

plt.plot(batch7_filtered_signal2, linewidth=0.5, alpha=0.5)

plt.show()



plt.figure(figsize=(25, 5))

plt.title(f'Batch 7 Signal Space - Part 2', size=18, pad=18)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.plot(signalB_good2, linewidth=0.5, alpha=0.5)

plt.twinx()

plt.plot(batch7_filtered_signal2, linewidth=0.5, alpha=0.5, c='orange')

plt.show()

    

df_train.loc[df_train['batch'] == 0, 'signal_processed_denoised'] = train_filtered_signals[0]

df_train.loc[df_train['batch'] == 1, 'signal_processed_denoised'] = train_filtered_signals[1]

df_train.loc[df_train['batch'] == 2, 'signal_processed_denoised'] = train_filtered_signals[2]

df_train.loc[df_train['batch'] == 3, 'signal_processed_denoised'] = train_filtered_signals[3]

df_train.loc[df_train['batch'] == 4, 'signal_processed_denoised'] = train_filtered_signals[4]

df_train.loc[df_train['batch'] == 5, 'signal_processed_denoised'] = train_filtered_signals[5]

df_train.loc[df_train['batch'] == 6, 'signal_processed_denoised'] = train_filtered_signals[6]

df_train.loc[df_train['batch'] == 6, 'signal_processed_denoised'] = train_filtered_signals[6]

df_train.loc[3500000:3642932 - 1, 'signal_processed_denoised'] = batch7_clean_signal1

df_train.loc[3822753 + 1:4000_000 - 1, 'signal_processed_denoised'] = batch7_clean_signal2

df_train.loc[df_train['batch'] == 8, 'signal_processed_denoised'] = train_filtered_signals[7]

df_train.loc[df_train['batch'] == 9, 'signal_processed_denoised'] = train_filtered_signals[8]
test_normalized_signals1 = np.split(sig_C[:1000000], 10)

test_original_signals1 = np.split(signalC[:1000000], 10)

test_normalized_signal2 = sig_C[1000000:]

test_original_signal2 = signalC[1000000:]

test_filtered_signals = []

test_supervised_noise = []



# Denoising test set sub batches part by part

for sub_batch, original_signal in enumerate(test_original_signals1):

    

    normalized_signal = test_normalized_signals1[sub_batch]    

    filtered_signal = bandstop(normalized_signal)

    noise = bandpass(normalized_signal)

        

    plt.figure(figsize=(25, 5))

    plt.title(f'Open Channels Denormalized - Test Sub-Batch {sub_batch}', size=18, pad=18)

    plt.tick_params(axis='x', labelsize=12)

    plt.tick_params(axis='y', labelsize=12)

    plt.plot(normalized_signal, label='Original Signal', linewidth=0.5, alpha=0.5)

    plt.plot(filtered_signal, label = 'Filtered Signal', linewidth=0.5, alpha=0.5)

    plt.show()

    

    clean_signal = original_signal - noise

    plt.figure(figsize=(25, 5))

    plt.title(f'Signal Space - Test Sub-Batch {sub_batch}', size=18, pad=18)

    plt.tick_params(axis='x', labelsize=12)

    plt.tick_params(axis='y', labelsize=12)

    plt.plot(original_signal, linewidth=0.5, alpha=0.5)

    plt.plot(clean_signal, linewidth=0.5, alpha=0.5)

    plt.show()



    plt.figure(figsize=(25, 5))

    plt.title(f'Signal Space - Test Sub-Batch {sub_batch}', size=18, pad=18)

    plt.tick_params(axis='x', labelsize=12)

    plt.tick_params(axis='y', labelsize=12)

    plt.plot(original_signal, linewidth=0.5, alpha=0.5)

    plt.twinx()

    plt.plot(filtered_signal, linewidth=0.5, alpha=0.5, c='orange')

    plt.show()

   

    test_filtered_signals.append(clean_signal)

    test_supervised_noise.append(noise)

        

# Denoising test set second half

test_filtered_signal2 = bandstop(test_normalized_signal2)

test_noise2 = bandpass(test_normalized_signal2)



plt.figure(figsize=(25, 5))

plt.title(f'Open Channels Denormalized - Test Batch 2 & 3 (Second Half)', size=18, pad=18)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.plot(test_normalized_signal2, label='Original Signal', linewidth=0.5, alpha=0.5)

plt.plot(test_filtered_signal2, label = 'Filtered Signal', linewidth=0.5, alpha=0.5)

plt.show()



test_clean_signal2 = test_original_signal2 - test_noise2

plt.figure(figsize=(25, 5))

plt.title(f'Signal Space - Test Batch 2 & 3 (Second Half)', size=18, pad=18)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.plot(test_original_signal2, linewidth=0.5, alpha=0.5)

plt.plot(test_clean_signal2, linewidth=0.5, alpha=0.5)

plt.show()



plt.figure(figsize=(25, 5))

plt.title(f'Signal Space - Test Batch 2 & 3 (Second Half)', size=18, pad=18)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.plot(test_original_signal2, linewidth=0.5, alpha=0.5)

plt.twinx()

plt.plot(test_filtered_signal2, linewidth=0.5, alpha=0.5, c='orange')

plt.show()



df_test.loc[0:100000 - 1, 'signal_processed_denoised'] = test_filtered_signals[0]

df_test.loc[100000:200000 - 1, 'signal_processed_denoised'] = test_filtered_signals[1]

df_test.loc[200000:300000 - 1, 'signal_processed_denoised'] = test_filtered_signals[2]

df_test.loc[300000:400000 - 1, 'signal_processed_denoised'] = test_filtered_signals[3]

df_test.loc[400000:500000 - 1, 'signal_processed_denoised'] = test_filtered_signals[4]

df_test.loc[500000:600000 - 1, 'signal_processed_denoised'] = test_filtered_signals[5]

df_test.loc[600000:700000 - 1, 'signal_processed_denoised'] = test_filtered_signals[6]

df_test.loc[700000:800000 - 1, 'signal_processed_denoised'] = test_filtered_signals[7]

df_test.loc[800000:900000 - 1, 'signal_processed_denoised'] = test_filtered_signals[8]

df_test.loc[900000:1000000 - 1, 'signal_processed_denoised'] = test_filtered_signals[9]

df_test.loc[1000000:, 'signal_processed_denoised'] = test_clean_signal2
fig, axes = plt.subplots(nrows=2, figsize=(20, 14), dpi=100)



df_train.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[0], alpha=0.5)

df_train.set_index('time')['signal_processed_denoised'].plot(label='Signal Denoised', ax=axes[0], alpha=0.5)

for batch in np.arange(0, 550, 50):

    axes[0].axvline(batch, color='r', linestyle='--', lw=2)

    

df_test.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[1], alpha=0.5)

df_test.set_index('time')['signal_processed_denoised'].plot(label='Signal Denoised', ax=axes[1], alpha=0.5)



for batch in np.arange(500, 600, 10):

    axes[1].axvline(batch, color='r', linestyle='--', lw=2)

for batch in np.arange(600, 700, 50):

    axes[1].axvline(batch, color='r', linestyle='--', lw=2)

    

axes[1].axvline(560, color='y', linestyle='dotted', lw=8)



for i in range(2):    

    for batch in np.arange(0, 550, 50):

        axes[i].axvline(batch, color='r', linestyle='--', lw=2)

        

    axes[i].set_xlabel('Time', size=15)

    axes[i].tick_params(axis='x', labelsize=12)

    axes[i].tick_params(axis='y', labelsize=12)

    axes[i].legend()

    

axes[0].set_title('Training Set Batches Processed and Denoised', size=18, pad=18)

axes[1].set_title('Public/Private Test Set Batches and Sub-batches Processed and Denoised', size=18, pad=18)



plt.show()
scaler = MinMaxScaler()



for i, batch in enumerate(df_train.groupby('batch')):

    time = df_train.loc[(df_train['batch'] == i), 'time'].values.reshape(-1, 1)

    df_train.loc[(df_train['batch'] == i), 'time_scaled'] = scaler.fit_transform(time)



for i, batch in enumerate(df_test.groupby('batch')):

    time = df_test.loc[(df_test['batch'] == i), 'time'].values.reshape(-1, 1)

    df_test.loc[(df_test['batch'] == i), 'time_scaled'] = scaler.fit_transform(time)

    

df_train['time_scaled'] = df_train['time_scaled'].astype(np.float32)

df_test['time_scaled'] = df_test['time_scaled'].astype(np.float32)
def kalman(signal, signal_covariance):

    

    kf = KalmanFilter(initial_state_mean=signal[0], 

                      initial_state_covariance=signal_covariance,

                      observation_covariance=signal_covariance, 

                      transition_covariance=0.1,

                      transition_matrices=1)

    

    pred_state, state_cov = kf.smooth(signal)

    return pred_state
%%time



# filter model 0

print(f'\n---------- Model 0 ----------\n')



batch0_corr = np.corrcoef(df_train[df_train['batch'] == 0]['signal_processed'], df_train[df_train['batch'] == 0]['open_channels'])[0][1]

df_train.loc[df_train['batch'] == 0, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 0]['signal_processed'].values, 0.6)

filtered_batch0_corr = np.corrcoef(df_train[df_train['batch'] == 0]['signal_processed_kalman'], df_train[df_train['batch'] == 0]['open_channels'])[0][1]

print(f'Training Batch 0 - Correlation between Signal and Open Channels increased from {batch0_corr:.6} to {filtered_batch0_corr:.6} (Covariance: {0.6})')



batch1_corr = np.corrcoef(df_train[df_train['batch'] == 1]['signal_processed'], df_train[df_train['batch'] == 1]['open_channels'])[0][1]

df_train.loc[df_train['batch'] == 1, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 1]['signal_processed'].values, 0.45)

filtered_batch1_corr = np.corrcoef(df_train[df_train['batch'] == 1]['signal_processed_kalman'], df_train[df_train['batch'] == 1]['open_channels'])[0][1]

print(f'Training Batch 1 - Correlation between Signal and Open Channels increased from {batch1_corr:.6} to {filtered_batch1_corr:.6} (Covariance: {0.45})')



print(f'Test Batch 0 Sub-batch 0 (Mean Model 0 Covariance: {0.525})')

df_test.loc[df_test.query('batch == 0 and (500 < time <= 510)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 0 and (500 < time <= 510)').index, 'signal_processed'].values, 0.525)



# filter model 1

print(f'\n---------- Model 1 ----------\n')



batch2_corr = np.corrcoef(df_train[df_train['batch'] == 2]['signal_processed'], df_train[df_train['batch'] == 2]['open_channels'])[0][1]

df_train.loc[df_train['batch'] == 2, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 2]['signal_processed'].values, 0.03)

filtered_batch2_corr = np.corrcoef(df_train[df_train['batch'] == 2]['signal_processed_kalman'], df_train[df_train['batch'] == 2]['open_channels'])[0][1]

print(f'Training Batch 2 - Correlation between Signal and Open Channels increased from {batch2_corr:.6} to {filtered_batch2_corr:.6} (Covariance: {0.03})')



batch6_corr = np.corrcoef(df_train[df_train['batch'] == 6]['signal_processed'], df_train[df_train['batch'] == 6]['open_channels'])[0][1]

df_train.loc[df_train['batch'] == 6, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 6]['signal_processed'].values, 0.03)

filtered_batch6_corr = np.corrcoef(df_train[df_train['batch'] == 6]['signal_processed_kalman'], df_train[df_train['batch'] == 6]['open_channels'])[0][1]

print(f'Training Batch 6 - Correlation between Signal and Open Channels increased from {batch6_corr:.6} to {filtered_batch6_corr:.6} (Covariance: {0.03})')



print(f'Test Batch 0 Sub-batch 4 (Mean Model 1 Covariance: {0.03})')

df_test.loc[df_test.query('batch == 0 and (540 < time <= 550)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 0 and (540 < time <= 550)').index, 'signal_processed'].values, 0.03)



# filter model 1.5

print(f'\n---------- Model 1.5 ----------\n')



print(f'Test Batch 0 Sub-batch 3 (Covariance: {0.1})')

df_test.loc[df_test.query('batch == 0 and (530 < time <= 540)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 0 and (530 < time <= 540)').index, 'signal_processed'].values, 0.1)



print(f'Test Batch 1 Sub-batch 3 (Covariance: {0.1})')

df_test.loc[df_test.query('batch == 1 and (580 < time <= 590)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 1 and (580 < time <= 590)').index, 'signal_processed'].values, 0.1)



print(f'Test Batch 2 (Covariance: {0.1})')

df_test.loc[df_test.query('batch == 2').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 2').index, 'signal_processed'].values, 0.1)

print(f'Test Batch 3 (Covariance: {0.1})')

df_test.loc[df_test.query('batch == 3').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 3').index, 'signal_processed'].values, 0.1)



# filter model 2

print(f'\n---------- Model 2 ----------\n')



batch3_corr = np.corrcoef(df_train[df_train['batch'] == 3]['signal_processed'], df_train[df_train['batch'] == 3]['open_channels'])[0][1]

df_train.loc[df_train['batch'] == 3, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 3]['signal_processed'].values, 0.01)

filtered_batch3_corr = np.corrcoef(df_train[df_train['batch'] == 3]['signal_processed_kalman'], df_train[df_train['batch'] == 3]['open_channels'])[0][1]

print(f'Training Batch 3 - Correlation between Signal and Open Channels increased from {batch3_corr:.6} to {filtered_batch3_corr:.6} (Covariance: {0.01})')



batch7_corr = np.corrcoef(df_train[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1)]['signal_processed'], df_train[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1)]['open_channels'])[0][1]

df_train.loc[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1), 'signal_processed_kalman'] = kalman(df_train[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1)]['signal_processed'].values, 0.01)

filtered_batch7_corr = np.corrcoef(df_train[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1)]['signal_processed_kalman'], df_train[(df_train['batch'] == 7) & (df_train['is_filtered'] != 1)]['open_channels'])[0][1]

print(f'Training Batch 7 - Correlation between Signal and Open Channels increased from {batch7_corr:.6} to {filtered_batch7_corr:.6} (Covariance: {0.01})')



print(f'Test Batch 0 Sub-batch 1 (Mean Model 2 Covariance: {0.01})')

df_test.loc[df_test.query('batch == 0 and (510 < time <= 520)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 0 and (510 < time <= 520)').index, 'signal_processed'].values, 0.01)

print(f'Test Batch 1 Sub-batch 4 (Mean Model 2 Covariance: {0.01})')

df_test.loc[df_test.query('batch == 1 and (590 < time <= 600)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 1 and (590 < time <= 600)').index, 'signal_processed'].values, 0.01)



# filter model 3

print(f'\n---------- Model 3 ----------\n')



batch5_corr = np.corrcoef(df_train[df_train['batch'] == 5]['signal_processed'], df_train[df_train['batch'] == 5]['open_channels'])[0][1]

df_train.loc[df_train['batch'] == 5, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 5]['signal_processed'].values, 0.005)

filtered_batch5_corr = np.corrcoef(df_train[df_train['batch'] == 5]['signal_processed_kalman'], df_train[df_train['batch'] == 5]['open_channels'])[0][1]

print(f'Training Batch 5 - Correlation between Signal and Open Channels increased from {batch5_corr:.6} to {filtered_batch5_corr:.6} (Covariance: {0.005})')



batch8_corr = np.corrcoef(df_train[df_train['batch'] == 8]['signal_processed'], df_train[df_train['batch'] == 8]['open_channels'])[0][1]

df_train.loc[df_train['batch'] == 8, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 8]['signal_processed'].values, 0.005)

filtered_batch8_corr = np.corrcoef(df_train[df_train['batch'] == 8]['signal_processed_kalman'], df_train[df_train['batch'] == 8]['open_channels'])[0][1]

print(f'Training Batch 8 - Correlation between Signal and Open Channels increased from {batch8_corr:.6} to {filtered_batch8_corr:.6} (Covariance: {0.005})')



print(f'Test Batch 0 Sub-batch 2 - (Mean Model 3 Covariance: {0.005})')

df_test.loc[df_test.query('batch == 0 and (520 < time <= 530)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 0 and (520 < time <= 530)').index, 'signal_processed'].values, 0.005)

print(f'Test Batch 1 Sub-batch 1 - (Mean Model 3 Covariance: {0.005})')

df_test.loc[df_test.query('batch == 1 and (560 < time <= 570)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 1 and (560 < time <= 570)').index, 'signal_processed'].values, 0.005)



# filter model 4

print(f'\n---------- Model 4 ----------\n')



batch4_corr = np.corrcoef(df_train[df_train['batch'] == 4]['signal_processed'], df_train[df_train['batch'] == 4]['open_channels'])[0][1]

df_train.loc[df_train['batch'] == 4, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 4]['signal_processed'].values, 0.005)

filtered_batch4_corr = np.corrcoef(df_train[df_train['batch'] == 4]['signal_processed_kalman'], df_train[df_train['batch'] == 4]['open_channels'])[0][1]

print(f'Training Batch 4 - Correlation between Signal and Open Channels increased from {batch4_corr:.6} to {filtered_batch4_corr:.6} (Covariance: {0.005})')



batch9_corr = np.corrcoef(df_train[df_train['batch'] == 9]['signal_processed'], df_train[df_train['batch'] == 9]['open_channels'])[0][1]

df_train.loc[df_train['batch'] == 9, 'signal_processed_kalman'] = kalman(df_train[df_train['batch'] == 9]['signal_processed'].values, 0.005)

filtered_batch9_corr = np.corrcoef(df_train[df_train['batch'] == 9]['signal_processed_kalman'], df_train[df_train['batch'] == 9]['open_channels'])[0][1]

print(f'Training Batch 9 - Correlation between Signal and Open Channels increased from {batch9_corr:.6} to {filtered_batch9_corr:.6} (Covariance: {0.005})')



print(f'Test Batch 1 Sub-batch 0 - (Mean Model 4 Covariance: {0.005})')

df_test.loc[df_test.query('batch == 1 and (550 < time <= 560)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 1 and (550 < time <= 560)').index, 'signal_processed'].values, 0.005)

print(f'Test Batch 1 Sub-batch 2 - (Mean Model 4 Covariance: {0.005})')

df_test.loc[df_test.query('batch == 1 and (570 < time <= 580)').index, 'signal_processed_kalman'] = kalman(df_test.loc[df_test.query('batch == 1 and (570 < time <= 580)').index, 'signal_processed'].values, 0.005)

fig, axes = plt.subplots(nrows=2, figsize=(20, 14), dpi=100)



df_train.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[0], alpha=0.4)

df_train.set_index('time')['signal_processed_kalman'].plot(label='Signal Kalman Filtered', ax=axes[0], alpha=0.8)

for batch in np.arange(0, 550, 50):

    axes[0].axvline(batch, color='r', linestyle='--', lw=2)

    

df_test.set_index('time')['signal_processed'].plot(label='Signal', ax=axes[1], alpha=0.4)

df_test.set_index('time')['signal_processed_kalman'].plot(label='Signal Kalman Filtered', ax=axes[1], alpha=0.8)



for batch in np.arange(500, 600, 10):

    axes[1].axvline(batch, color='r', linestyle='--', lw=2)

for batch in np.arange(600, 700, 50):

    axes[1].axvline(batch, color='r', linestyle='--', lw=2)

    

axes[1].axvline(560, color='y', linestyle='dotted', lw=8)



for i in range(2):    

    for batch in np.arange(0, 550, 50):

        axes[i].axvline(batch, color='r', linestyle='--', lw=2)

        

    axes[i].set_xlabel('Time', size=15)

    axes[i].tick_params(axis='x', labelsize=12)

    axes[i].tick_params(axis='y', labelsize=12)

    axes[i].legend()

    

axes[0].set_title('Training Set Batches Raw/Filtered', size=18, pad=18)

axes[1].set_title('Public/Private Test Set Batches and Sub-batches Raw/Filtered', size=18, pad=18)



plt.show()
df_train.to_pickle('train.pkl')

df_test.to_pickle('test.pkl')



print('Training Set Shape = {}'.format(df_train.shape))

print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))

print('Test Set Shape = {}'.format(df_test.shape))

print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))