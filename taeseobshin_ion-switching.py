# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")

print(train_data.head())

print(train_data.describe())
test_data = pd.read_csv("/kaggle/input/liverpool-ion-switching/test.csv")

print(test_data.head())

print(test_data.describe())
sample_submission_data = pd.read_csv("/kaggle/input/liverpool-ion-switching/sample_submission.csv")

print(sample_submission_data.head())

print(sample_submission_data.describe())
plt.figure(figsize=(20, 5))

plt.plot(train_data['time'], train_data['signal'])

plt.show()
plt.figure(figsize=(20,5)); res = 10 #간격을 10 만큼씩 띄어서 표시

x = range(0,train_data.shape[0],res)

y = train_data.signal[0::res]

plt.plot(x,y,'b',alpha=0.7)

for i in range(11): 

    plt.plot([i*500000,i*500000],[-5,12.5],'r')

for j in range(10): 

    plt.text(j*500000+200000,10,str(j+1),size=16)

plt.xlabel('Row',size=16); plt.ylabel('Signal & Ion channels',size=16); 

plt.title('Training Data Signal - 10 batches',size=20)



#plt.figure(figsize=(20,5))

y2 = train_data.open_channels[0::res]

plt.plot(x,y2,'r',alpha=0.3) #Ion 채널 수 표시



plt.show()
plt.figure(figsize=(20,5)); res = 1000 #간격을 1000 만큼씩 띄어서 표시

x = range(0,train_data.shape[0],res)

y = train_data.signal[0::res]

plt.plot(x,y,'b',alpha=0.7)

for i in range(11): 

    plt.plot([i*500000,i*500000],[-5,12.5],'r')

for j in range(10): 

    plt.text(j*500000+200000,10,str(j+1),size=16)

plt.xlabel('Row',size=16); plt.ylabel('Signal & Ion channels',size=16); 

plt.title('Training Data Signal - 10 batches',size=20)



#plt.figure(figsize=(20,5))

y2 = train_data.open_channels[0::res]

plt.plot(x,y2,'r',alpha=0.3) #Ion 채널 수 표시



plt.show()
plt.figure(figsize=(20,5)); res = 10 #간격을 1000 만큼씩 띄어서 표시

x = range(0,test_data.shape[0],res)

y = test_data.signal[0::res]

plt.plot(x,y,'b',alpha=0.7)

for i in range(21): 

    plt.plot([i*100000,i*100000],[-5,12.5],'r')

for j in range(20): 

    plt.text(j*100000+30000,10,str(j+1),size=16)

plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 

plt.title('Test Data Signal - 20 batches',size=20)



plt.show()
#각 배치를 10개 데이터 리스트로 구분

batch = []

gap = 500000

batch.append([]) #dummy

for i in range(10):

    start = i*gap

    batch.append(train_data[start:start+gap])
#plot작성용 helper 

def plot_batch(batch_no=0, start=0, end=500000):

    x = range(end-start)

    y1 = batch[batch_no].signal[start:end]

    y2 = batch[batch_no].open_channels[start:end]

    fig, ax1 = plt.subplots(figsize=(20,5))

    ax1.set_xlabel('time (ms)')

    ax1.set_ylabel('signal')

    ax1.plot(x, y1, 'blue')

    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()

    ax2.set_ylabel('open channels')

    ax2.plot(x, y2, 'red', alpha=0.5)

    ax2.tick_params(axis='y')

    fig.tight_layout()

    plt.legend()

    plt.show()
plot_batch(6) #6번 배치를 출력
plot_batch(6,0,100) # 6번 배치에서 0~100
plot_batch(6,40,60) #6번 배치에서 40~60
plot_batch(5,0,2000) #5번 배치에서 0~2000
plot_batch(7)
plot_batch(7,0,500)
plot_batch(7,250000,250100)
def average_smoothing(signal, kernel_size=10):

    #sample = []

    #start = 0

    #end = kernel_size

    #while start <= len(signal):

    #    start += 1

    #    end += 1

    #    #sample.extend(np.ones(kernel_size)*np.mean(signal[start:end]))

    #    if end >= len(signal):

    #        end = len(signal)

    #    sample.append(np.mean(signal[start:end]))

    #return np.array(sample)

    return signal.rolling(window=kernel_size, min_periods=1).mean()
def average_smoothing_center(signal, kernel_size=10):

    #sample = []

    #center = 0

    #while center < len(signal):

    #    start = center - int(kernel_size / 2)

    #    end = center + int(kernel_size / 2)

    #    if start < 0:

    #        start = 0

    #    if end > len(signal):

    #        end = len(signal)

    #    sample.append(np.mean(signal[start:end]))

    #    center += 1

    #return np.array(sample)

    return signal.rolling(window=kernel_size, min_periods=1, center=True).mean()
x = train_data.loc[:100]["time"]

y1 = train_data.loc[:100]["signal"]

y_a1 = average_smoothing(train_data.loc[:100]["signal"])

y_b1 = average_smoothing_center(train_data.loc[:100]["signal"])

y2 = train_data.loc[100:200]["signal"]

y_a2 = average_smoothing(train_data.loc[100:200]["signal"])

y_b2 = average_smoothing_center(train_data.loc[100:200]["signal"])

y3 = train_data.loc[200:300]["signal"]

y_a3 = average_smoothing(train_data.loc[200:300]["signal"])

y_b3 = average_smoothing_center(train_data.loc[200:300]["signal"])
fig = make_subplots(rows=3, cols=1)



fig.add_trace(

    go.Scatter(x=x, mode='lines+markers', y=y1, marker=dict(color="lightskyblue"), showlegend=False,

               name="Original signal"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_a1, mode='lines', marker=dict(color="navy"), showlegend=False,

               name="Denoised signal"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_b1, mode='lines', marker=dict(color="red"), showlegend=False,

               name="Denoised signal (center)"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=x, mode='lines+markers', y=y2, marker=dict(color="mediumaquamarine"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_a2, mode='lines', marker=dict(color="darkgreen"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_b2, mode='lines', marker=dict(color="red"), showlegend=False,

               name="Denoised signal (center)"),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=x, mode='lines+markers', y=y3, marker=dict(color="thistle"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_a3, mode='lines', marker=dict(color="indigo"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_b3, mode='lines', marker=dict(color="red"), showlegend=False,

               name="Denoised signal (center)"),

    row=3, col=1

)



fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) signals")

fig.show()
x = train_data.loc[:1000]["time"]

y1 = train_data.loc[:1000]["signal"]

y_a1 = average_smoothing(train_data.loc[:1000]["signal"])

y_b1 = average_smoothing_center(train_data.loc[:1000]["signal"])

y2 = train_data.loc[1000:2000]["signal"]

y_a2 = average_smoothing(train_data.loc[1000:2000]["signal"])

y_b2 = average_smoothing_center(train_data.loc[1000:2000]["signal"])

y3 = train_data.loc[2000:3000]["signal"]

y_a3 = average_smoothing(train_data.loc[2000:3000]["signal"])

y_b3 = average_smoothing_center(train_data.loc[2000:3000]["signal"])
fig = make_subplots(rows=3, cols=1)



fig.add_trace(

    go.Scatter(x=x, mode='lines+markers', y=y1, marker=dict(color="lightskyblue"), showlegend=False,

               name="Original signal"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_a1, mode='lines', marker=dict(color="navy"), showlegend=False,

               name="Denoised signal"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_b1, mode='lines', marker=dict(color="red"), showlegend=False,

               name="Denoised signal (center)"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=x, mode='lines+markers', y=y2, marker=dict(color="mediumaquamarine"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_a2, mode='lines', marker=dict(color="darkgreen"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_b2, mode='lines', marker=dict(color="red"), showlegend=False,

               name="Denoised signal (center)"),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=x, mode='lines+markers', y=y3, marker=dict(color="thistle"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_a3, mode='lines', marker=dict(color="indigo"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=x, y=y_b3, mode='lines', marker=dict(color="red"), showlegend=False,

               name="Denoised signal (center)"),

    row=3, col=1

)



fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) signals")

fig.show()
plt.figure(figsize=(20, 5))

sns.boxplot(x="open_channels", y="signal", data=train_data)

plt.show()
train_data.boxplot(column=["signal"], by=["open_channels"], figsize=(20, 5))

plt.show()
train_data["signal_smoothing"] = average_smoothing_center(train_data["signal"])
plt.figure(figsize=(20, 5))

sns.boxplot(x="open_channels", y="signal_smoothing", data=train_data)

plt.show()
train_data.boxplot(column=["signal", "signal_smoothing"], by=["open_channels"], figsize=(20, 5))

plt.show()
def f(x,low,high,mid): return -((-low+high)/625)*(x-mid)**2+high -low



# CLEAN TRAIN BATCH 7

batch = 7; a = 500000*(batch-1); b = 500000*batch

train_data.loc[train_data.index[a:b],'signal_undrifted'] = train_data.signal.values[a:b] - f(train_data.time[a:b].values,-1.817,3.186,325)

# CLEAN TRAIN BATCH 8

batch = 8; a = 500000*(batch-1); b = 500000*batch

train_data.loc[train_data.index[a:b],'signal_undrifted'] = train_data.signal.values[a:b] - f(train_data.time[a:b].values,-0.094,4.936,375)

# CLEAN TRAIN BATCH 9

batch = 9; a = 500000*(batch-1); b = 500000*batch

train_data.loc[train_data.index[a:b],'signal_undrifted'] = train_data.signal.values[a:b] - f(train_data.time[a:b].values,1.715,6.689,425)

# CLEAN TRAIN BATCH 10

batch = 10; a = 500000*(batch-1); b = 500000*batch

train_data.loc[train_data.index[a:b],'signal_undrifted'] = train_data.signal.values[a:b] - f(train_data.time[a:b].values,3.361,8.45,475)
plt.figure(figsize=(20,5))

sns.lineplot(train_data.time[::1000],train_data.signal[::2000],color='r').set_title('Training Batches 7-10 with Parabolic Drift')

#plt.figure(figsize=(20,5))

g = sns.lineplot(train_data.time[::1000],train_data.signal_undrifted[::2000],color='g').set_title('Training Batches 7-10 without Parabolic Drift')

plt.legend(title='Train Data',loc='upper left', labels=['Original Signal', 'UnDrifted Signal'])

plt.show(g)
test_data['signal_undrifted'] = test_data.signal



# REMOVE BATCH 1 DRIFT

start=500

a = 0; b = 100000

test_data.loc[test_data.index[a:b],'signal_undrifted'] = test_data.signal.values[a:b] - 3*(test_data.time.values[a:b]-start)/10.

start=510

a = 100000; b = 200000

test_data.loc[test_data.index[a:b],'signal_undrifted'] = test_data.signal.values[a:b] - 3*(test_data.time.values[a:b]-start)/10.

start=540

a = 400000; b = 500000

test_data.loc[test_data.index[a:b],'signal_undrifted'] = test_data.signal.values[a:b] - 3*(test_data.time.values[a:b]-start)/10.



# REMOVE BATCH 2 DRIFT

start=560

a = 600000; b = 700000

test_data.loc[test_data.index[a:b],'signal_undrifted'] = test_data.signal.values[a:b] - 3*(test_data.time.values[a:b]-start)/10.

start=570

a = 700000; b = 800000

test_data.loc[test_data.index[a:b],'signal_undrifted'] = test_data.signal.values[a:b] - 3*(test_data.time.values[a:b]-start)/10.

start=580

a = 800000; b = 900000

test_data.loc[test_data.index[a:b],'signal_undrifted'] = test_data.signal.values[a:b] - 3*(test_data.time.values[a:b]-start)/10.



# REMOVE BATCH 3 DRIFT

def f(x):

    return -(0.00788)*(x-625)**2+2.345 +2.58

a = 1000000; b = 1500000

test_data.loc[test_data.index[a:b],'signal_undrifted'] = test_data.signal.values[a:b] - f(test_data.time[a:b].values)
plt.figure(figsize=(20,5))

sns.lineplot(test_data.time[::1000],test_data.signal[::1000],color='r').set_title('Test Batches with Parabolic Drift')

#plt.figure(figsize=(20,5))

g = sns.lineplot(test_data.time[::1000],test_data.signal_undrifted[::1000],color='g').set_title('Test Batches without Parabolic Drift')

plt.legend(title='Test Data',loc='upper right', labels=['Original Signal', 'UnDrifted Signal'])

plt.show(g)