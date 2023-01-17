import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score
train_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
train_df.head()
train_df.info(null_counts=True)
test_df.head()
test_df.info(null_counts=True)
train_df['open_channels'].value_counts().plot(kind='bar')

plt.title('Open channels distribution')

plt.show()
plt.hist(train_df.signal, bins=20)

plt.hist(test_df.signal, bins=20)

plt.title('Signal Distribution for Test and Train')

plt.legend(labels=['Train', 'Test'])

plt.show()

print('Train mean {}, median {}, standard deviation {}'.format(np.mean(train_df.signal), np.median(train_df.signal), np.std(train_df.signal)))

print('Test mean {}, median {}, standard deviation {}'.format(np.mean(test_df.signal), np.median(test_df.signal), np.std(test_df.signal)))

print('\nTrain:', stats.normaltest(train_df.signal))

print('Test:', stats.normaltest(train_df.signal))
stats.ttest_ind(train_df.signal, test_df.signal)
plt.hist(train_df.time)

plt.hist(test_df.time)

plt.legend(labels=['Train', 'Test'])

plt.title('Time Distribution (Just Checking)')

plt.show()
train_df['batch'] = 0

for i in range(0, 10):

    train_df.iloc[i * 500000: 500000 * (i + 1), 3] = i
plt.figure(figsize=(20,10))

plt.plot(train_df.signal[train_df.time < 3])

plt.plot(train_df.open_channels[train_df.time < 3])

plt.legend(labels=['Signal', 'Open Channels'], fontsize=16)

plt.title('Signal and Open Channels', fontsize=20)

plt.show()
fig, axes = plt.subplots(2, 5, figsize=(16, 8))

num_batches = len(train_df.batch.unique())

fig.suptitle('Signal and Open Channels by Batch. Blue == signal, Orange == open_channels', fontsize=16)

axis_on = True

for i in range(num_batches):

    axes[i // (num_batches // 2), i % (num_batches // 2)].plot(train_df.signal[train_df.batch == i])

    axes[i // (num_batches // 2), i % (num_batches // 2)].plot(train_df.open_channels[train_df.batch == i])

    axes[i // (num_batches // 2), i % (num_batches // 2)].set_yticks(range(-4, 13))

    if axis_on == False:

        axes[i // (num_batches // 2), i % (num_batches // 2)].set_xticks([])

        axes[i // (num_batches // 2), i % (num_batches // 2)].set_yticks([])

    axis_on = False
test_df['batch'] = 0

for i in range(0, 4):

    test_df.iloc[i * 500000: 500000 * (i + 1), 2] = i



fig, axes = plt.subplots(2, 2, figsize=(10, 10))

num_batches = len(test_df.batch.unique())

fig.suptitle('Test Distributions', fontsize=16)

axis_on = True

for i in range(num_batches):

    axes[i // (num_batches // 2), i % (num_batches // 2)].plot(test_df.signal[test_df.batch == i])

    axes[i // (num_batches // 2), i % (num_batches // 2)].set_yticks(range(-4, 13))

    if axis_on == False:

        axes[i // (num_batches // 2), i % (num_batches // 2)].set_xticks([])

        axes[i // (num_batches // 2), i % (num_batches // 2)].set_yticks([])

    axis_on = False
fig, axes = plt.subplots(4, 3, figsize=(15, 20))

fig.suptitle('Signal Distributions at Number of Open Channels', fontsize=16)

for i in range(11):

    n, bins, patches = axes[i // 3, i % 3].hist(train_df.signal[train_df.open_channels == i], bins=40)

    ind = list(n).index(max(n))

    mean = round(np.mean(train_df.signal[train_df.open_channels == i]), 2)

    binned_mode = (bins[ind] + bins[ind + 1])/2

    axes[i // 3, i % 3].set_title('Channels {}, BinMode {}, Mean {}'.format(i, round(binned_mode, 2), mean))

    axes[i // 3, i % 3].set_xticks([-5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5])

    axes[i // 3, i % 3].axvline(binned_mode , color='orange')

    axes[i // 3, i % 3].axvline(mean , color='green')

plt.show()
start = 0.72

end = 0.727

plt.figure(figsize=(20,10))

plt.plot(train_df.signal[(train_df.time > start) & (train_df.time < end)])

plt.plot(train_df.open_channels[(train_df.time > start) & (train_df.time < end)])

plt.legend(['Signal', 'Open Channels'], fontsize=16)

plt.show()
start = 200.07

end = 200.08

plt.figure(figsize=(20,10))

plt.plot(train_df.signal[(train_df.time > start) & (train_df.time < end)])

plt.plot(train_df.open_channels[(train_df.time > start) & (train_df.time < end)])

plt.legend(['Signal', 'Open Channels'], fontsize=16)

plt.show()
start = 310.07

end = 310.08

plt.figure(figsize=(20,10))

plt.plot(train_df.signal[(train_df.time > start) & (train_df.time < end)])

plt.plot(train_df.open_channels[(train_df.time > start) & (train_df.time < end)])

plt.legend(['Signal', 'Open Channels'], fontsize=16)

plt.show()
rfc = RandomForestClassifier(n_estimators=100, max_depth=5)

X = train_df.drop('open_channels', 1)

Y = train_df.open_channels

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

rfc.fit(X_train, y_train)

preds = rfc.predict(X_test)

print('Results with just signal and time:', cohen_kappa_score(preds, y_test, weights='quadratic'))
previous_signal = []

for batch in train_df.batch.unique():

    previous_signal += [train_df[train_df.batch == batch].signal.iloc[0]]

    previous_signal += list(train_df[train_df.batch == batch].signal.iloc[:-1])

train_df['previous'] = previous_signal

train_df['previous'] = train_df.previous - train_df.signal



second_prev = []

for batch in train_df.batch.unique():

    second_prev += list(train_df[train_df.batch == batch].signal.iloc[:2])

    second_prev += list(train_df[train_df.batch == batch].signal.iloc[:-2])

train_df['second'] = second_prev

train_df['second'] = train_df.second - train_df.signal



third_prev = []

for batch in train_df.batch.unique():

    third_prev += list(train_df[train_df.batch == batch].signal.iloc[:3])

    third_prev += list(train_df[train_df.batch == batch].signal.iloc[:-3])

train_df['third'] = third_prev

train_df['third'] = train_df.third - train_df.signal



chunk_size = 20

batch_size = len(train_df[train_df.batch == 0])

mean_chunks = []

if batch_size // chunk_size == batch_size / chunk_size:

    for i in range(len(train_df) // chunk_size):

        mean_chunks += [np.mean(train_df.signal.iloc[chunk_size * i : chunk_size * (i + 1)])] * chunk_size

else:

    print('Error! Not an even split!')

train_df['MicroMean'] = mean_chunks



chunk_size = 500

batch_size = len(train_df[train_df.batch == 0])

mean_chunks = []

if batch_size // chunk_size == batch_size / chunk_size:

    for i in range(len(train_df) // chunk_size):

        mean_chunks += [np.mean(train_df.signal.iloc[chunk_size * i : chunk_size * (i + 1)])] * chunk_size

else:

    print('Error! Not an even split!')

train_df['LocalMean'] = mean_chunks



chunk_size = 5000

batch_size = len(train_df[train_df.batch == 0])

mean_chunks = []

if batch_size // chunk_size == batch_size / chunk_size:

    for i in range(len(train_df) // chunk_size):

        mean_chunks += [np.mean(train_df.signal.iloc[chunk_size * i : chunk_size * (i + 1)])] * chunk_size

else:

    print('Error! Not an even split!')

train_df['MacroMean'] = mean_chunks
fig, axes = plt.subplots(3, 2, figsize=(10, 15))

axes[0, 0].hist(train_df.previous, bins=30)

axes[0, 0].set_title('Previous')

axes[0, 1].hist(train_df.second, bins=30)

axes[0, 1].set_title('Second')

axes[1, 0].hist(train_df.third, bins=30)

axes[1, 0].set_title('Third')

axes[1, 1].hist(train_df.previous, bins=30)

axes[1, 1].set_title('MicroMean')

axes[2, 0].hist(train_df.previous, bins=30)

axes[2, 0].set_title('LocalMean')

axes[2, 1].hist(train_df.previous, bins=30)

axes[2, 1].set_title('MacroMean')

plt.show()
def plot_features(col_names, start, stop, title, lw):

    plt.figure(figsize=(20,10))

    for col in range(len(col_names)):

        plt.plot(train_df[col_names[col]].iloc[start:stop], lw=lw[col])

    plt.legend(col_names, fontsize=16)

    plt.title(title, fontsize=20)

    plt.show()
plot_features(['signal', 'previous', 'open_channels'], 0, 500000, 'Previous in Batch 0', [1,1,1])
plot_features(['signal', 'previous', 'open_channels'], 7215, 7250, '\"Previous\" Close Up on a Bump', [3,3,3])
plot_features(['signal', 'second', 'open_channels'], 7215, 7250, '\"Second\" Close Up on a Bump', [3,3,3])
plot_features(['signal', 'third', 'open_channels'], 7215, 7250, '\"Third\" Close Up on a Bump', [3,3,3])
plot_features(['signal', 'MicroMean'], 3000000, 3010000, 'Signal and MicroMean in Batch 6', [1, 2])
plot_features(['signal', 'LocalMean'], 3000000, 3010000, 'Signal and LocalMean in Batch 6', [1, 5])
plot_features(['signal', 'MacroMean'], 3000000, 3010000, 'Signal and MacroMean in Batch 6', [1, 5])
plot_features(['open_channels', 'MicroMean'], 472000, 478000, 'MicroMean in Batch 0', [1, 2])
plot_features(['open_channels', 'LocalMean'], 500000, 700000, 'LocalMean in Batch 1', [1, 2])
plot_features(['open_channels', 'MacroMean'], 3000000, 3500000, 'MacroMean in Batch 1', [1, 2])
results = []

for batch in range(10):

    batch_results = []

    rfc = RandomForestClassifier(n_estimators=100, max_depth=4)

    X = train_df.drop(['open_channels', 'time', 'batch'], 1)[train_df.batch == batch]

    Y = train_df['open_channels'][train_df.batch == batch]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

    rfc.fit(X_train, y_train)

    

    for feature in zip(X.columns, rfc.feature_importances_):

        batch_results.append(feature[1])

    

    preds = rfc.predict(X_test)

    batch_results.append(cohen_kappa_score(preds, y_test, weights='quadratic'))

    results.append(batch_results)

    

results_df = pd.DataFrame()

results_df['Signal'] = [item[0] for item in results]

results_df['Previous'] = [item[1] for item in results]

results_df['Second'] = [item[2] for item in results]

results_df['Third'] = [item[3] for item in results]

results_df['MicroMean'] = [item[4] for item in results]

results_df['LocalMean'] = [item[5] for item in results]

results_df['MacroMean'] = [item[6] for item in results]

results_df['Kappa'] = [item[7] for item in results]
results_df
rfc = RandomForestClassifier(n_estimators=100, max_depth=5)

X = train_df[['signal', 'previous', 'second', 'third', 'MicroMean', 'LocalMean', 'MacroMean']]

Y = train_df['open_channels']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

rfc.fit(X_train, y_train)

preds = rfc.predict(X_test)

print('Results with original features:', cohen_kappa_score(preds, y_test, weights='quadratic'))
avg = []

win_size = 100

cs = 500000 #chunk size

for i in range(len(train_df)):

    if (i % cs) - win_size <= 0:

        avg.append(np.mean(train_df.signal.iloc[(i//cs) * cs : ((i//cs) * cs) + win_size]))

    else:

        avg.append(np.mean(train_df.signal.iloc[i - win_size : i]))

train_df['PrevAvgLittle'] = avg



avg = []

for i in range(len(train_df)):

    if (i % cs) > cs - win_size:

        avg.append(np.mean(train_df.signal.iloc[((i//cs + 1) * cs) - win_size : (i//cs + 1) * cs]))

    else:

        avg.append(np.mean(train_df.signal.iloc[i : i + win_size]))

train_df['FutAvgLittle'] = avg

                                

train_df['SlopeLittle'] = train_df['FutAvgLittle'] - train_df['PrevAvgLittle']



train_df['PrevAvgLittle'].fillna(method='bfill', inplace=True)

train_df['FutAvgLittle'].fillna(method='ffill', inplace=True)

train_df['SlopeLittle'].fillna(method='bfill', inplace=True)

train_df['SlopeLittle'].fillna(method='ffill', inplace=True)
avg = []

win_size = 1000

cs = 500000 #chunk size

for i in range(len(train_df)):

    if (i % cs) - win_size <= 0:

        avg.append(np.mean(train_df.signal.iloc[(i//cs) * cs : ((i//cs) * cs) + win_size]))

    else:

        avg.append(np.mean(train_df.signal.iloc[i - win_size : i]))

train_df['PrevAvgMedium'] = avg



avg = []

for i in range(len(train_df)):

    if (i % cs) > cs - win_size:

        avg.append(np.mean(train_df.signal.iloc[((i//cs + 1) * cs) - win_size : (i//cs + 1) * cs]))

    else:

        avg.append(np.mean(train_df.signal.iloc[i : i + win_size]))

train_df['FutAvgMedium'] = avg

                                

train_df['SlopeMedium'] = train_df['FutAvgMedium'] - train_df['PrevAvgMedium']



train_df['PrevAvgMedium'].fillna(method='bfill', inplace=True)

train_df['FutAvgMedium'].fillna(method='ffill', inplace=True)

train_df['SlopeMedium'].fillna(method='bfill', inplace=True)

train_df['SlopeMedium'].fillna(method='ffill', inplace=True)
avg = []

win_size = 5000

cs = 500000 #chunk size

for i in range(len(train_df)):

    if (i % cs) - win_size <= 0:

        avg.append(np.mean(train_df.signal.iloc[(i//cs) * cs : ((i//cs) * cs) + win_size]))

    else:

        avg.append(np.mean(train_df.signal.iloc[i - win_size : i]))

train_df['PrevAvgBig'] = avg



avg = []

for i in range(len(train_df)):

    if (i % cs) > cs - win_size:

        avg.append(np.mean(train_df.signal.iloc[((i//cs + 1) * cs) - win_size : (i//cs + 1) * cs]))

    else:

        avg.append(np.mean(train_df.signal.iloc[i : i + win_size]))

train_df['FutAvgBig'] = avg

                                

train_df['SlopeBig'] = train_df['FutAvgBig'] - train_df['PrevAvgBig']



train_df['PrevAvgBig'].fillna(method='bfill', inplace=True)

train_df['FutAvgBig'].fillna(method='ffill', inplace=True)

train_df['SlopeBig'].fillna(method='bfill', inplace=True)

train_df['SlopeBig'].fillna(method='ffill', inplace=True)
avg = []

win_size = 15000

cs = 500000 #chunk size

for i in range(len(train_df)):

    if (i % cs) - win_size <= 0:

        avg.append(np.mean(train_df.signal.iloc[(i//cs) * cs : ((i//cs) * cs) + win_size]))

    else:

        avg.append(np.mean(train_df.signal.iloc[i - win_size : i]))

train_df['PrevAvgRealBig'] = avg



avg = []

for i in range(len(train_df)):

    if (i % cs) > cs - win_size:

        avg.append(np.mean(train_df.signal.iloc[((i//cs + 1) * cs) - win_size : (i//cs + 1) * cs]))

    else:

        avg.append(np.mean(train_df.signal.iloc[i : i + win_size]))

train_df['FutAvgRealBig'] = avg

                                

train_df['SlopeRealBig'] = train_df['FutAvgRealBig'] - train_df['PrevAvgRealBig']



train_df['PrevAvgRealBig'].fillna(method='bfill', inplace=True)

train_df['FutAvgRealBig'].fillna(method='ffill', inplace=True)

train_df['SlopeRealBig'].fillna(method='bfill', inplace=True)

train_df['SlopeRealBig'].fillna(method='ffill', inplace=True)
plot_features(['signal', 'PrevAvgLittle', 'FutAvgLittle', 'SlopeLittle'],

                     0, 500000, 'Is LocalSlope Showing What We Want?', [1, 5, 5, 1])
plot_features(['signal', 'PrevAvgLittle', 'FutAvgLittle', 'SlopeLittle', 'open_channels'],

                     470000, 500000, 'Zoomed Once', [1, 3, 3, 1, 1])
plot_features(['signal', 'PrevAvgLittle', 'FutAvgLittle', 'SlopeLittle', 'open_channels'],

                     496000, 499000, 'Zoomed Twice', [1, 3, 3, 3, 1])
plot_features(['signal', 'PrevAvgLittle', 'FutAvgLittle', 'SlopeLittle', 'open_channels'],

                     498000, 502000, 'Different spot', [1, 3, 3, 3, 1])
plot_features(['signal', 'PrevAvgMedium', 'FutAvgMedium', 'SlopeMedium'],

                     500000, 1000000, 'Medium Slope', [1, 3, 3, 3])
plot_features(['signal', 'PrevAvgMedium', 'FutAvgMedium', 'SlopeMedium'],

                     495000, 505000, 'SlopeMedium', [1, 3, 3, 3])
plot_features(['signal', 'PrevAvgBig', 'FutAvgBig', 'SlopeBig'],

                     500000, 550000, 'Macro Slope', [1, 3, 3, 3])
plot_features(['signal', 'PrevAvgBig', 'FutAvgBig', 'SlopeBig'],

                     3000000, 3500000, 'Macro Slope', [1, 3, 3, 3])
plot_features(['signal', 'PrevAvgRealBig', 'FutAvgRealBig', 'SlopeRealBig'],

                     3000000, 3500000, 'Is RealBig Showing What We Want?', [1, 5, 5, 1])
plt.figure(figsize=(20,10))

plt.plot(train_df.SlopeRealBig.iloc[3000000:3500000], lw=3)

plt.plot(train_df.SlopeBig.iloc[3000000:3500000], alpha=.8)

plt.plot(train_df.SlopeMedium.iloc[3000000:3500000], alpha=.6)

plt.axhline(0, color='red')

plt.title('Comparison of Slope Features in Batch 6')

plt.legend(labels=['SlopeRealBig', 'SlopeBig', 'SlopeMedium'])

plt.show()
results = []

for batch in range(10):

    batch_results = []

    rfc = RandomForestClassifier(n_estimators=150, max_depth=5)

    X = train_df.drop(['open_channels', 'time', 'batch'], 1)[train_df.batch == batch]

    Y = train_df['open_channels'][train_df.batch == batch]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

    rfc.fit(X_train, y_train)

    

    for feature in zip(X.columns, rfc.feature_importances_):

        batch_results.append(feature[1])

    

    preds = rfc.predict(X_test)

    batch_results.append(cohen_kappa_score(preds, y_test, weights='quadratic'))

    results.append(batch_results)

    

results_df = pd.DataFrame()

for column in range(len(X.columns)):

    results_df[X.columns[column]] = [item[column] for item in results]
results_df
train_df.drop(['SlopeLittle', 'SlopeMedium', 'SlopeBig', 'SlopeRealBig'], 1, inplace=True)
rfc = RandomForestClassifier(n_estimators=100, max_depth=5)

X = train_df.drop(['open_channels', 'time', 'batch'], 1)

Y = train_df['open_channels']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

rfc.fit(X_train, y_train)

preds = rfc.predict(X_test)

print('Results with all features:', cohen_kappa_score(preds, y_test, weights='quadratic'))
for i in range(4, 9):

    prev = []

    for batch in train_df.batch.unique():

        prev += list(train_df[train_df.batch == batch].signal.iloc[:i])

        prev += list(train_df[train_df.batch == batch].signal.iloc[:-i])

    train_df['{}_prev'.format(i)] = prev

    train_df['{}_prev'.format(i)] -= train_df.signal
rfc = RandomForestClassifier(n_estimators=100, max_depth=5)

X = train_df.drop(['open_channels', 'time', 'batch'], 1)

Y = train_df['open_channels']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

rfc.fit(X_train, y_train)

preds = rfc.predict(X_test)

print('Results with all features:', cohen_kappa_score(preds, y_test, weights='quadratic'))