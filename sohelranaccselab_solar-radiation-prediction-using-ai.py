import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from keras.layers import Dense, Dropout

from keras.optimizers import SGD, Adam

from keras.models import Sequential

from collections import Counter

from scipy import stats
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
data = pd.read_csv('../input/SolarEnergy/SolarPrediction.csv')
data.head()
X = data.drop(['UNIXTime', 'Radiation'], axis=1)

y = data['Radiation']
X['TSR_Minute'] = pd.to_datetime(X['TimeSunRise']).dt.minute

X['TSS_Minute'] = pd.to_datetime(X['TimeSunSet']).dt.minute

X['TSS_Hour'] = np.where(pd.to_datetime(X['TimeSunSet']).dt.hour==18, 1, 0)
time = np.array([])



for i in X['Data']:

    splt = i.split()

    time = np.append(time, splt[0])

    

X['Month'] = pd.to_datetime(X['Data']).dt.month

X['Day'] = pd.to_datetime(X['Data']).dt.day
X['Hour'] = pd.to_datetime(X['Time']).dt.hour

X['Minute'] = pd.to_datetime(X['Time']).dt.minute

X['Second'] = pd.to_datetime(X['Time']).dt.second
count = Counter(time)

plt.figure(figsize=(20, 7))

plt.bar(count.keys(), count.values(), color='purple')

plt.xticks([])

plt.xlabel('Days in the year')

plt.ylabel('Number of days recorded')

plt.title('The amount of days recorded')

plt.show()
norm = {'Temperature' : (X['Temperature']+1).transform(np.log), 

        'Humidity' : stats.boxcox(X['Humidity']+1)[0], 'Speed' : (X['Speed']+1).transform(np.log), 

        'WindDirection(Degrees)' : MinMaxScaler().fit_transform(np.array(X['WindDirection(Degrees)']).reshape(-1, 1)),

        'TSS_Minute' : stats.boxcox(X['TSS_Minute']+1)[0]}



for i in norm:

    b=50

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 5))

    pd.DataFrame(X[i]).hist(ax=ax1, bins=b)

    pd.DataFrame((X[i]+1).transform(np.log)).hist(ax=ax2, bins=b)

    pd.DataFrame(stats.boxcox(X[i]+1)[0]).hist(ax=ax3, bins=b)

    

    pd.DataFrame(StandardScaler().fit_transform(np.array(X[i]).reshape(-1, 1))).hist(ax=ax4, bins=b)

    pd.DataFrame(MinMaxScaler().fit_transform(np.array(X[i]).reshape(-1, 1))).hist(ax=ax5, bins=b)

    

    ax1.set_ylabel('Normal')

    ax2.set_ylabel('Log')

    ax3.set_ylabel('Box Cox')

    ax4.set_ylabel('Standard')

    ax5.set_ylabel('MinMax')

    

    X[i] = norm[i]
X = X.drop(['Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1)
X.head()
X['WindDirection(Degrees)_bin'] = np.digitize(X['WindDirection(Degrees)'], np.arange(0.0, 1.0, 0.02).tolist())

X['TSS_Minute_bin'] = np.digitize(X['TSS_Minute'], np.arange(0.0, 288.0, 12).tolist())

X['Humidity_bin'] = np.digitize(X['Humidity'], np.arange(32, 3192, 128).tolist())
feats = {'WindDirection(Degrees)_bin':'blue', 'TSS_Minute_bin':'red', 'Humidity_bin':'green'}

for i in feats:

    count = Counter(X[i])

    plt.bar(count.keys(), count.values(), color=feats[i])

    plt.title('Distribution')

    plt.ylabel('Occurrence')

    plt.xlabel(i)

    plt.show()
X.head()
sns.heatmap(X.corr())

plt.show()
X.describe()
X.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = Sequential()

    

model.add(Dense(256, activation='relu', input_dim=16))

model.add(Dropout(0.25))

    

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))



model.add(Dense(64, activation='relu'))

model.add(Dropout(0.4))



model.add(Dense(1, activation='linear'))
model.compile(metrics='mse',

              loss='mae',

              optimizer=Adam(learning_rate=0.003))

history = model.fit(X_train,

                    y_train,

                    validation_data=(X_test, y_test),

                    epochs=70,

                    batch_size=64)
fit = history.history

scores = model.evaluate(X_test, y_test)

mae = scores[0]

mse = scores[1]

print('Mean absolute error: ' + str(mae) + '. Mean squared error: ' + str(mse) + '.')



for i in fit:

    plt.plot(fit[i])

    plt.title(i + ' over epochs')

    plt.ylabel(i)

    plt.xlabel('epochs')

    plt.show()