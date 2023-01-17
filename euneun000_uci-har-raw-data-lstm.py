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
#train raw data 로드

body_acc_x_train = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/Inertial Signals/body_acc_x_train.txt', delim_whitespace=True, header=None)

body_acc_x_train = pd.DataFrame(body_acc_x_train)

body_acc_y_train = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/Inertial Signals/body_acc_y_train.txt', delim_whitespace=True, header=None)

body_acc_y_train = pd.DataFrame(body_acc_y_train)

body_acc_z_train = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/Inertial Signals/body_acc_z_train.txt', delim_whitespace=True, header=None)

body_acc_z_train = pd.DataFrame(body_acc_z_train)

body_gyro_x_train = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/Inertial Signals/body_gyro_x_train.txt', delim_whitespace=True, header=None)

body_gyro_x_train = pd.DataFrame(body_gyro_x_train)

body_gyro_y_train = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/Inertial Signals/body_gyro_y_train.txt', delim_whitespace=True, header=None)

body_gyro_y_train = pd.DataFrame(body_gyro_y_train)

body_gyro_z_train = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/Inertial Signals/body_gyro_z_train.txt', delim_whitespace=True, header=None)

body_gyro_z_train = pd.DataFrame(body_gyro_z_train)

total_acc_x_train = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/Inertial Signals/total_acc_x_train.txt', delim_whitespace=True, header=None)

total_acc_x_train = pd.DataFrame(total_acc_x_train)

total_acc_y_train = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/Inertial Signals/total_acc_y_train.txt', delim_whitespace=True, header=None)

total_acc_y_train = pd.DataFrame(total_acc_y_train)

total_acc_z_train = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/Inertial Signals/total_acc_z_train.txt', delim_whitespace=True, header=None)

total_acc_z_train = pd.DataFrame(total_acc_z_train)
#test raw data 로드

body_acc_x_test = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/test/Inertial Signals/body_acc_x_test.txt', delim_whitespace=True, header=None)

body_acc_x_test = pd.DataFrame(body_acc_x_test)

body_acc_y_test = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/test/Inertial Signals/body_acc_y_test.txt', delim_whitespace=True, header=None)

body_acc_y_test = pd.DataFrame(body_acc_y_test)

body_acc_z_test = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/test/Inertial Signals/body_acc_z_test.txt', delim_whitespace=True, header=None)

body_acc_z_test = pd.DataFrame(body_acc_z_test)

body_gyro_x_test = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/test/Inertial Signals/body_gyro_x_test.txt', delim_whitespace=True, header=None)

body_gyro_x_test = pd.DataFrame(body_gyro_x_test)

body_gyro_y_test = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/test/Inertial Signals/body_gyro_y_test.txt', delim_whitespace=True, header=None)

body_gyro_y_test = pd.DataFrame(body_gyro_y_test)

body_gyro_z_test = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/test/Inertial Signals/body_gyro_z_test.txt', delim_whitespace=True, header=None)

body_gyro_z_test = pd.DataFrame(body_gyro_z_test)

total_acc_x_test = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/test/Inertial Signals/total_acc_x_test.txt', delim_whitespace=True, header=None)

total_acc_x_test = pd.DataFrame(total_acc_x_test)

total_acc_y_test = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/test/Inertial Signals/total_acc_y_test.txt', delim_whitespace=True, header=None)

total_acc_y_test = pd.DataFrame(total_acc_y_test)

total_acc_z_test = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/test/Inertial Signals/total_acc_z_test.txt', delim_whitespace=True, header=None)

total_acc_z_test = pd.DataFrame(total_acc_z_test)
body_gyro_x_train.sample() #확인
total_acc_z_test.sample() #확인
from sklearn.decomposition import PCA #PCA 적용



pca = PCA(n_components=0.9)

body_acc_x_train_reduce = pd.DataFrame(pca.fit_transform(body_acc_x_train))

body_acc_x_test_reduce = pd.DataFrame(pca.transform(body_acc_x_test))



body_acc_y_train_reduce = pd.DataFrame(pca.fit_transform(body_acc_y_train))

body_acc_y_test_reduce = pd.DataFrame(pca.transform(body_acc_y_test))



body_acc_z_train_reduce = pd.DataFrame(pca.fit_transform(body_acc_z_train))

body_acc_z_test_reduce = pd.DataFrame(pca.transform(body_acc_z_test))



body_gyro_x_train_reduce = pd.DataFrame(pca.fit_transform(body_gyro_x_train))

body_gyro_x_test_reduce = pd.DataFrame(pca.transform(body_gyro_x_test))



body_gyro_y_train_reduce = pd.DataFrame(pca.fit_transform(body_gyro_y_train))

body_gyro_y_test_reduce = pd.DataFrame(pca.transform(body_gyro_y_test))



body_gyro_z_train_reduce = pd.DataFrame(pca.fit_transform(body_gyro_z_train))

body_gyro_z_test_reduce = pd.DataFrame(pca.transform(body_gyro_z_test))



total_acc_x_train_reduce = pd.DataFrame(pca.fit_transform(total_acc_x_train))

total_acc_x_test_reduce = pd.DataFrame(pca.transform(total_acc_x_test))



total_acc_y_train_reduce = pd.DataFrame(pca.fit_transform(total_acc_y_train))

total_acc_y_test_reduce = pd.DataFrame(pca.transform(total_acc_y_test))



total_acc_z_train_reduce = pd.DataFrame(pca.fit_transform(total_acc_z_train))

total_acc_z_test_reduce = pd.DataFrame(pca.transform(total_acc_z_test))



body_acc_x_train_reduce.shape
pca_new = pd.DataFrame(body_acc_x_train_reduce.mean(axis=1))

pca_new.columns = ['tBodyAcc-mean()-x']

pca_new['tBodyAcc-mean()-y'] = body_acc_y_train_reduce.mean(axis=1)

pca_new['tBodyAcc-mean()-z'] = body_acc_z_train_reduce.mean(axis=1)

pca_new['tBodyGyro-mean()-x'] = body_gyro_x_train_reduce.mean(axis=1)

pca_new['tBodyGyro-mean()-y'] = body_gyro_y_train_reduce.mean(axis=1)

pca_new['tBodyGyro-mean()-z'] = body_gyro_z_train_reduce.mean(axis=1)

pca_new['tTotalAcc-mean()-x'] = total_acc_x_train_reduce.mean(axis=1)

pca_new['tTotalAcc-mean()-y'] = total_acc_y_train_reduce.mean(axis=1)

pca_new['tTotalAcc-mean()-z'] = total_acc_z_train_reduce.mean(axis=1)



pca_new['tBodyAcc-std()-x'] = body_acc_x_train_reduce.std(axis=1)

pca_new['tBodyAcc-std()-y'] = body_acc_y_train_reduce.std(axis=1)

pca_new['tBodyAcc-std()-z'] = body_acc_z_train_reduce.std(axis=1)

pca_new['tBodyGyro-std()-x'] = body_gyro_x_train_reduce.std(axis=1)

pca_new['tBodyGyro-std()-y'] = body_gyro_y_train_reduce.std(axis=1)

pca_new['tBodyGyro-std()-z'] = body_gyro_z_train_reduce.std(axis=1)

pca_new['tTotalAcc-std()-x'] = total_acc_x_train_reduce.std(axis=1)

#pca_new['tTotalAcc-std()-y'] = total_acc_y_train_reduce.std(axis=1) -> NaN으로 나와서 LSTM이 돌아가지 않기에 생략

#pca_new['tTotalAcc-std()-z'] = total_acc_z_train_reduce.std(axis=1) -> 위와 동일



pca_new['tBodyAcc-mad()-x'] = body_acc_x_train_reduce.mad(axis=1)

pca_new['tBodyAcc-mad()-y'] = body_acc_y_train_reduce.mad(axis=1)

pca_new['tBodyAcc-mad()-z'] = body_acc_z_train_reduce.mad(axis=1)

pca_new['tBodyGyro-mad()-x'] = body_gyro_x_train_reduce.mad(axis=1)

pca_new['tBodyGyro-mad()-y'] = body_gyro_y_train_reduce.mad(axis=1)

pca_new['tBodyGyro-mad()-z'] = body_gyro_z_train_reduce.mad(axis=1)

pca_new['tTotalAcc-mad()-x'] = total_acc_x_train_reduce.mad(axis=1)

#pca_new['tTotalAcc-mad()-y'] = total_acc_y_train_reduce.mad(axis=1) -> 모두 0으로 나와서 의미가 없기에 생략

#pca_new['tTotalAcc-mad()-z'] = total_acc_z_train_reduce.mad(axis=1) -> 위와 동일



pca_new['tBodyAcc-max()-x'] = body_acc_x_train_reduce.max(axis=1)

pca_new['tBodyAcc-max()-y'] = body_acc_y_train_reduce.max(axis=1)

pca_new['tBodyAcc-max()-z'] = body_acc_z_train_reduce.max(axis=1)

pca_new['tBodyGyro-max()-x'] = body_gyro_x_train_reduce.max(axis=1)

pca_new['tBodyGyro-max()-y'] = body_gyro_y_train_reduce.max(axis=1)

pca_new['tBodyGyro-max()-z'] = body_gyro_z_train_reduce.max(axis=1)

pca_new['tTotalAcc-max()-x'] = total_acc_x_train_reduce.max(axis=1)

pca_new['tTotalAcc-max()-y'] = total_acc_y_train_reduce.max(axis=1)

pca_new['tTotalAcc-max()-z'] = total_acc_z_train_reduce.max(axis=1)



pca_new['tBodyAcc-min()-x'] = body_acc_x_train_reduce.min(axis=1)

pca_new['tBodyAcc-min()-y'] = body_acc_y_train_reduce.min(axis=1)

pca_new['tBodyAcc-min()-z'] = body_acc_z_train_reduce.min(axis=1)

pca_new['tBodyGyro-min()-x'] = body_gyro_x_train_reduce.min(axis=1)

pca_new['tBodyGyro-min()-y'] = body_gyro_y_train_reduce.min(axis=1)

pca_new['tBodyGyro-min()-z'] = body_gyro_z_train_reduce.min(axis=1)

pca_new['tTotalAcc-min()-x'] = total_acc_x_train_reduce.min(axis=1)

pca_new['tTotalAcc-min()-y'] = total_acc_y_train_reduce.min(axis=1)

pca_new['tTotalAcc-min()-z'] = total_acc_z_train_reduce.min(axis=1)



pca_new['tBodyAcc-median()-x'] = body_acc_x_train_reduce.median(axis=1)

pca_new['tBodyAcc-median()-y'] = body_acc_y_train_reduce.median(axis=1)

pca_new['tBodyAcc-median()-z'] = body_acc_z_train_reduce.median(axis=1)

pca_new['tBodyGyro-median()-x'] = body_gyro_x_train_reduce.median(axis=1)

pca_new['tBodyGyro-median()-y'] = body_gyro_y_train_reduce.median(axis=1)

pca_new['tBodyGyro-median()-z'] = body_gyro_z_train_reduce.median(axis=1)

pca_new['tTotalAcc-median()-x'] = total_acc_x_train_reduce.median(axis=1)

pca_new['tTotalAcc-median()-y'] = total_acc_y_train_reduce.median(axis=1)

pca_new['tTotalAcc-median()-z'] = total_acc_z_train_reduce.median(axis=1)
test_pca_new = pd.DataFrame(body_acc_x_test_reduce.mean(axis=1))

test_pca_new.columns = ['tBodyAcc-mean()-x']

test_pca_new['tBodyAcc-mean()-y'] = body_acc_y_test_reduce.mean(axis=1)

test_pca_new['tBodyAcc-mean()-z'] = body_acc_z_test_reduce.mean(axis=1)

test_pca_new['tBodyGyro-mean()-x'] = body_gyro_x_test_reduce.mean(axis=1)

test_pca_new['tBodyGyro-mean()-y'] = body_gyro_y_test_reduce.mean(axis=1)

test_pca_new['tBodyGyro-mean()-z'] = body_gyro_z_test_reduce.mean(axis=1)

test_pca_new['tTotalAcc-mean()-x'] = total_acc_x_test_reduce.mean(axis=1)

test_pca_new['tTotalAcc-mean()-y'] = total_acc_y_test_reduce.mean(axis=1)

test_pca_new['tTotalAcc-mean()-z'] = total_acc_z_test_reduce.mean(axis=1)



test_pca_new['tBodyAcc-std()-x'] = body_acc_x_test_reduce.std(axis=1)

test_pca_new['tBodyAcc-std()-y'] = body_acc_y_test_reduce.std(axis=1)

test_pca_new['tBodyAcc-std()-z'] = body_acc_z_test_reduce.std(axis=1)

test_pca_new['tBodyGyro-std()-x'] = body_gyro_x_test_reduce.std(axis=1)

test_pca_new['tBodyGyro-std()-y'] = body_gyro_y_test_reduce.std(axis=1)

test_pca_new['tBodyGyro-std()-z'] = body_gyro_z_test_reduce.std(axis=1)

test_pca_new['tTotalAcc-std()-x'] = total_acc_x_test_reduce.std(axis=1)

#test_pca_new['tTotalAcc-std()-y'] = total_acc_y_test_reduce.std(axis=1) -> NaN으로 나와서 LSTM이 돌아가지 않기에 생략

#test_pca_new['tTotalAcc-std()-z'] = total_acc_z_test_reduce.std(axis=1) -> 위와 동일



test_pca_new['tBodyAcc-mad()-x'] = body_acc_x_test_reduce.mad(axis=1)

test_pca_new['tBodyAcc-mad()-y'] = body_acc_y_test_reduce.mad(axis=1)

test_pca_new['tBodyAcc-mad()-z'] = body_acc_z_test_reduce.mad(axis=1)

test_pca_new['tBodyGyro-mad()-x'] = body_gyro_x_test_reduce.mad(axis=1)

test_pca_new['tBodyGyro-mad()-y'] = body_gyro_y_test_reduce.mad(axis=1)

test_pca_new['tBodyGyro-mad()-z'] = body_gyro_z_test_reduce.mad(axis=1)

test_pca_new['tTotalAcc-mad()-x'] = total_acc_x_test_reduce.mad(axis=1)

#test_pca_new['tTotalAcc-mad()-y'] = total_acc_y_test_reduce.mad(axis=1) -> 모두 0으로 나와서 의미가 없기에 생략

#test_pca_new['tTotalAcc-mad()-z'] = total_acc_z_test_reduce.mad(axis=1) -> 위와 동일



test_pca_new['tBodyAcc-max()-x'] = body_acc_x_test_reduce.max(axis=1)

test_pca_new['tBodyAcc-max()-y'] = body_acc_y_test_reduce.max(axis=1)

test_pca_new['tBodyAcc-max()-z'] = body_acc_z_test_reduce.max(axis=1)

test_pca_new['tBodyGyro-max()-x'] = body_gyro_x_test_reduce.max(axis=1)

test_pca_new['tBodyGyro-max()-y'] = body_gyro_y_test_reduce.max(axis=1)

test_pca_new['tBodyGyro-max()-z'] = body_gyro_z_test_reduce.max(axis=1)

test_pca_new['tTotalAcc-max()-x'] = total_acc_x_test_reduce.max(axis=1)

test_pca_new['tTotalAcc-max()-y'] = total_acc_y_test_reduce.max(axis=1)

test_pca_new['tTotalAcc-max()-z'] = total_acc_z_test_reduce.max(axis=1)



test_pca_new['tBodyAcc-min()-x'] = body_acc_x_test_reduce.min(axis=1)

test_pca_new['tBodyAcc-min()-y'] = body_acc_y_test_reduce.min(axis=1)

test_pca_new['tBodyAcc-min()-z'] = body_acc_z_test_reduce.min(axis=1)

test_pca_new['tBodyGyro-min()-x'] = body_gyro_x_test_reduce.min(axis=1)

test_pca_new['tBodyGyro-min()-y'] = body_gyro_y_test_reduce.min(axis=1)

test_pca_new['tBodyGyro-min()-z'] = body_gyro_z_test_reduce.min(axis=1)

test_pca_new['tTotalAcc-min()-x'] = total_acc_x_test_reduce.min(axis=1)

test_pca_new['tTotalAcc-min()-y'] = total_acc_y_test_reduce.min(axis=1)

test_pca_new['tTotalAcc-min()-z'] = total_acc_z_test_reduce.min(axis=1)



test_pca_new['tBodyAcc-median()-x'] = body_acc_x_test_reduce.median(axis=1)

test_pca_new['tBodyAcc-median()-y'] = body_acc_y_test_reduce.median(axis=1)

test_pca_new['tBodyAcc-median()-z'] = body_acc_z_test_reduce.median(axis=1)

test_pca_new['tBodyGyro-median()-x'] = body_gyro_x_test_reduce.median(axis=1)

test_pca_new['tBodyGyro-median()-y'] = body_gyro_y_test_reduce.median(axis=1)

test_pca_new['tBodyGyro-median()-z'] = body_gyro_z_test_reduce.median(axis=1)

test_pca_new['tTotalAcc-median()-x'] = total_acc_x_test_reduce.median(axis=1)

test_pca_new['tTotalAcc-median()-y'] = total_acc_y_test_reduce.median(axis=1)

test_pca_new['tTotalAcc-median()-z'] = total_acc_z_test_reduce.median(axis=1)
print(pca_new.shape)

print(test_pca_new.shape)
pd.isna(pca_new).sum() #LSTM이 잘 돌아가기 위해서는 Null이 없어야되므로 최종 확인
y_train = pd.read_csv('/kaggle/input/uci-har/UCI HAR Dataset for Kaggle/UCI HAR Dataset for Kaggle/train/y_train.txt', header=None)

y_train = pd.DataFrame(y_train)

y_train.value_counts()
y_train.columns = ['label']

y_train = pd.get_dummies(y_train['label']) #LSTM 결과값을 위해 라벨들을 one-hot encoding 적용하기.

y_train.head()
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(pca_new,y_train,test_size=0.3,random_state=42)
#LSTM은 3차원 Input이 필요해서 reshape

X_train = X_train.values.reshape(5146, 50, 1)

X_valid = X_valid.values.reshape(2206, 50, 1)

test_pca_new = test_pca_new.values.reshape(2947, 50, 1)

print(X_train.shape)

print(y_train.shape)
from keras.layers import LSTM

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import optimizers

from keras.callbacks import EarlyStopping

from keras.regularizers import l2

from keras import initializers



model = Sequential()

model.add(LSTM(64, recurrent_regularizer=l2(0.8), 

                bias_initializer='zeros',

               input_shape=(50, 1), return_sequences=True))

model.add(Dropout(0.5))

model.add(LSTM(32,input_shape=(50, 1), return_sequences=True))



model.add(LSTM(28,input_shape=(50, 1), recurrent_regularizer=l2(0.5), return_sequences=True))

model.add(Dropout(0.5))



model.add(LSTM(25,input_shape=(50, 1)))

model.add(Dense(6, activation='softmax')) # output = 6



adamax = optimizers.Adamax(lr=0.001, clipnorm=1.)

model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])

model.summary()
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)



history = model.fit(X_train, y_train, epochs=100,

                    validation_data = (X_valid, y_valid),

                    batch_size=128, verbose=1, callbacks=[early_stop])
y_pred_test = model.predict(test_pca_new)

print(y_pred_test)
y_pred_1 = [np.argmax(line)+1 for line in y_pred_test] #인덱스로 반환하였기에 +1 추가.
submit = pd.read_csv("/kaggle/input/uci-har/sample_submit.csv")

for i in range(len(y_pred_1)):

    submit['Label'][i] = y_pred_1[i]

submit.to_csv('LSTM_pca_new_predict.csv', index=False)

submit.head()
submit['Label'].value_counts()