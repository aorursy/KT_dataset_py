import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from statistics import mean

import gc





from sklearn.preprocessing import StandardScaler,OneHotEncoder



import os

from tqdm import tqdm

import random



import warnings

warnings.filterwarnings('ignore')

from keras import backend as K

K.tensorflow_backend._get_available_gpus()
from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Input, CuDNNLSTM, Flatten

from keras.optimizers import Adam

from keras.losses import mean_squared_error

from keras.callbacks import History
train_data = pd.read_csv('../input/train.csv',dtype = {'acoustic_data':np.float32,'time_to_failure':np.float32})





rows = 150000

segments = int(np.floor(train_data.shape[0] / rows))



X_train = np.zeros((segments,150000))

y_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['time_to_failure'])



for segment in tqdm(range(segments)):

    x = train_data.iloc[segment*rows:segment*rows+rows]

    y = x['time_to_failure'].values[-1]

    x = x['acoustic_data'].values

    y_train.loc[segment,'time_to_failure'] = y

    X_train[segment] = x

del train_data
gc.collect()
X_train.shape

y_train['time_to_failure'] = round(y_train['time_to_failure'])
y_train['time_to_failure'] = y_train['time_to_failure'].astype(np.int32)
ohe = OneHotEncoder()

y_train = ohe.fit_transform(np.array(y_train['time_to_failure']).reshape(-1,1))
y_train.shape
model = Sequential()

model.add(Conv1D(filters=40, kernel_size=20, strides=2, activation='relu', input_shape=(150000,1)))

model.add(MaxPooling1D(3))

model.add(Conv1D(filters=40, kernel_size=20, strides=1, activation='relu'))

model.add(MaxPooling1D(3))

model.add(Conv1D(filters=40, kernel_size=20, strides=1, activation='relu'))

model.add(MaxPooling1D(3))

model.add(CuDNNLSTM(8,return_sequences=True))

model.add(CuDNNLSTM(8,return_sequences=True))

#model.add(Flatten())

model.add(Conv1D(filters=40, kernel_size=10, strides=1, activation='relu'))

model.add(GlobalAveragePooling1D())

model.add(Dropout(rate=0.1))

model.add(Dense(17,activation = 'softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(X_train.reshape(-1,150000,1),y_train,epochs = 100, validation_split = 0.1,batch_size = 16)
def predictSubmission(seg_id):

    test_df = pd.read_csv('../input/test/' + seg_id + '.csv')

    #y = model.predict(prepareAd(test_df.acoustic_data.values).reshape(1,150000,1))*16

    #x = sc.fit_transform(test_df.acoustic_data.values.reshape(-1,1))

    x = test_df.acoustic_data.values

    y = model.predict(x.reshape(1,150000,1))

    return np.argmax(y)
submission = pd.read_csv('../input/sample_submission.csv')

submission['time_to_failure']=submission['seg_id'].apply(predictSubmission)

submission.to_csv('submission_8.csv',index=False)
submission.head()
import matplotlib.pyplot as plt

plt.hist(submission['time_to_failure'])

submission.to_csv('l.csv',index = False)
from IPython.display import HTML



def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='l.csv')
from IPython.display import HTML

html = '<a href = "l.csv">d</a>'

HTML(html)
def testInfo(seg_id):

    test_df = pd.read_csv('../input/test/' + seg_id + '.csv')

    return(test_df.acoustic_data.max())

#submission = pd.read_csv('../input/sample_submission.csv')

#submission['time_to_failure']=submission['seg_id'].apply(testInfo)