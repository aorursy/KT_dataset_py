# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras as K
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv1D, Flatten, BatchNormalization, MaxPooling1D, Input, TimeDistributed, concatenate, LSTM
from keras.constraints import maxnorm
from keras.utils import to_categorical
from sklearn import preprocessing
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/final1/DF_distEmb_prev2_OHE.csv")
#print(list(train.columns.values))
train.drop(train[train['Productivity'] >=1000].index, inplace = True) 
train.drop(train[train['Productivity'] <=1].index, inplace = True)
#train.drop(train[train['Crop_Rice'] ==0].index, inplace = True)
print(train.shape)
train=train.dropna(axis=0)
print(train.shape)
train=(train)/train.max()
#print(train.head())
train=train.dropna(axis=1)
train=train.dropna(axis=0)
#print(train.isnull().any())
print(train.head())
tempmax0=[]
tempmax1=[]
tempmax2=[]
tempmin0=[]
tempmin1=[]
tempmin2=[]
rain1=[]
rain2=[]
rain3=[]
yeild0=[]
yeild1=[]
yeild2=[]
area0=[]
area1=[]
area2=[]
prod0=[]
prod1=[]
prod2=[]
df=train[['Jan_max', 'Feb_max', 'Mar_max', 'Apr_max', 'May_max', 'Jun_max', 'Jul_max', 'Aug_max', 'Sep_max', 'Oct_max', 'Nov_max', 'Dec_max']].copy()
tempmax0=df.to_numpy()
df=train[['Jan_min', 'Feb_min', 'Mar_min', 'Apr_min', 'May_min', 'Jun_min', 'Jul_min', 'Aug_min', 'Sep_min', 'Oct_min', 'Nov_min', 'Dec_min']].copy()
tempmin0=df.to_numpy()
df=train[['Jan_max_prev', 'Feb_max_prev', 'Mar_max_prev', 'Apr_max_prev', 'May_max_prev', 'Jun_max_prev', 'Jul_max_prev', 'Aug_max_prev', 'Sep_max_prev', 'Oct_max_prev', 'Nov_max_prev', 'Dec_max_prev']].copy()
tempmax1=df.to_numpy()
df=train[['Jan_min_prev', 'Feb_min_prev', 'Mar_min_prev', 'Apr_min_prev', 'May_min_prev', 'Jun_min_prev', 'Jul_min_prev', 'Aug_min_prev', 'Sep_min_prev', 'Oct_min_prev', 'Nov_min_prev', 'Dec_min_prev']].copy()
tempmin1=df.to_numpy()
df=train[['Jan_max_prev2', 'Feb_max_prev2', 'Mar_max_prev2', 'Apr_max_prev2', 'May_max_prev2', 'Jun_max_prev2', 'Jul_max_prev2', 'Aug_max_prev2', 'Sep_max_prev2', 'Oct_max_prev2', 'Nov_max_prev2', 'Dec_max_prev2']].copy()
tempmax2=df.to_numpy()
df=train[['Jan_min_prev2', 'Feb_min_prev2', 'Mar_min_prev2', 'Apr_min_prev2', 'May_min_prev2', 'Jun_min_prev2', 'Jul_min_prev2', 'Aug_min_prev2', 'Sep_min_prev2', 'Oct_min_prev2', 'Nov_min_prev2', 'Dec_min_prev2']].copy()
tempmin2=df.to_numpy()
df=train[[ 'Jan_rain', 'Feb_rain', 'Mar_rain ', 'Apr_rain', 'May_rain', 'Jun_rain', 'Jul_rain', 'Aug_rain', 'Sep_rain', 'Oct_rain', 'Nov_rain', 'Dec_rain']].copy()
temprain0=df.to_numpy()
df=train[['Jan_rain_prev', 'Feb_rain_prev', 'Mar_rain _prev', 'Apr_rain_prev', 'May_rain_prev', 'Jun_rain_prev', 'Jul_rain_prev', 'Aug_rain_prev', 'Sep_rain_prev', 'Oct_rain_prev', 'Nov_rain_prev', 'Dec_rain_prev']].copy()
temprain1=df.to_numpy()
df=train[['Jan_rain_prev2', 'Feb_rain_prev2', 'Mar_rain _prev2', 'Apr_rain_prev2', 'May_rain_prev2', 'Jun_rain_prev2', 'Jul_rain_prev2', 'Aug_rain_prev2', 'Sep_rain_prev2', 'Oct_rain_prev2', 'Nov_rain_prev2', 'Dec_rain_prev2']].copy()
temprain2=df.to_numpy()
df=train[['Area','Area_prev', 'Production_prev', 'Productivity_prev', 'Area_prev2', 'Production_prev2', 'Productivity_prev2']].copy()
mainshit=df.to_numpy()

a=train.drop(columns=['Jan_max', 'Feb_max', 'Mar_max', 'Apr_max', 'May_max', 'Jun_max', 'Jul_max', 'Aug_max', 'Sep_max', 'Oct_max', 'Nov_max', 'Dec_max','Jan_min', 'Feb_min', 'Mar_min', 'Apr_min', 'May_min', 'Jun_min', 'Jul_min', 'Aug_min', 'Sep_min', 'Oct_min', 'Nov_min', 'Dec_min', 'Jan_rain', 'Feb_rain', 'Mar_rain ', 'Apr_rain', 'May_rain', 'Jun_rain', 'Jul_rain', 'Aug_rain', 'Sep_rain', 'Oct_rain', 'Nov_rain', 'Dec_rain','Jan_max_prev', 'Feb_max_prev', 'Mar_max_prev', 'Apr_max_prev', 'May_max_prev', 'Jun_max_prev', 'Jul_max_prev', 'Aug_max_prev', 'Sep_max_prev', 'Oct_max_prev', 'Nov_max_prev', 'Dec_max_prev', 'Jan_min_prev', 'Feb_min_prev', 'Mar_min_prev', 'Apr_min_prev', 'May_min_prev', 'Jun_min_prev', 'Jul_min_prev', 'Aug_min_prev', 'Sep_min_prev', 'Oct_min_prev', 'Nov_min_prev', 'Dec_min_prev', 'Jan_rain_prev', 'Feb_rain_prev', 'Mar_rain _prev', 'Apr_rain_prev', 'May_rain_prev', 'Jun_rain_prev', 'Jul_rain_prev', 'Aug_rain_prev', 'Sep_rain_prev', 'Oct_rain_prev', 'Nov_rain_prev', 'Dec_rain_prev','Jan_max_prev2', 'Feb_max_prev2', 'Mar_max_prev2', 'Apr_max_prev2', 'May_max_prev2', 'Jun_max_prev2', 'Jul_max_prev2', 'Aug_max_prev2', 'Sep_max_prev2', 'Oct_max_prev2', 'Nov_max_prev2', 'Dec_max_prev2', 'Jan_min_prev2', 'Feb_min_prev2', 'Mar_min_prev2', 'Apr_min_prev2', 'May_min_prev2', 'Jun_min_prev2', 'Jul_min_prev2', 'Aug_min_prev2', 'Sep_min_prev2', 'Oct_min_prev2', 'Nov_min_prev2', 'Dec_min_prev2', 'Jan_rain_prev2', 'Feb_rain_prev2', 'Mar_rain _prev2', 'Apr_rain_prev2', 'May_rain_prev2', 'Jun_rain_prev2', 'Jul_rain_prev2', 'Aug_rain_prev2', 'Sep_rain_prev2', 'Oct_rain_prev2', 'Nov_rain_prev2', 'Dec_rain_prev2'])
#print(list(a.columns.values))
print(mainshit.shape)
df=a[['Productivity']].copy()
yeild0=df.to_numpy()
b=a.drop(columns=['Production', 'Productivity','Crop_Year_prev','Crop_Year_prev2'])

print(list(b.columns.values))
print(b.shape)
X=b.to_numpy()
print(X.shape)
sum=X.sum()
print(np.isnan(sum))
print(len(temprain0))
print(temprain0[0])
abc=[]

for j in range(len(temprain0)):
    p=[]
    o=[]
    u=[]
    for i in range(12):
        p=p+[[tempmax2[j][i],tempmin2[j][i],temprain2[j][i]]]
    for i in range(12):
        o=o+[[tempmax1[j][i],tempmin1[j][i],temprain1[j][i]]]
    for i in range(12):
        u=u+[[tempmax0[j][i],tempmin0[j][i],temprain0[j][i]]]
    pou=[p]+[o]+[u]
    abc=abc+[pou]
    if(j%1000==0):
        print(j)

abcd=np.array(abc)
print(abcd)

inputA = Input(shape=(170))
inputB = Input(shape=(3,12,3))
inputC = Input(shape=(7))

# the first branch operates on the first input

w = Dense(10, activation="elu")(inputC)
#w = Dropout(0.3)(w)
w = Dense(10, activation="elu")(w)
#w = Dropout(0.3)(w)
w = Dense(5, activation="elu")(w)
w = Dense(1, activation="elu")(w)
w = Model(inputs=inputC, outputs=w)


x = Dense(200, activation="relu")(inputA)
x = Dropout(0.2)(x)
x = Dense(60, activation="elu")(x)
#x = Dropout(0.4)(x)
x = Dense(5, activation="elu")(x)
x = Dense(1, activation="elu")(x)
x = Model(inputs=inputA, outputs=x)


# the second branch opreates on the second input
y = TimeDistributed(Conv1D(16,kernel_size=3,activation='relu'))(inputB)
y = TimeDistributed(MaxPooling1D(pool_size=2))(y)
y = TimeDistributed(Flatten())(y)
y = LSTM(units = 4, return_sequences= False)(y)
y = Dense(units=5,activation='elu')(y)
y = Dense(units=1, activation='elu')(y)

y = Model(inputs=inputB, outputs=y)
# combine the output of the two branches
combined = concatenate([x.output, y.output, w.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(3, activation="elu")(combined)
z = Dense(1, activation="tanh")(z)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[x.input, y.input, w.input], outputs=z)
model.summary()
print(yeild0)

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)
opt = Adam(lr=0.001, clipnorm=1)
model.compile(optimizer=opt, loss='mape',  metrics=correlation_coefficient_loss)
history=model.fit(x=(X,abcd, mainshit), y=yeild0, epochs=6000, validation_split=0.1, batch_size=512)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['correlation_coefficient_loss'])
plt.plot(history.history['val_correlation_coefficient_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


