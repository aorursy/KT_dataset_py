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
import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv("../input/temperature-on-top-of-kathmandu-airport/KTM AIRPORT.csv")

data.set_index("DATE",inplace=True)

data.tail(10)
data.head(5)
data.isnull().sum().plot.barh()
#lets drop TMIN and TMAX



data['TAVG'] = (data['TMAX']+ data['TMIN'])/2

data.head()
data.drop(['STATION','NAME','PRCP','SNWD'],axis=1,inplace=True)

data.head()
data['TAVG'].isnull().sum()
data.dropna(inplace=True)

data.head()
#lets turn the temperature to celsius

data['TAVG'] = data['TAVG'].apply(lambda x: round((x - 32) * 5/9),2)

data['TMAX'] = data['TMAX'].apply(lambda x: round((x - 32) * 5/9),2) 

data['TMIN'] = data['TMIN'].apply(lambda x: round((x - 32) * 5/9),2)
data.head()
data.index
data.isnull().sum()
plt.figure(figsize=(20,10))

plt.style.use("fivethirtyeight")

y=list( range( 0,len(data['TAVG'][-730:]) ) )

plt.plot(y, data['TAVG'][-730:], label="Average Temperature")

plt.plot(y, data['TMAX'][-730:], label="Maximum Temperature")

plt.plot(y, data['TMIN'][-730:], label="Minimum Temperature")

plt.legend(loc="upper left")

plt.title("Temperature in Kathmandu")

plt.xlabel("Temperature in Degree Celsius")

plt.xlabel("X Days Ago")

plt.show()
data.describe()
data.index.shape
X=[]

for x in data.index:

    X.append(x.replace("-",""))

    

X[1:6]
from sklearn.preprocessing import MinMaxScaler

X=np.array(X)



minmax = MinMaxScaler(feature_range=(0,1))

minmax.fit(X.reshape(1,-1))

X = minmax.transform(X.reshape(1,-1))

X.ndim
data['TAVG'] = data['TAVG']

data.tail()
X=X.reshape(-1,1,1)

X.shape
def build_model():

    model = tf.keras.models.Sequential(name="Weather-Fn")

    

    model.add(tf.keras.layers.LSTM(64, input_shape=(32,1), activation="relu", return_sequences = True,name='layer-1'))

    model.add(tf.keras.layers.Dropout(0.2))

    

    model.add(tf.keras.layers.LSTM(64, activation="relu", return_sequences=False, name="layer-2"))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())

    

    model.add(tf.keras.layers.Dense(1,activation="sigmoid",name="Output_Layer"))

    print(model.summary())

    return model

    

    

model = build_model()
Y = np.array(data['TAVG'])

Y = Y.reshape(-1,1,1)



print(Y)



model.compile(

    loss="mse",

    optimizer="sgd",

    metrics=["accuracy"],

)

model.fit(X,Y,batch_size=32,epochs=10)
def predictweather(date):

    date = np.array(date)

    date = date.reshape(-1,1)

    

    date = minmax.fit_transform(date)

    x = date.reshape(-1,1,1)

    

    pred = model.predict(x)

    return pred



for dates in [20200528,20200529,20200530,20200628]:

    print(predictweather(dates))