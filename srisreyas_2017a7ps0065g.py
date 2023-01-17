import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Dropout

from sklearn.preprocessing import StandardScaler
df2 = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv")

df3= pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv")
df2.head()
df2=pd.get_dummies(df2,columns=['Size'],prefix=['Size'])

idc=df2.loc[:,'ID']

df2.drop(['ID'],axis=1,inplace=True)

df3=pd.get_dummies(df3,columns=['Size'],prefix=['Size'])

idc1=df3.loc[:,'ID']

df3.drop(['ID'],axis=1,inplace=True)
df2.drop('Size_?',axis=1,inplace=True)
df2=df2.replace('?',-1)

df3=df3.replace('?',-1)
df2['Class'].unique()
x_train=df2.drop(['Class'],axis=1)

y_train=df2['Class']

x_test=df3

s=StandardScaler()

x_train=s.fit(x_train).transform(x_train)

x_test=s.fit(x_test).transform(x_test)

x_train.shape, x_test.shape
model=Sequential()

model.add(Dense(16,input_dim=x_train.shape[1],activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(16,activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(8,activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_split=0,epochs=100,batch_size=10)


p=model.predict(x_test)

pl=[]

for i in p:

    pl.append(np.argmax(i))

pl
df=pd.DataFrame({'ID':idc1,'Class':pl})

df.head()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df)