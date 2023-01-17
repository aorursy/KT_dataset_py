import numpy as np

import pandas as pd

from tensorflow import keras

from keras.layers import Dense, Dropout

from keras.models import Sequential

#from keras.datasets import boston_housing

from sklearn import preprocessing

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler
nd = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',')

nd.replace("?",np.nan,inplace=True)

df2=nd.dropna(axis = 0, how ='any')

#df2['Sig']=df2['Number of Quantities']-df2['Number of Insignificant Quantities']

#df2.dtypes

#df2.head(17)
#df2 = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',')

#df2.replace("?",np.nan,inplace=True)

#df2.dropna()

df3 = df2.copy()

df3 = pd.get_dummies(df2, columns=['Size'], prefix = ['Size'])

#df2.Size=pd.get_dummies(Y2)

size={'Small':1,'Medium':2,'Big':3}

#df2.Size=[size[item] for item in df2.Size]

#df3=df3.astype(float)

#df3['Sig']=df3['Number of Quantities']-df3['Number of Insignificant Quantities']

X11=df3.drop("Class",axis=1)

X1=X11.drop("ID",axis=1)

#X1=X2.drop("Number of Quantities",axis=1)

#X=X1.drop("Number of Insignificant Quantities",axis=1)

df1 = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=',')

df11 = df1.copy()

df11 = pd.get_dummies(df1, columns=['Size'], prefix = ['Size'])

#df1.Size=[size[item] for item in df1.Size]

#df11=df11.astype(float)

#df11['Sig']=df11['Number of Quantities']-df11['Number of Insignificant Quantities']

Xtest=df11.drop("ID",axis=1)

#Xtest1=Xtest2.drop("Number of Quantities",axis=1)

#Xtest=Xtest1.drop("Number of Insignificant Quantities",axis=1)
Y1=df2['Class']

Y=pd.get_dummies(Y1)

X1.shape,Y.shape,Xtest.shape
corr=df3.corr()

abs(corr['Class']).sort_values()

#corr
#X=X1.drop(['Total Number of Words','Total Number of Characters'],axis=1)

#Xtest1=Xtest.drop(['Total Number of Words','Total Number of Characters'],axis=1)

#X.shape
from keras.utils import plot_model

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.regularizers import l2

visible = Input(shape=(13,))

hidden1 = Dense(26, activation='relu')(visible)

hidden2 = Dense(13, activation='relu')(hidden1)

hidden3 = Dense(26, activation='relu')(hidden2)

output = Dense(6, activation='softmax')(hidden3)

model = Model(inputs=visible, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(X1,Y, validation_split=0.2, epochs=200,batch_size=5)
pred = model.predict(Xtest)

a=np.zeros((len(pred),2),int)

for i in range(len(pred)):

    a[i][0]=371+i

    a[i][1]=np.argmax(pred[i])

df = pd.DataFrame(data=a, columns=["ID", "Class"] )

df.to_csv('mycsvfile.csv',index=False)
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