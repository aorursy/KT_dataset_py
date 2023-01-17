# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



from keras.layers import Dense, Dropout

from keras.models import Sequential



from sklearn.metrics import mean_absolute_error
df=pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv')
for i in df[df['Number of Quantities']=='?']['Number of Quantities'].index:

    #print(i)

    #print(df.loc[i]['Number of Quantities'])

    df.loc[df['Number of Quantities']==df.loc[i]['Number of Quantities'],'Number of Quantities']='2'
for i in df[df['Number of Insignificant Quantities']=='?']['Number of Insignificant Quantities'].index:

    #print(i)

    #print(df.loc[i]['Number of Insignificant Quantities'])

    df.loc[df['Number of Insignificant Quantities']==df.loc[i]['Number of Insignificant Quantities'],'Number of Insignificant Quantities']='0'
for i in df[df['Total Number of Words']=='?']['Total Number of Words'].index:

    #print(i)

    #print(df.loc[i]['Number of Insignificant Quantities'])

    df.loc[df['Total Number of Words']==df.loc[i]['Total Number of Words'],'Total Number of Words']='20'
for i in df[df['Number of Special Characters']=='?']['Number of Special Characters'].index:

    #print(i)

    #print(df.loc[i]['Number of Insignificant Quantities'])

    df.loc[df['Number of Special Characters']==df.loc[i]['Number of Special Characters'],'Number of Special Characters']='3'
ll=[]

for i in df[df['Difficulty']!='?']['Difficulty'].index:

    ll.append(float(df.loc[i]['Difficulty']))
import statistics

statistics.median(ll)
for i in df[df['Difficulty']=='?']['Difficulty'].index:

    #print(i)

    #print(df.loc[i]['Number of Insignificant Quantities'])

    df.loc[df['Difficulty']==df.loc[i]['Difficulty'],'Difficulty']='4.68845891272019'
df=pd.get_dummies(df,columns=['Size'], prefix=['Siz'])
df['Difficulty']=df['Difficulty'].astype('float64')
df['Number of Special Characters']=df['Number of Special Characters'].astype('int64')
df['Number of Quantities']=df['Number of Quantities'].astype('int64')

df['Number of Insignificant Quantities']=df['Number of Insignificant Quantities'].astype('int64')
df['Total Number of Words']=df['Total Number of Words'].astype('int64')
df.info()
missc=df.isnull().sum()

missc[missc>0]
df.drop(['ID','Siz_?'],axis=1,inplace=True)
df.info()
df.corr()
#plt.figure(figsize=(10,5)

#sns.heatmap(df.corr())

plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),mask=np.zeros_like(df.corr(),dtype=np.bool),square=True,annot=True)
df.head()
y=df['Class']
df.drop(['Class'],axis=1,inplace=True)
X=df
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import RobustScaler

scaler=RobustScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.fit_transform(X_test)
from keras.utils import to_categorical

y_train=to_categorical(y_train,6)

y_test=to_categorical(y_test,6)
X_train.shape, y_test.shape, X_test.shape, y_test.shape
from keras.regularizers import l2

from keras.layers import BatchNormalization

model=Sequential()

model.add(Dense(50,input_dim=13,kernel_initializer='uniform',activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

#model.add(Dropout(rate=0.1))

model.add(BatchNormalization())

model.add(Dense(40,activation='relu',kernel_initializer='uniform',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

#model.add(Dropout(rate=0.1))

model.add(BatchNormalization())

model.add(Dense(30,activation='relu',kernel_initializer='uniform',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

#model.add(Dropout(rate=0.1))

model.add(BatchNormalization())

model.add(Dense(20,activation='relu',kernel_initializer='uniform',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

#model.add(Dropout(rate=0.1))

model.add(Dense(6,activation='softmax'))
from keras import optimizers

adam=optimizers.Adam(lr=0.05,beta_1=0.9,beta_2=0.999, amsgrad=False)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,verbose=1,validation_split=0.2,epochs=80,batch_size=20)
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint



callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model.fit(X_train, y_train, validation_split=0.2, epochs=80,callbacks=callbacks,verbose=1)
test_results = model.evaluate(X_test, y_test, verbose=1)

print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
test_results1 = model.evaluate(X_train, y_train, verbose=1)

print(f'Test results - Loss: {test_results1[0]} - Accuracy: {test_results1[1]}%')
pred=model.predict_classes(X_test)
pred
y_test.argmax(1)
np.unique(pred,return_counts=True)
np.unique(y_test.argmax(1),return_counts=True)
x=np.unique(y_train.argmax(1),return_counts=True)
np.true_divide(x,np.sum(x[1]))
df1=pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
df1.head()
df1.info()
df1=pd.get_dummies(df1,columns=['Size'],prefix=['Siz'])
df1.isnull().sum()
df1.info()
X1=df1
X1.drop('ID',axis=1,inplace=True)
X1.head()
X1=scaler.fit_transform(X1)
fin1=model.predict_classes(X1)
np.unique(fin1,return_counts=True)
df2=pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
df3=pd.DataFrame(index=df2['ID'])

df3['Class']=fin1
xx=np.unique(fin1,return_counts=True)
xx
xx=xx/(np.sum(xx[1]))
xx
#[0.2972973 , 0.0472973 , 0.20608108, 0.09797297, 0.125     ,

 #       0.22635135]])
df3.to_csv('s7.csv')
df4=pd.read_csv('/kaggle/working/s7.csv')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df4, title = "Download CSV file", filename = "data.csv"):

    csv = df4.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df4)
#for the second submission-
df=pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv')
for i in df[df['Number of Quantities']=='?']['Number of Quantities'].index:

    #print(i)

    #print(df.loc[i]['Number of Quantities'])

    df.loc[df['Number of Quantities']==df.loc[i]['Number of Quantities'],'Number of Quantities']='2'
for i in df[df['Number of Insignificant Quantities']=='?']['Number of Insignificant Quantities'].index:

    #print(i)

    #print(df.loc[i]['Number of Insignificant Quantities'])

    df.loc[df['Number of Insignificant Quantities']==df.loc[i]['Number of Insignificant Quantities'],'Number of Insignificant Quantities']='0'
for i in df[df['Total Number of Words']=='?']['Total Number of Words'].index:

    #print(i)

    #print(df.loc[i]['Number of Insignificant Quantities'])

    df.loc[df['Total Number of Words']==df.loc[i]['Total Number of Words'],'Total Number of Words']='20'
for i in df[df['Number of Special Characters']=='?']['Number of Special Characters'].index:

    #print(i)

    #print(df.loc[i]['Number of Insignificant Quantities'])

    df.loc[df['Number of Special Characters']==df.loc[i]['Number of Special Characters'],'Number of Special Characters']='3'
ll=[]

for i in df[df['Difficulty']!='?']['Difficulty'].index:

    ll.append(float(df.loc[i]['Difficulty']))
import statistics

statistics.median(ll)
for i in df[df['Difficulty']=='?']['Difficulty'].index:

    #print(i)

    #print(df.loc[i]['Number of Insignificant Quantities'])

    df.loc[df['Difficulty']==df.loc[i]['Difficulty'],'Difficulty']='4.68845891272019'
df=pd.get_dummies(df,columns=['Size'], prefix=['Siz'])
df['Difficulty']=df['Difficulty'].astype('float64')
df['Number of Special Characters']=df['Number of Special Characters'].astype('int64')
df['Number of Quantities']=df['Number of Quantities'].astype('int64')

df['Number of Insignificant Quantities']=df['Number of Insignificant Quantities'].astype('int64')
df['Total Number of Words']=df['Total Number of Words'].astype('int64')
df.info()
missc=df.isnull().sum()

missc[missc>0]
df.drop(['ID','Siz_?'],axis=1,inplace=True)
df.corr()
#plt.figure(figsize=(10,5)

#sns.heatmap(df.corr())

plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),mask=np.zeros_like(df.corr(),dtype=np.bool),square=True,annot=True)
df.head()
y=df['Class']
df.drop(['Class','Total Number of Words','Total Number of Characters'],axis=1,inplace=True)
X=df
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import RobustScaler

scaler=RobustScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.fit_transform(X_test)
from keras.utils import to_categorical

y_train=to_categorical(y_train,6)

y_test=to_categorical(y_test,6)
X_train.shape, y_test.shape, X_test.shape, y_test.shape
model=Sequential()

model.add(Dense(30,input_dim=11,activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(60,activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(40,activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,verbose=1,validation_split=0.2,epochs=80,batch_size=20)
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint



callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model.fit(X_train, y_train, validation_split=0.2, epochs=80,callbacks=callbacks,verbose=1)
test_results = model.evaluate(X_test, y_test, verbose=1)

print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
test_results1 = model.evaluate(X_train, y_train, verbose=1)

print(f'Test results - Loss: {test_results1[0]} - Accuracy: {test_results1[1]}%')
pred=model.predict_classes(X_test)
pred
y_test.argmax(1)
np.unique(pred,return_counts=True)
np.unique(y_test.argmax(1),return_counts=True)
x=np.unique(y_train.argmax(1),return_counts=True)
np.true_divide(x,np.sum(x[1]))
df1=pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
df1.head()
df1.info()
df1=pd.get_dummies(df1,columns=['Size'],prefix=['Siz'])
df1.isnull().sum()
df1.info()
X1=df1
X1.drop(['ID','Total Number of Words','Total Number of Characters'],axis=1,inplace=True)
X1.head()
X1=scaler.fit_transform(X1)
fin1=model.predict_classes(X1)
np.unique(fin1,return_counts=True)
df2=pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
df3=pd.DataFrame(index=df2['ID'])

df3['Class']=fin1
xx=np.unique(fin1,return_counts=True)
xx
xx=xx/(np.sum(xx[1]))
xx
#[0.2972973 , 0.0472973 , 0.20608108, 0.09797297, 0.125     ,

 #       0.22635135]])
df3.to_csv('s3.csv')
df5=pd.read_csv('/kaggle/working/s3.csv')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df5, title = "Download CSV file", filename = "data.csv"):

    csv = df5.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df5)