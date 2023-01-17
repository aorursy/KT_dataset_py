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
tr=pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
ts=pd.read_csv('/kaggle/input/sf-crime/test.csv.zip')
tr=pd.concat([tr.drop('DayOfWeek',axis=1),pd.get_dummies(tr['DayOfWeek'])],axis=1)
ts=pd.concat([ts.drop('DayOfWeek',axis=1),pd.get_dummies(ts['DayOfWeek'])],axis=1)
tr=pd.concat([tr.drop('PdDistrict',axis=1),pd.get_dummies(tr['PdDistrict'])],axis=1)
ts=pd.concat([ts.drop('PdDistrict',axis=1),pd.get_dummies(ts['PdDistrict'])],axis=1)
tr['year']=0

for i in range(tr.shape[0]):

    tr['year'][i]=tr['Dates'][i][0:4]
ts['year']=0

for i in range(ts.shape[0]):

    ts['year'][i]=ts['Dates'][i][0:4]
tr=pd.concat([tr.drop('year',axis=1),pd.get_dummies(tr['year'])],axis=1)
ts=pd.concat([ts.drop('year',axis=1),pd.get_dummies(ts['year'])],axis=1)
tr['month']=0

for i in range(tr.shape[0]):

    tr['month'][i]=tr['Dates'][i][5:7]
ts['month']=0

for i in range(ts.shape[0]):

    ts['month'][i]=ts['Dates'][i][5:7]
tr['month'].replace(to_replace=[i for i in range(1,13)],value=['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec'],inplace=True)
ts['month'].replace(to_replace=[i for i in range(1,13)],value=['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec'],inplace=True)
tr=pd.concat([tr.drop('month',axis=1),pd.get_dummies(tr['month'])],axis=1)
ts=pd.concat([ts.drop('month',axis=1),pd.get_dummies(ts['month'])],axis=1)
# tr['place']=np.zeros(tr.shape[0])

# for i in range(tr.shape[0]):

#     tr['place'][i]=tr['Address'][i][-9:]

    
# ts['place']=np.zeros(ts.shape[0])

# for i in range(ts.shape[0]):

#     ts['place'][i]=ts['Address'][i][-9:]

    
tr['Date']=np.zeros(tr.shape[0])

for i in range(tr.shape[0]):

    tr['Date'][i]=tr['Dates'][i][8:10]
ts['Date']=np.zeros(ts.shape[0])

for i in range(ts.shape[0]):

    ts['Date'][i]=ts['Dates'][i][8:10]
tr=pd.concat([tr.drop('Date',axis=1),pd.get_dummies(tr['Date'])],axis=1)
ts=pd.concat([ts.drop('Date',axis=1),pd.get_dummies(ts['Date'])],axis=1)
# tr=pd.concat([tr.drop('place',axis=1),pd.get_dummies(tr['place'])],axis=1)
# ts=pd.concat([ts.drop('place',axis=1),pd.get_dummies(ts['place'])],axis=1)
tr['time']=np.zeros(tr.shape[0])

for i in range(tr.shape[0]):

    tr['time'][i]=tr['Dates'][i][11:13] + tr['Dates'][i][14:16]
ts['time']=np.zeros(ts.shape[0])

for i in range(ts.shape[0]):

    ts['time'][i]=ts['Dates'][i][11:13] + ts['Dates'][i][14:16]
tr.shape
tr.drop(['Resolution','Address','Dates','Descript'],axis=1,inplace=True)
ts.drop(['Address','Dates'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(tr.drop(['Category'],axis=1),pd.get_dummies(tr['Category']))
xtr.shape
from keras.layers import Dense,Activation,Dropout,Conv1D,Flatten

from keras.models import Sequential
#without dates



md=Sequential()

md.add(Dense(300,input_shape=(1,76)))

md.add(Activation('relu'))

md.add(Dense(200))

md.add(Activation('relu'))

md.add(Dense(100))

md.add(Activation('relu'))

md.add(Dense(50))

md.add(Activation('relu'))

md.add(Dense(39))

md.add(Activation('softmax'))

md.summary()

# md=Sequential()

# md.add(Conv1D(32,2,activation='relu',padding='same',input_shape=76))

# # md.add(Conv1D(64,2,activation='relu',padding='same'))

# # md.add(Conv1D(32,2,activation='relu',padding='same'))

# # md.add(Dropout(0.25))

# md.add(Dense(3,activation='softmax'))

# md.summary()
md.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
md.fit(xtr,ytr,batch_size=2048,epochs=100,verbose=1,validation_data=(xts,yts))
pred=md.predict(ts.drop('Id',axis=1))
pred.shape
sub=pd.DataFrame(data=pred,columns=ytr.columns)
sub=pd.DataFrame(np.where(sub.T == sub.T.max(), 1, 0),index=sub.columns).T
sub['Id']=ts['Id']

sub.head()
#sub.set_index('Id',inplace=True)

sub.head()
sam=pd.read_csv('/kaggle/input/sf-crime/sampleSubmission.csv.zip')

sam.head()
sub=sam[sam.columns]
sub.to_csv('../working/submission.csv', index=False)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()

reg.fit(xtr,ytr)

reg.score(xts,yts)