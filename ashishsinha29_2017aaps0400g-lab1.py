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

from sklearn.preprocessing import MinMaxScaler,StandardScaler

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',')

test = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=',')

sample_sub = pd.read_csv('/kaggle/input/bitsf312-lab1/sample_submission.csv')
train.head(10)
train.apply(lambda row: row.astype(str).str.contains('\?').any(), axis=1).value_counts()
train.head(10)
train.apply(lambda row: row.astype(str).str.contains('\?').any(), axis=1).value_counts()
train['Class'].value_counts()
test.tail(5)
train['Number of Insignificant Quantities'].iloc[0]
for i in range(len(train)):

    if train['Number of Insignificant Quantities'].iloc[i]=='0' and train['Number of Quantities'].iloc[i]=='?' :

        train['Number of Quantities'][i]='2'

        print(train['Number of Quantities'][i])

    elif train['Number of Insignificant Quantities'].iloc[i]=='1' and train['Number of Quantities'].iloc[i]=='?':

        train['Number of Quantities'][i]='3'

        print(train['Number of Quantities'][i])

  

    if train['Number of Quantities'].iloc[i]=='3' and train['Number of Insignificant Quantities'].iloc[i]=='?':

        train['Number of Insignificant Quantities'][i]='1'

        print(train['Number of Insignificant Quantities'][i])

    elif train['Number of Quantities'].iloc[i]=='2' and train['Number of Insignificant Quantities'].iloc[i]=='?':

        train['Number of Insignificant Quantities'][i]='0'

        print(train['Number of Insignificant Quantities'][i])

    if train['Number of Quantities'].iloc[i]=='?' or train['Number of Insignificant Quantities'].iloc[i]=='?':

        print(i)

          
for i in range(len(train)):

    if train['Number of Special Characters'].iloc[i]=='?' or train['Number of Sentences'].iloc[i]=='?':

        print(i)

    if train['Number of Sentences'].iloc[i]=='?' :

        train['Number of Sentences'][i]=train['Number of Special Characters'][i]

    if train['Number of Special Characters'].iloc[i]=='?':

        train['Number of Special Characters'][i]=train['Number of Sentences'][i]

    

    



train.apply(lambda row: row.astype(str).str.contains('\?').any(), axis=1).value_counts()
train['Number of Quantities']=train['Number of Quantities'].astype(int)

train['Number of Insignificant Quantities']=train['Number of Insignificant Quantities'].astype(int)
print(train['Number of Insignificant Quantities'].value_counts())

print(train['Number of Quantities'].value_counts())
train['Size'].unique()
train=train.applymap(lambda x: np.NaN if x=='?' else x ).dropna()

train = train.reset_index(drop=True)

y=train['Class'].astype('category')

y
train['Size']=train['Size'].astype('category')

test['Size']=test['Size'].astype('category')
train.corr()
train=pd.get_dummies(train,prefix='Size',columns=['Size'],dtype=int)

test=pd.get_dummies(test,prefix='Size',columns=['Size'],dtype=int)
train.describe()
train.apply(lambda row: row.astype(str).str.contains('\?').any(), axis=1).value_counts()
train=train.astype(float)

test=test.astype(float)
train['CharacterperWords']=train['Total Number of Characters']/train['Total Number of Words']

train['CharacterperSentences']=train['Total Number of Characters']/train['Number of Sentences']

train['WordsperSentences']=train['Total Number of Words']/train['Number of Sentences']

train['ScoreDifficulty']=(train['Difficulty']/(train['Score']+1))



train['root']=(1/(train['Score']+1)**(0.3))

train['doubleroot']=1/(train['ScoreDifficulty']+0.1)**0.01

train['Root Number of Sentences']=train['Number of Sentences']**(0.25)



train['inv special']=1/(train['Number of Special Characters'])**0.5

test['CharacterperWords']=test['Total Number of Characters']/test['Total Number of Words']

test['CharacterperSentences']=test['Total Number of Characters']/test['Number of Sentences']

test['WordsperSentences']=test['Total Number of Words']/test['Number of Sentences']

test['ScoreDifficulty']=test['Difficulty']/(test['Score']+1)



test['root']=1/((test['Score']+1)**0.3)

test['doubleroot']=1/(test['ScoreDifficulty']+0.1)**0.01

test['Root Number of Sentences']=test['Number of Sentences']**(0.25)

test['inv special']=1/(test['Number of Special Characters'])**0.5
train.corr()
train.describe()
scaler=StandardScaler()

scaler.fit(train.drop(['ID','Class','Number of Quantities'],axis=1))

x=scaler.transform(train.drop(['ID','Class','Number of Quantities'],axis=1))

test_x=scaler.transform(test.drop(['ID','Number of Quantities'],axis=1))
y=train['Class']

y
x.shape,test_x.shape
from numpy import array

from numpy import argmax

from keras.utils import to_categorical

# one hot encode

encoded = to_categorical(y)

print(encoded)

# invert encoding

target = argmax(encoded[0])
from keras.models import Sequential

from keras.layers import Dropout,Dense

from keras.regularizers import l2
# Build the architecture

model = Sequential()

model.add(Dense(16,input_dim=20, activation='relu'))

model.add(Dropout(0.6))

model.add(Dense(32, activation='relu',bias_regularizer=l2(0.01)))

model.add(Dropout(0.6))

model.add(Dense(8, activation='relu',bias_regularizer=l2(0.01)))

model.add(Dropout(0.6))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
# Set callback functions to early stop training and save the best model so far

from keras.callbacks import EarlyStopping, ModelCheckpoint



callbacks = [EarlyStopping(monitor='val_loss', patience=30),

            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

#callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model.fit(x,encoded,validation_split=0.1,epochs=1000,verbose=1,batch_size=64,callbacks=callbacks)
model.fit(x,encoded,validation_split=0.1,epochs=1000,verbose=1,batch_size=64,callbacks=callbacks)
model.fit(x,encoded,validation_split=0.1,epochs=1000,verbose=1,batch_size=32,callbacks=callbacks)
model.fit(x,encoded,validation_split=0.1,epochs=1000,verbose=1,batch_size=32,callbacks=callbacks)
model.fit(x,encoded,validation_split=0.1,epochs=1000,verbose=1,batch_size=32,callbacks=callbacks)
model.fit(x,encoded,validation_split=0.05,epochs=1000,verbose=1,batch_size=2,callbacks=callbacks)
model.fit(x,encoded,validation_split=0.05,epochs=1000,verbose=1,batch_size=2,callbacks=callbacks)
model.save_weights('best_model.h5')

model.save('best_model.h5')

prediction=(model.predict(test_x))
test_out=[]

for i in range(len(test)):

    test_out.append(np.argmax(prediction[i]))
for i in range(len(sample_sub)):

    sample_sub['Class'][i]=test_out[i]

sample_sub    
df=sample_sub

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