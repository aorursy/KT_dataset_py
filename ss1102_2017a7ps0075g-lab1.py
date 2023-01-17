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
df = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv',na_values =['?'])
df.fillna(df.mode(),inplace= True)

df.isnull().sum()

df.dropna(inplace = True)
df=pd.get_dummies(df,columns=['Size'])
df.head()
df.info()
missc=df.isnull().sum()

missc[missc>0]
df.drop(['ID'],axis=1,inplace=True)
df.info()
df.corr()
df.head()
y=df['Class']
df.drop('Class',axis=1,inplace=True)
X=df
X.head()
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

model.add(Dense(30,input_dim=13,activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(60,activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(40,activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,verbose=1,validation_split=0.2,epochs=80,batch_size=20)
model.summary()
test_results = model.evaluate(X_test, y_test, verbose=1)

print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
pred=model.predict_classes(X_test)
pred
df1=pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
df1.head()
df1=pd.get_dummies(df1,columns=['Size'])
df1.isnull().sum()
df1.info()
X1=df1
X1.drop('ID',axis=1,inplace=True)
X1.head()
X1=scaler.fit_transform(X1)
fin1=model.predict_classes(X1)
fin1
df2=pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
df3=pd.DataFrame(index=df2['ID'])

df3['Class']=fin1
df3.to_csv('s2.csv')
df5=pd.read_csv('/kaggle/working/s2.csv')
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