# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly as py

import plotly.graph_objs as go

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot, plot

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid=pd.read_csv("/kaggle/input/covid19-datasetoctober/owid-covid-data.csv")
covid_india=covid[covid["location"]=="India"]
covid_india.isnull().sum()
covid_india=pd.DataFrame(covid_india)
covid_india["date"]=pd.to_datetime(covid_india["date"],infer_datetime_format=True)
data=covid_india["new_cases"].values
data1=covid_india["date"].values
covid_india.head()
data1[:5]
data[:5]
coco=pd.DataFrame({"Date":data1,"New_Cases":data})
coco.head()
coco["Date"].isnull().sum()
coco["New_Cases"].isnull().sum()
coco["New_Cases"].fillna(method='ffill',inplace=True)
coco["Date"]=pd.to_datetime(coco["Date"],infer_datetime_format=True)
coco=coco.set_index(['Date'])
coco.head()
def plo(j,m,k):

            fig = px.line(j,y=m)

            fig.update_layout(

            title={'text':k,'x':0.5},title_font_color="black")

            fig.update_xaxes(rangeslider_visible=True)

            fig.show()
plo(coco["New_Cases"],"New_Cases","New Cases in India")
len(coco)
train1=coco["New_Cases"]
train1=train1.values.reshape(-1,1)
from sklearn.preprocessing import MinMaxScaler

scale=MinMaxScaler(feature_range=(0,1))

scale=scale.fit(train1)
scale.data_min_
normalize=scale.transform(train1)
normalize[:5]
res=normalize.flatten()
res[:5]
tt={}



ww=[]

uu=[]

def make(a,b,ff):

        m=0

        k=0

       

        while k<ff:

            qq=[]

            for i in range(b):

                        u=a[m]

                        m=m+1

                        qq.append(u)

            ww.append(qq)  

            uu.append(a[m])

   

            m=k+1

            k=k+1

len(covid_india)
make(res,2,289)
ww[:5]
from numpy import array
www=array(ww)

www[:5]

uu=array(uu)
xtrain,ytrain=www[:230],uu[:230]
xtest,ytest=www[230:],uu[230:]
len(xtest)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Bidirectional
model = Sequential()

model.add(Bidirectional(LSTM(500, activation='relu',return_sequences=True,input_shape=(291,1))))

model.add(Dropout(0.2))

model.add(LSTM(200,activation="relu"))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
xtest.shape[0]
xtest.shape[1]
xtrain.shape[1]
xtrain
xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)

xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=50,batch_size=40,verbose=2)

y_pred=model.predict(xtest)
y_pred[:5]
ypred=scale.inverse_transform(y_pred)
ytest=ytest.reshape(-1,1)
ytest=scale.inverse_transform(ytest)
ypred[:5]
ytest[:5]
ypred[:5]
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(ytest,ypred))
ypre=ypred.flatten()
ypre
tttt=covid_india["new_cases"].iloc[:232].values
tttt=array(tttt)
tyyy=np.concatenate([tttt,ypre])
covi=coco.copy()

covi=pd.DataFrame(covi)

covi["predicted_cases"]=tyyy
ypre
def ivso(i,k,r1,r2):

    fig = px.line(covi,x=covi.index, y=[r1,r2])

    fig.update_layout(

    title={'text':k,'x':0.5},title_font_color="black")

    fig.update_xaxes(rangeslider_visible=True)

    fig.show()

ivso("New_Cases","new vs pred","New_Cases","predicted_cases")