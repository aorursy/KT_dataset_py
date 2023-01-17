import numpy as np 

import plotly.express as px,plotly.graph_objs as go

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression

import os

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Activation

from tensorflow.keras.optimizers import Adam

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date:date.year)
df.drop(['date'],axis=1,inplace=True)
px.defaults.color_continuous_scale

px.scatter(df,x='long',y='lat',color='price',color_continuous_scale=px.colors.sequential.Emrld)
df= df.drop(['zipcode','id'],axis=1)
df2 = df[df['price']<=3000000]
df2.describe()
X = df2.drop('price',axis=1)

y = df2['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)

X_test = scale.transform(X_test)
lg = LinearRegression()
lg.fit(X_train,y_train)
pred1 = lg.predict(X_test)
pred1 = pd.Series(pred1)
pred1.index = y_test.index
model = Sequential()



model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(1))



model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train.values,

          validation_data=(X_test,y_test.values),

          batch_size=128,epochs=1000)
losses = pd.DataFrame(model.history.history)
losses.plot()
predictions = model.predict(X_test)
a=[]

for i in range(len(predictions)):

    a.append(predictions[i][0])
pred = pd.Series(a)
pred.index = y_test.index
p=pd.concat([pred,pred1,y_test],axis=1)
p.columns=['Neural Network','Linear Regression','Original']
p['Neural net score']=None

for i in p.index:



    

    if p['Original'][i] >= p['Neural Network'][i]:

        p['Neural net score'][i] = p['Original'][i] - p['Neural Network'][i]

    else:

        p['Neural net score'][i] = p['Neural Network'][i] - p['Original'][i] 
p['linear score']=None

for i in p.index:

    if p['Original'][i] >= p['Linear Regression'][i]:

        p['linear score'][i] = p['Original'][i] - p['Linear Regression'][i]

    else:

        p['linear score'][i] = p['Linear Regression'][i] - p['Original'][i] 
linear_score = np.array(p['linear score'])
Neural_score = np.array(p['Neural net score'])
np.average(Neural_score)
np.average(linear_score)
p
plt.title('Linear Regression')

plt.scatter(y_test,pred1)

plt.plot(y_test,y_test,'r')

plt.title('Neural Network')

plt.scatter(y_test,predictions)

plt.plot(y_test,y_test,'r')