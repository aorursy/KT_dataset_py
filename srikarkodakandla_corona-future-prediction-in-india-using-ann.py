import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
pd.read_csv('/kaggle/input/time-series-covid-19-conformed/time_series_covid_19_confirmed.csv')
a=pd.read_csv('/kaggle/input/time-series-covid-19-conformed/time_series_covid_19_confirmed.csv')
a
b=a.transpose()
b[[131]]
train=b[[131]].values
train_x=train[4:]
len(train_x)
import seaborn as sns
#infection in the  nth day in india

sns.relplot(data=pd.DataFrame(train_x))

#x axis is days and y axis is infected people
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
import tensorflow as tf
predict=Sequential([Dense(74,activation='relu',input_shape=(1,)),

                    Dense(74*2,activation='relu'),

                    Dense(74*2*2,activation='relu'),

                    Dense(74*2*2*2,activation='relu'),

                    Dense(74*2*2*2,activation='relu'),

                    Dense(1)])
predict.compile(optimizer='adam',loss='mse',metrics=['mse','mae','accuracy'])
y=train_x
y=np.append(y,20100)
x=np.array(range(len(y)))
y=np.array(y,dtype=int)
'''You can uncomment to start training from begining or you can use weights of pre trained model'''

#predict.fit(x,y/y[-1],epochs=5000,batch_size=600)
#adam=tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#predict.compile(optimizer=adam,loss='mse',metrics=['mse','mae','accuracy'])
#predict.fit(x,y/y[-1],epochs=50000,batch_size=600)
#un comment this if you start training from begining

predict.load_weights('/kaggle/input/weights/weights (1).h5')
predict.summary()
#You can uncomment the following things if you have started to train from the begining

#loss=predict.history.history

#loss_pd=pd.DataFrame(loss)

#loss_pd.plot(figsize=(12,8))

#loss_pd.to_csv('loss.csv')
loss_pd=pd.read_csv('/kaggle/input/corona/loss (1).csv')
loss_pd=loss_pd.drop('Unnamed: 0',axis=1)
loss_pd
loss_pd.plot()
y[90]
predict.predict([[90]])*y[-1]
predicted_values=[]

print('||',end='')

for i in range(1000):

    if i%9 is 0:

        print('=',end='')

    predicted_values.append(predict.predict([[i]])*y[-1])

print('||')
f=predicted_values

future=[]

for i in range(len(f)):

    if (f[i][0][0])<0:

        future.append(0)

    else:

        future.append(round(f[i][0][0]))
future[92]
future_df=pd.DataFrame({'Infected_people':future})
#future_df
print("Actual_graph")

sns.relplot(data=pd.DataFrame(y))

print("predicted_graph")

sns.relplot(data=future_df[:90])
print("Prediction on future days")

future_df
print("Future predictions graph")

sns.relplot(data=future_df[:90])
def preprocess(day):

    return round(day)
#prediction on the nth day 

'''set your reference to August 21th as 89th day'''

'''change the day number to your own number to predict'''

day=89

if preprocess((predict.predict([day]))[0][0]*y[-1] <= 0):

    infected=0

else:

    infected=(preprocess((predict.predict([day]))[0][0]*y[-1]))

print("The Predicted infected people on the day",day,"in India are :",infected)    

sns.relplot(data=future_df[:day])
print("Day 90 is August 22")

day=90

if preprocess((predict.predict([day]))[0][0]*y[-1] <= 0):

    infected=0

else:

    infected=(preprocess((predict.predict([day]))[0][0]*y[-1]))

print("The Predicted infected people on the day",day,"in India are :",infected)    

sns.relplot(data=future_df[:day])
'''Change the day number as your wish , set your reference to August 22nd as 90th day'''

'''Enter any day number to predict the infection rate on the specific day'''



day=100



if preprocess((predict.predict([day]))[0][0]*y[-1] <= 0):

    infected=0

else:

    infected=(preprocess((predict.predict([day]))[0][0]*y[-1]))

print("The Predicted infected people on the day",day,"in India are :",infected)    

sns.relplot(data=future_df[:day])
'''If you see the future prediction is more than the actual one then you have to understand that the corona virus is spreading than normal'''
"""The day in which the future prediction is more than the actual value then we can say that the corona virus in india is decreasing and we can expect the government will soon remove the lockdown in india"""
"""Hope this notebook will help you in predicting the virus state in india"""