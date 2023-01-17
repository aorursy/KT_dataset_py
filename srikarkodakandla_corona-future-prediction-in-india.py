import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
a=pd.read_csv('../input/time-series-covid19/time_series_covid19_confirmed_global.csv')
a
b=a.transpose()
b
train=b[[131]].values
train_x=train[4:]
y=[]

for i in train_x:

    y.append(i[0])
y.append(3374)
#infected in india in the following days

y
x=list(range(len(y)))
#nth day

x
import seaborn as sns
#infection in the  nth day in india

sns.relplot(data=pd.DataFrame(y))

#x axis is days and y axis is infected people
x=np.array(x)

y=np.array(y)
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
'''You can uncomment to start training from begining or you can use weights of pre trained model'''

#predict.fit(x,y,batch_size=75,epochs=50000)
predict.load_weights('/kaggle/input/corona-india-future-predictions-weights/corona_future_prediction_india_weights.h5')
predict.summary()
'''You can uncomment the following things if you have started to train from the begining'''

#loss=predict.history.history

#loss_pd=pd.DataFrame(loss)

#loss_pd.plot()

#loss_pd.to_csv('loss.csv')
loss_pd=pd.read_csv('/kaggle/input/model-loss/loss.csv')
loss_pd=loss_pd.drop('Unnamed: 0',axis=1)
loss_pd
loss_pd.plot()
predict.predict([[74]]) #75th day is August 5 2020
predicted_values=[]

print('||',end='')

for i in range(1000):

    if i%9 is 0:

        print('=',end='')

    predicted_values.append(predict.predict([[i]]))

print('||')    
predicted_values
future=[]

for i in predicted_values:

    if round(i[0][0])<=0:

        future.append(0)

    else:

        future.append(round(i[0][0]))
future_df=pd.DataFrame({'Infected_people':future})
future_df
print("Actual_graph")

sns.relplot(data=pd.DataFrame(y))

print("predicted_graph")

sns.relplot(data=future_df[:75])
print("Prediction on future days")

future_df
print("Future predictions graph")

sns.relplot(data=future_df[:100])
def preprocess(day):

    return round(day)
#prediction on the nth day 

'''74th  day is 5th april 2020'''

'''change the day number to your own number to predict'''

day=74

if preprocess(predict.predict([[day]])[0][0] <= 0):

    infected=0

else:

    infected=(preprocess(predict.predict([[day]])[0][0]))

print("The Predicted infected people on the day",day,"in India are :",infected)    

sns.relplot(data=future_df[:day])
print("Day 89 is August 20")

day=89

if preprocess(predict.predict([[day]])[0][0] <= 0):

    infected=0

else:

    infected=(preprocess(predict.predict([[day]])[0][0]))

print("The Predicted infected people on the day",day,"in India are :",infected)    

sns.relplot(data=future_df[:day])
'''If it goes on like this in india then approximately 8426 people will get infected on Auguest 20 2020 so please keep masks and wash your hands to save yourself and others'''
'''Change the day number as your wish , set your reference to August 5th as 74th day'''

'''Enter any day number to predict the infection rate on the specific day'''



day=100





if preprocess(predict.predict([[day]])[0][0] <= 0):

    infected=0

else:

    infected=(preprocess(predict.predict([[day]])[0][0]))

print("The Predicted infected people on the day",day,"in India are :",infected)    

sns.relplot(data=future_df[:day])
future_df.to_csv('submission.csv')