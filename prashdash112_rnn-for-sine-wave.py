import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,LSTM,Dense
from tensorflow.keras.callbacks import EarlyStopping
x=np.linspace(0,50,501)
y=np.sin(x)
plt.plot(x,y)
df=pd.DataFrame(data=y,index=x,columns=['sin'])
df.head(5)
test_percent=0.1
test_point=np.round(len(df)*test_percent)
test_index=int(len(df)-test_point)
test_index
train=df.iloc[:451]
test=df.iloc[451:]
sc=MinMaxScaler()
sc.fit(train)
scaled_train=sc.transform(train)
scaled_test=sc.transform(test)
length=50
batch_size=1
generator=TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=batch_size)
len(generator)
# Model definition
n_features=1
model=Sequential()
model.add(SimpleRNN(50,input_shape=(length,n_features)))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.summary()
model.fit_generator(generator,epochs=5)
losses=pd.DataFrame(model.history.history)
losses.plot()
# evaluation batch
first_eval_batch= scaled_train[-length:]
print(first_eval_batch.shape)
first_eval_batch=first_eval_batch.reshape(1,length,1)
print(first_eval_batch.shape)
print(model.predict(first_eval_batch))
print(scaled_test[0])
test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0] 
    
    # store prediction 
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
test_predictions
true_pred=sc.inverse_transform(test_predictions)
print(true_pred.shape)
print(test.shape)
test['rnn_predictions']=true_pred
test.head(5)
test.plot(figsize=(10,5))
early_stop = EarlyStopping(monitor='val_loss',patience=2)
length=49

generator = TimeseriesGenerator(scaled_train,scaled_train,
                               length=length,batch_size=1)

validation_generator= TimeseriesGenerator(scaled_test,scaled_test,length=length,batch_size=batch_size)
model=Sequential()
model.add(LSTM(50,input_shape=(length,n_features)))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit_generator(generator,epochs=5,validation_data=validation_generator,callbacks=[early_stop])
test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0] 
    
    # store prediction 
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
test_predictions
true_pred=sc.inverse_transform(test_predictions)
print(true_pred.shape)
test['LSTM Predictions'] = true_pred
test.plot(figsize=(14,5))