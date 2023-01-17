import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
uploaded = files.upload()
car_df=pd.read_csv('Car_Purchasing_Data.csv',encoding='ISO-8859-1')
sns.pairplot(car_df)
car_df
x=car_df.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'],axis=1)
x
y=car_df['Car Purchase Amount']
from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
x_scaled=scaler.fit_transform(x)
from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
y_scaled=scaler.fit_transform(y)

y = y.values.reshape(-1,1)
x_scaled
scaler.data_max_
scaler.data_min_
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_scaled,y_scaled)
import tensorflow.keras 
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(25,input_dim=5,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(1,activation='linear'))
model.summary()
model.compile(optimizer='adam',loss='mean_squared_error')
epochs_hist=model.fit(X_train,y_train,epochs=100,batch_size=50,verbose=1,validation_split=0.2)
X_test =np.array([[1,50,50000,1000,600000]])
y_predict=model.predict(X_test)
y_predict
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('model loss progress during training')
plt.ylabel('training & validation loss')
plt.xlabel('epoch number')
plt.legend(['Training loss','Validation loss'])