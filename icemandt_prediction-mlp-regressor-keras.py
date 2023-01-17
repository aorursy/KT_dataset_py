import pandas as pd 
import csv 
from pandas import DataFrame as df
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential 
from keras.layers import Dense , Dropout
from sklearn.cross_validation import train_test_split 
from sklearn.linear_model import LinearRegression
data_1 = pd.read_csv('train.csv')
data_1.shape
data_1.head(5)
data_2 = pd.read_csv('test.csv')
data_2.shape
data_2.head(5)
output_target = data_1.target
output_target.shape
data_1 = data_1.drop('target',axis=1)
data_1.shape
#frame =[data_1,data_2]
#data = pd.concat(frame)
#data.shape
le = LabelEncoder()
ID = le.fit_transform(data_1.ID)
data_1 = data_1.drop('ID' , axis = 1)
ID = pd.DataFrame(ID, columns=['ID'])
data_1 = ID.join(data_1)
data_1.shape
#data_1 = data_1.fillna(0)
scale = MinMaxScaler()
data_1 = scale.fit_transform(data_1)
data_1 = pd.DataFrame(data_1)
data_1.shape
output_target = output_target.reshape(-1,1)
output_target = scale.fit_transform(output_target)
output_target = pd.DataFrame(output_target)
x_train , x_test , y_train ,y_test = train_test_split(data_1 , output_target , test_size=0.3,random_state=0)
regressor = Sequential()
regressor.add(Dense(output_dim=1000,init='uniform',activation='relu',input_dim=4992))
regressor.add(Dropout(0.2))
regressor.add(Dense(output_dim=500,init='uniform',activation='relu',input_dim=1000))
regressor.add(Dropout(0.2))
regressor.add(Dense(output_dim=100,init='uniform',activation='relu',input_dim=500))
regressor.add(Dropout(0.2))
regressor.add(Dense(output_dim=10,init='uniform',activation='relu',input_dim=100))
regressor.add(Dropout(0.2))
regressor.add(Dense(output_dim=1,init='uniform',activation='linear'))
regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])

regressor.summary()
regressor.fit(x_train , y_train ,epochs=200,batch_size=64)
ID_2 = data_2.ID
ID_2 = le.fit_transform(ID_2)
data_2 = data_2.drop('ID',axis=1)
ID_2 = pd.DataFrame(ID_2,columns=['ID'])
data_2 = ID_2.join(data_2)
values = regressor.predict(data_2)
#score_validation = regressor.score(y_test , x_test)
#print(score_validation)
values = abs(values)
final_df = le.inverse_transform(data_2.ID)
final_df = pd.DataFrame(final_df , columns=['ID'])
final_df_2 = pd.DataFrame(values , columns=['Target'])
final_test_target = final_df.join(final_df_2)
final_test_target.shape
final_test_target.head(20)
final_test_target.to_csv('final_target.csv' , index=False)
