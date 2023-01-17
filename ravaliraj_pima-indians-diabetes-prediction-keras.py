

import pandas as pd

from keras.models import Sequential

from keras.layers import *

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split



data_frame = pd.read_csv('../input/diabetes.csv')



data_frame.head()

training_data_df, testing_data_df = train_test_split(data_frame, test_size=0.25)



training_data_df.head()

testing_data_df.head()
### scale the values to 0&1

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))



scaled_train = scaler.fit_transform(training_data_df)

scaled_test = scaler.transform(testing_data_df)



### creating pandas dataframe for scaled data

scaled_train_df = pd.DataFrame(data=scaled_train,columns=training_data_df.columns.values)

scaled_test_df = pd.DataFrame(data=scaled_test,columns=testing_data_df.columns.values)



scaled_train_df.head()

scaled_test_df.head()
#### creating a neural net model



model = Sequential()

model.add(Dense(10,input_dim=8,activation='elu'))

model.add(Dense(30,activation='elu'))

model.add(Dense(10,activation='elu'))

model.add(Dense(1,activation='linear'))



model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
X_train = scaled_train_df.drop(['Outcome'],axis=1).values

Y_train = scaled_train_df['Outcome'].values



X_test = scaled_test_df.drop(['Outcome'],axis=1).values

Y_test = scaled_test_df['Outcome'].values



#Fitting the Model

pima_model= model.fit(X_train,Y_train,shuffle=True,verbose=2,epochs=50)
test_error_rate  = model.evaluate(X_test,Y_test,verbose=0)



print('Test error rate is {}'.format(test_error_rate[0]))

print('Accuracy on test dataset {}'.format(test_error_rate[1]*100))
plt.plot(pima_model.history['acc'])

plt.title('Accuracy of Model along epochs')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train','Test'])

plt.show()