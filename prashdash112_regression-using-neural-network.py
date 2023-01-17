import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
filename="../input/neural-net-regression-data/fake_reg.csv"

df=pd.read_csv(filename)

df
sns.pairplot(df)
from sklearn.model_selection import train_test_split
X=df[['feature1','feature2']].values

y=df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
X_train.max()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
model=Sequential()



model.add(Dense(4,activation='relu'))

model.add(Dense(4,activation='relu'))

model.add(Dense(4,activation='relu'))

model.add(Dense(1))



model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=X_train,y=y_train,epochs=250)
loss_df=pd.DataFrame(model.history.history)

loss_df.plot()
model.evaluate(x=X_test,y=y_test,verbose=10)
model.evaluate(x=X_train,y=y_train,verbose=10)
prediction=model.predict(X_test)

prediction
test_pred=pd.Series(prediction.reshape(300,))

test_pred
pred_df=pd.DataFrame(data=y_test,columns=['True value(y)'])

pred_df
pred_df=pd.concat([pred_df,test_pred],axis=1)

pred_df
pred_df.columns=['true value(y)','predicted_vals']
pred_df
sns.scatterplot(x='true value(y)',y='predicted_vals',data=pred_df)
from sklearn.metrics import mean_absolute_error,mean_squared_error
ae=mean_absolute_error(pred_df['true value(y)'],pred_df['predicted_vals'])

se=mean_squared_error(pred_df['true value(y)'],pred_df['predicted_vals'])

print('mean absolute error is {}'.format(ae))

print('mean squared error is {}'.format(se))
df.describe()
from tensorflow.keras.models import load_model
model.save('first_neural_net.h5')
later_model=load_model('first_neural_net.h5')