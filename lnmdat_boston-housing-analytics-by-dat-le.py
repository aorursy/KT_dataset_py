from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

seed = 13

np.random.seed(seed)
#Import the data

dataset = np.loadtxt('../input/bostonhousing.csv')

df = pd.DataFrame(dataset,columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])

print(df.info())
"""Statistical Information about dataset"""

print('The average crime rate per capita by town: ',df['CRIM'].mean())

print('The average number of rooms per dwelling: ',df['RM'].mean())

print('The average age of units: ',df['AGE'].mean())

print('The average pupil-teacher ratio by town: ',df['PTRATIO'].mean())

print('The average value of homes in $1000: ',df['MEDV'].mean())
#Histogram of MEDV

df['MEDV'].hist()

#histogram of full tax rate

df['TAX'].hist()
#index of accessibility to radial highways

sns.factorplot('RAD',data=df,kind='count')

#Number of samples near bank river or not

sns.factorplot('CHAS', data=df, kind='count')

#Relationship between MEDV and Crime Rate and residential density

sns.scatterplot(x="MEDV",y="CRIM",size="ZN", hue="ZN",data=df)

#Relationship between MEDV and pupil-teacher

sns.scatterplot(x="MEDV",y="PTRATIO",data=df)

#Relationship between MEDV and nitric oxides concentration

sns.scatterplot(x="MEDV",y="NOX",data=df)

#Relationship between MEDV and TAX Rate

sns.scatterplot(x="MEDV",y="TAX",data=df)
"""Training Part"""

#Use 400 samples for training

X_train= dataset[0:400,0:13]

Y_train= dataset[0:400,13]

#Use 106 samples for evaluation

X_eval= dataset[400:507,0:13]

Y_eval= dataset[400:507,13]
#Define the model

model = Sequential()

model.add(Dense(13,input_dim=13, activation='relu'))

model.add(Dense(6, activation='relu'))

model.add(Dense(1))

#Compile the model

model.compile(loss='mse',optimizer='adam',metrics=['mse'])

#In the training, there are some points that if increase the epochs, there is no improvement.

#Here I set that if 3 consecutive rows do not improve, stop

#The plot here to see the variance of MSE while training the set

early_stopping_monitor = EarlyStopping(patience=3)

history=model.fit(X_train,Y_train, epochs=500,verbose=0,callbacks=[early_stopping_monitor])

plt.plot(history.epoch, np.array(history.history['mean_squared_error']),

           label='Train Loss')

plt.xlabel('Epoch')

plt.ylabel('Mean Squared Error [1000$]')

plt.legend()

plt.ylim([0, 200])

plt.show()
#Testing the trained model and provde the mean squared error of testing set

[los,mse] = model.evaluate(X_eval,Y_eval)

print('Testing set Mean Squared Error is ${:7.2f}'.format(mse*1000))
#use to visualize the evaluation by using predict function and compare with actual value

prediction = model.predict(X_eval).flatten()

plt.scatter(Y_eval,prediction)

plt.xlabel('Real Values [$1000]')

plt.ylabel('Prediction Value [$1000]')

plt.axis('equal')

plt.xlim(plt.xlim())

plt.ylim(plt.ylim())

plt.plot([-100, 100], [-100, 100])

plt.show()

#display the histogram of prediction error

error = prediction - Y_eval

plt.hist(error, bins = 50)

plt.xlabel("Prediction Error [1000$]")

_ = plt.ylabel("Count")