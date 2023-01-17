import numpy as np

from keras.layers import Dense, Activation

from keras.models import Sequential

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pandas as pd 
df = pd.read_csv('../input/datamahedi/data.csv',delimiter=';')

df.columns
df.head(7)
df.isnull().sum()
df = df.dropna()
df.describe(include='all')
#import seaborn as sns 

#sns.pairplot(df)
df['Y']
x=df.drop(['Y'], axis=1)

y=df['Y'].values

y

#if you write the code y = df['y'] you will get a series of values, but if you add (.values) you will get an array of values which can be reshaped after that. 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=False)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)





y= y.reshape(-1,1)

y=sc.fit_transform(y)
from keras import Sequential

from keras.layers import Dense

def build_regressor():

    regressor = Sequential()

    regressor.add(Dense(units=380, input_dim=380))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])

    return regressor
from keras.wrappers.scikit_learn import KerasRegressor

regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=100)
results=regressor.fit(X_train,y_train)
y_pred= regressor.predict(X_test)
fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
"""Import the required modules"""

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score
reg = MLPRegressor(hidden_layer_sizes=(380,100,10),activation="relu" ,random_state=1, max_iter=2000).fit(X_train,y_train)
y_pred=reg.predict(X_test)

print("The Score with ", (r2_score(y_pred, y_test)))
fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Measured')



ax.set_ylabel('Predicted')

plt.show()