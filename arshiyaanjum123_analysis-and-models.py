                                                                                                                        # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
df=pd.read_csv("../input/insurance/insurance.csv")
df.head()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (18,8))
plt.subplot(131)
p1=sns.boxplot(y='age',x='sex',data=df).set_title("AGE VS SEX")
plt.subplot(132)
p2=sns.boxplot(y='bmi',x='sex',data=df).set_title("BMI VS SEX")
plt.subplot(133)
p3=sns.boxplot(y='charges',x='sex',data=df).set_title("CHARGES VS SEX")
plt.show()
sns.boxplot(x='smoker',y='charges',data=df)
df['sex']=df['sex'].replace('female',0)
df['sex']=df['sex'].replace('male',1)
df['smoker']=df['smoker'].replace('yes',1)
df['smoker']=df['smoker'].replace('no',0)
sns.countplot(df['region'])
one_hot=pd.get_dummies(df['region'], sparse=True)
df = df.drop('region',axis = 1)
# Join the encoded df
df = df.join(one_hot)

df.head()
df.corr()
df=df.dropna()
X=df.drop('charges',axis=1)
Y=df['charges']
# basic data splitting
from sklearn.model_selection import train_test_split

# basic data holdout
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
results=lm.fit(X_train,Y_train)
myPreds = lm.predict(X_test)
myActual = Y_test
# import matplotlib
plt.scatter(myPreds, myActual)
r_sq = lm.score(X_test, Y_test)
print('coefficient of determination:', r_sq)
print('intercept:', lm.intercept_)
print('Coefficients:', lm.coef_)
import statsmodels.api as sm
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
X2 = sm.add_constant(X_train)
est = sm.OLS(np.array(Y_train,dtype=float), np.array(X2,dtype=float),missing='drop')
est2 = est.fit()
print(est2.summary())
df=df.drop('sex',axis=1)
df.head()
X1=df.drop('charges',axis=1)
Y1=df['charges']
# basic data splitting
from sklearn.model_selection import train_test_split

# basic data holdout
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
results=lm.fit(X_train1,Y_train1)
myPreds = lm.predict(X_test1)
myActual = Y_test1
# import matplotlib
plt.scatter(myPreds, myActual)
r_sq = lm.score(X_test1, Y_test1)
print('coefficient of determination:', r_sq)
print('intercept:', lm.intercept_)
print('Coefficients:', lm.coef_)
import statsmodels.api as sm
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
X2 = sm.add_constant(X_train1)
est = sm.OLS(np.array(Y_train1,dtype=float), np.array(X2,dtype=float),missing='drop')
est2 = est.fit()
print(est2.summary())
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(Y_test1, myPreds))
import tensorflow as tf
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Dense(256, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.2)) 
model1.add(tf.keras.layers.Dense(128, activation='relu'))
model1.add(tf.keras.layers.Dense(32, activation='relu'))
model1.add(tf.keras.layers.Dense(1))
#compiling
from sklearn.model_selection import GridSearchCV
model1.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
# now run the model!
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=3, 
                   verbose=1)
history = model1.fit(np.array(X_train,dtype=float), np.array(Y_train,dtype=float), 
          validation_data=(np.array(X_test,dtype=float), np.array(Y_test,dtype=float)),
          epochs=100, batch_size= 30, 
          verbose=1, callbacks=[es])
import matplotlib.pyplot as plt

mae = history.history['mae']
val_mae = history.history['val_mae']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(mae) + 1)

plt.plot(epochs, mae, 'bo', label='Training MAE')
plt.plot(epochs, val_mae, 'orange', label='Validation MAE')
plt.title('Training and Validation MAE')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()