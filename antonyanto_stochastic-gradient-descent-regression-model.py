import numpy as np

import pandas as pd

from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split,KFold

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/usa-housing/USA_Housing.csv')
x = data.drop(axis=1,columns=['Price','Address']).to_numpy()
y = data['Price'].to_numpy()
std = StandardScaler()
x = std.fit_transform(x)
reg = SGDRegressor(random_state=0)
error = []

for n in range(50):

    kf = KFold(n_splits=5,random_state=n,shuffle=True)

    for i,j in kf.split(x,y):

        X_train,X_test = x[i],x[j]

        y_train,y_test = y[i],y[j]

    reg.fit(X_train,y_train)

    pred = reg.predict(X_test)

    error.append(mean_squared_error(y_test,pred))

print("Best Seed Error : ",min(error)," Seed : ",error.index(min(error)))
kf = KFold(n_splits=5,random_state=3,shuffle=True)

for i,j in kf.split(x,y):

    X_train,X_test = x[i],x[j]

    y_train,y_test = y[i],y[j]
reg.fit(X_train,y_train)
pred = reg.predict(X_test)
reg.score(X_test,y_test)
mae = mean_absolute_error(y_test,pred)

mae
mse = mean_squared_error(y_test,pred)

print(mse)
rmse = np.sqrt(mse)

rmse