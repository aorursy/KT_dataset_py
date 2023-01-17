import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

data=pd.read_csv("../input/canada_per_capita_income.csv", header=None)
data.head()
data.describe()
data.shape
plt.figure(figsize=(16, 8))

plt.scatter(data[0],data[1],color="b",marker="o")

plt.xticks(np.arange(1960,2030,step=10))

plt.yticks(np.arange(3000,46000,step=2000))

plt.xlabel("year")

plt.ylabel("capita")

plt.title("capita and year")

x=data[0].values.reshape(-1,1)

y=data[1].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x2=np.array(X_train**2)

x3=np.array(X_train**3)

x4=np.array(X_train**4)
X=np.append(X_train,x2,axis=1)

X=np.append(X,x3,axis=1)

X.shape
reg = LinearRegression()

reg.fit(X,y_train)
predictions = reg.predict(X)

plt.figure(figsize=(16, 8))

plt.scatter(X_train,y_train,c='black')

plt.plot(X_train,predictions,c='blue',linewidth=2)

plt.xlabel("Years")

plt.ylabel("Capita ($)")

plt.show()
y_pred=reg.predict(X)
metrics=np.vstack((y,y_pred)).T
df=pd.DataFrame({'Actual':y_train.flatten(),'Predicted':y_pred.flatten()})

df
print('Mean Absolute Error:',mean_absolute_error(y_train,y_pred))
x2=np.array(X_test**2)

x3=np.array(X_test**3)

x4=np.array(X_test**4)
Xa=np.append(X_test,x2,axis=1)

Xa=np.append(Xa,x3,axis=1)

Xa.shape
y_predt=reg.predict(Xa)
metrics=np.vstack((y_test,y_predt)).T
df=pd.DataFrame({'Year':X_test.flatten() ,'Actual':y_test.flatten(),'Predicted':y_predt.flatten()})

df
print('Mean Absolute Error:',mean_absolute_error(y_test,y_predt))