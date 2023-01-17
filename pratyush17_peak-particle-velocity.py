import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

#import statsmodels.formula.api as sm

from sklearn.metrics import mean_squared_error,mean_absolute_error

from random import shuffle

from sklearn.model_selection import cross_val_score

%matplotlib inline

from sklearn import metrics

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/bl3312/blasting2.csv') 
data.describe()


data.head()
data = data.values

data = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

real_x = data[:,0:7]

real_y = data[:,8]
train_x,test_x,train_y,test_y= train_test_split(real_x,real_y,test_size=0.2,random_state=0)

mlr=LinearRegression()

mlr.fit(train_x,train_y)
pred=mlr.predict(test_x)
ic=mlr.intercept_

print(f"The Intercept is {ic}")
print('training score:',mlr.score(train_x,train_y)*100)

print('test score:',mlr.score(test_x,test_y)*100)
mse=mean_squared_error(pred,test_y)

mae=mean_absolute_error(pred,test_y)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pred)))

print(f'Mean  Squared Error is {mse}')

print(f'Mean Absolute Squared Error is {mae}')
colors=('red','green')

plt.scatter(pred,test_y)

plt.title('Scatter plot')

plt.xlabel('prediction')

plt.ylabel('test_y')

plt.show()
df1 = pd.DataFrame({'Actual': test_y.flatten(), 'Predicted': pred.flatten()})

df1
df2 = df1.head(30)

df2.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
mreg=LinearRegression()
cross_val_score(mreg,real_x,real_y,cv=30).mean()
lin_sc=cross_val_score(mreg,real_x,real_y,cv=20,scoring='neg_mean_absolute_error')
lin_sc=-lin_sc
lin_sc
np.mean(lin_sc)


from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error





forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_x, train_y)

melb_preds = forest_model.predict(test_x)



mse1=mean_squared_error(melb_preds,test_y)

mae1=mean_absolute_error(melb_preds,test_y)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, melb_preds)))

print(f'mean  sq error is {mse1}')

print(f'mean absol sq error is {mae1}')
df3 = pd.DataFrame({'Actual': test_y.flatten(), 'Predicted': melb_preds.flatten()})

df3
df4 = df3.head(30)

df2.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()