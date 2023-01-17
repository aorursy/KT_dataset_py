import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



df = pd.read_csv('../input/prediction/House_prediction.csv')

data=df.groupby('city').mean()

print(data)

data['area'].plot()



plt.xlabel('City')

plt.ylabel('Area')

plt.title('Area v/s city')

plt.show()



data['rooms'].plot()

plt.xlabel('City')

plt.ylabel('No. of Rooms')

plt.title('No. of Rooms v/s city')

plt.show()



data['bathroom'].plot()

plt.xlabel('City')

plt.ylabel('No. of Bathroom')

plt.title('No. of Bathroom v/s city')

plt.show()



data['parking spaces'].plot()

plt.xlabel('City')

plt.ylabel('Parking Space')

plt.title('Parking Space v/s city')

plt.show()



data['hoa (R$)'].plot()

plt.xlabel('City')

plt.ylabel('HOA(R$)')

plt.title('HOA(R$) v/s city')

plt.show()



data['rent amount (R$)'].plot()

plt.xlabel('Rent')

plt.ylabel('Area')

plt.title('Rent v/s city')

plt.show()



data['property tax (R$)'].plot()

plt.xlabel('City')

plt.ylabel('Property Tax')

plt.title('Property Tax v/s city')

plt.show()





data['fire insurance (R$)'].plot()

plt.xlabel('City')

plt.ylabel('Fire Insaurance')

plt.title('Fire Insaurance v/s city')

plt.show()



data['total (R$)'].plot()

plt.xlabel('City')

plt.ylabel('Total(R$)')

plt.title('Total(R$) v/s city')

plt.show()



data.plot()

plt.xlabel('City')

plt.title('All')

plt.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  

from sklearn.linear_model import LinearRegression

from sklearn import metrics



df = pd.read_csv('../input/prediction/House_prediction.csv')



train=df.iloc[0:10000,:]

test=df.iloc[10000:,:]





x_train=train['area'].values.reshape(-1,1)

y_train=train['rent amount (R$)'].values.reshape(-1,1)



x_test=test['area'].values.reshape(-1,1)

y_test=test['rent amount (R$)'].values.reshape(-1,1)



regressor = LinearRegression()  

regressor.fit(x_train, y_train)



y_pred = regressor.predict(x_test)

ans = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

print(ans)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))





plt.scatter(x_test, y_test,   color='blue')

plt.plot(x_test, y_pred, color='red', linewidth=2)

plt.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  

from sklearn.linear_model import LinearRegression

from sklearn import metrics



df = pd.read_csv('../input/prediction/House_prediction.csv')





train=df.iloc[0:10000,:]

test=df.iloc[10000:,:]





x_train=train['rooms'].values.reshape(-1,1)

y_train=train['rent amount (R$)'].values.reshape(-1,1)



x_test=test['rooms'].values.reshape(-1,1)

y_test=test['rent amount (R$)'].values.reshape(-1,1)



regressor = LinearRegression()  

regressor.fit(x_train, y_train)



y_pred = regressor.predict(x_test)

ans = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

print(ans)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))





plt.scatter(x_test, y_test,   color='blue')

plt.plot(x_test, y_pred, color='red', linewidth=2)

plt.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  

from sklearn.linear_model import LinearRegression

from sklearn import metrics



df = pd.read_csv('../input/prediction/House_prediction.csv')





train=df.iloc[0:10000,:]

test=df.iloc[10000:,:]





x_train=train['bathroom'].values.reshape(-1,1)

y_train=train['rent amount (R$)'].values.reshape(-1,1)



x_test=test['bathroom'].values.reshape(-1,1)

y_test=test['rent amount (R$)'].values.reshape(-1,1)



regressor = LinearRegression()  

regressor.fit(x_train, y_train)



y_pred = regressor.predict(x_test)

ans = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

print(ans)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))





plt.scatter(x_test, y_test,   color='blue')

plt.plot(x_test, y_pred, color='red', linewidth=2)

plt.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  

from sklearn.linear_model import LinearRegression

from sklearn import metrics



df = pd.read_csv('../input/prediction/House_prediction.csv')

df.animal[df.animal == 'acept'] = 1

df.animal[df.animal == 'not acept'] = 0 

df.furniture[df.furniture == 'furnished'] = 1

df.furniture[df.furniture == 'not furnished'] =0

df.floor[df.floor == '-'] =0



train=df.iloc[0:10000,:]

test=df.iloc[10000:,:]

x_train=train[['area' , 'rooms' , 'bathroom', 'parking spaces', 'floor', 'animal', 'furniture', 'hoa (R$)','property tax (R$)','fire insurance (R$)']].values

y_train=train['rent amount (R$)'].values



x_test=test[['area' , 'rooms' , 'bathroom', 'parking spaces', 'floor', 'animal', 'furniture', 'hoa (R$)', 'property tax (R$)','fire insurance (R$)']].values

y_test=test['rent amount (R$)'].values



regressor = LinearRegression()  

regressor.fit(x_train, y_train)





y_pred = regressor.predict(x_test)

ans = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(ans)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
