
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt

dataset = pd.read_csv('../input/Salary_Data_own.csv')
dataset

X = dataset.iloc[: , :-1].values
Y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train ,Y_test = train_test_split(X ,Y , test_size = 1/3 , random_state = 0)

from sklearn.linear_model import LinearRegression
req = LinearRegression()
req.fit(X_train,Y_train)

Y_pred = req.predict(X_test)

plt.scatter (X_test , Y_test )
plt.plot(X_train, req.predict(X_train))
plt.title('Salary vs Experiance')
plt.xlabel('Experiance')
plt.ylabel('salary')
plt.show()




# Any results you write to the current directory are saved as output.
