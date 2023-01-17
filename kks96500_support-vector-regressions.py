import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/Position_Salaries.csv')
X=dataset.iloc[:,1:2].values

y=dataset.iloc[:,2].values
y
y=y.reshape(-1,1)

y.shape

#feature scaling

from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

sc_y=StandardScaler()

X=sc_X.fit_transform(X)

y=sc_y.fit_transform(y)
#now fitting the svr model to our dataset

from sklearn.svm import SVR

regressor=SVR(kernel='rbf')

regressor.fit(X,y)
#predicting the new reslut

y_predict=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#here we have used sc_y.inverse_transform to get the output in real format beacuse we have applied 

#featue scaling to the dataset

y_predict
#visualising the SVR results

plt.scatter(X,y,color='red')

plt.plot(X,regressor.predict(X),color='blue')

plt.title('SVR model')

plt.xlabel('Position')

plt.ylabel('salary')

plt.show()
#visualising the Svr results (with higher resolution and smoother curve)

X_grid = np.arange(min(X),max(X),0.1)

X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color='red')

plt.plot(X_grid,regressor.predict(X_grid),color='blue')

plt.title('SVR model')

plt.xlabel('Position')

plt.ylabel('salary')

plt.show()