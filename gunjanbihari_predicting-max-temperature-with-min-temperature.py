import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('../input/weatherww2/Summary of Weather.csv')

x=dataset.MinTemp.values

y=dataset.MaxTemp.values

x= x.reshape(-1,1)

y= y.reshape(-1,1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split (x,y,test_size=1/5,random_state=0)
from sklearn.linear_model import LinearRegression

regressor= LinearRegression()

regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')

plt.plot(x_train,regressor.predict(x_train),c='blue')

plt.show()
X=np.array([10,20,30,40,50]).reshape(-1,1)

print("Results")

for i in X:

    print("Min:",i,"Predicted Max:",regressor.predict([i]))
