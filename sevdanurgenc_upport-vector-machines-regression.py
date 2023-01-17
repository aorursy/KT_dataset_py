import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
dataset = pd.read_csv('../input/position-salaries/position-salaries.csv')
X = dataset.iloc[:,1:2].values.astype(float)
y = dataset.iloc[:,2:3].values.astype(float)
dataset
from sklearn.preprocessing import StandardScaler
 
print(X, '\n')
print(y)
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X,y)
y_pred = regressor.predict([[6.5]])
y_pred
plt.figure(dpi=200)
plt.scatter(X, y, color = 'blue')
plt.plot(X, regressor.predict(X), color = 'green')
#green modelimizi gösterir.
plt.title('Support Vector Machines Regression Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#standart scaler yapılmadığında; modelimiz linear bir modele dönüşüyor ve değerler yüksek çıkıyor.
#hiç bir şekilde değişiklik olmuyor.