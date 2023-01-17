import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
df = pd.read_csv("../input/red-wine-quality/winequality-red.csv")

df[:10]
df.shape
df.describe().T
df.isnull().any().sum()
X = df[['fixed acidity', 

        'volatile acidity', 

        'citric acid', 

        'residual sugar', 

        'chlorides', 

        'free sulfur dioxide', 

        'total sulfur dioxide', 

        'density', 

        'pH', 

        'sulphates',

        'alcohol']].values

y = df[["quality"]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = 0.2, 

                                                    random_state=0)
regression = LinearRegression()

regression.fit(X_train,y_train)

regression.coef_
regression.intercept_
y_pred = regression.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df1 = df.head(25)

df1
y_pred.shape
df1.plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))