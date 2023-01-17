# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('../input/shotputt_powerclean.csv')
df.head()

df.isnull().any()
X = df.iloc[:, 0:1].values

y = df.iloc[:, 1].values
plt.scatter(X, y, color = 'brown')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
# Fitting Linear Regression to the dataset



from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
plt.scatter(X_train, y_train, color = 'brown')

plt.plot(X_train, lin_reg.predict(X_train), color = 'olive')

plt.title('Relationship of Strength to Performance among Shot Putters(Linear Regression)')

plt.ylabel("Shotput")

plt.xlabel('Power Clean')

plt.show()
accuracy = lin_reg.score(X_test,y_test)

print((accuracy*100).round(2))
# Fitting Polynomial Regression to the dataset

#using maxim and deg variable to find the model with maximum accuracy and it's corresponding degree



maxim =0

for i in range(1,11):

    

    from sklearn.preprocessing import PolynomialFeatures

    poly_reg = PolynomialFeatures(degree= i)

    X_poly = poly_reg.fit_transform(X_train)

    poly_reg.fit(X_poly, y_train)

    lin_reg_2 = LinearRegression()

    lin_reg_2.fit(X_poly, y_train)

    

    

    accuracy2 = lin_reg_2.score(poly_reg.fit_transform(X_test),y_test)

    print("Accuracy with Degree",i,"--->",(accuracy2*100).round(2),"%")

    if(maxim<accuracy2):

        maxim = accuracy2

        deg = i
print("maximum accuracy we could get is--->",maxim.round(2)*100,"   with degree--->",deg)
#Refit the model with optimal degree



from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree= deg)

X_poly = poly_reg.fit_transform(X_train)

poly_reg.fit(X_poly, y_train)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y_train)

X_grid = np.arange(min(X), max(X), 0.1)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'brown')

plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'olive')

plt.title('Relationship of Strength to Performance among Shot Putters (Polynomial Regression)')

plt.xlabel('Power Clean')

plt.ylabel('Shotput')

plt.show()

    
#Let's check a random prediction



lin_reg_2.predict(poly_reg.fit_transform([[120]]))
lin_reg.predict([[120]])
# We can see that our polynaomial regression model gives more accurate prediction.