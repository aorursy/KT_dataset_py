import sys



print(sys.version)
import numpy as np
x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])

y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])
# Type your code here - 



from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(x, y)

plt.xlabel("Area")

plt.ylabel("Price")
# Type your code here - 

def f(x, w0, w1):

    y = w0 + w1 * x

    return y
# Type your code here - 

def square_loss(x, y, w0, w1):

    loss = sum(np.square(y - (w0 + w1*x)))

    return loss
# Type your code here - 

def w_calculator(x, y):

    n = len(x)

    w1 = (n*sum(x*y) - sum(x)*sum(y))/(n*sum(x*x) - sum(x)*sum(x))

    w0 = (sum(x*x)*sum(y) - sum(x)*sum(x*y))/(n*sum(x*x)-sum(x)*sum(x))

    return w0, w1
# Type your code here - 

w_calculator(x, y)
# Type your code here - 

w0 = w_calculator(x, y)[0]

w1 = w_calculator(x, y)[1]

square_loss(x, y, w0, w1)
# Type your code here - 



x_temp = np.linspace(50,120,100) # Plot testing points

plt.scatter(x, y)

plt.plot(x_temp, x_temp*w1 + w0, 'r')
# Type your code here - 



# Section 2.4



f(150, w0, w1)
# Type your code here - 



# Type your code here - 

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x.reshape(len(x),1) , y) # Train
model.intercept_, model.coef_
model.predict([[150]])
# Type your code here - 



def w_matrix(x, y):

    w = (x.T * x).I * x.T * y

    return w
# Type your code here - 



x = np.matrix([[1,56],[1,72],[1,69],[1,88],[1,102],[1,86],[1,76],[1,79],[1,94],[1,74]])

y = np.matrix([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

w_matrix(x, y.reshape(10,1))
# Type your code here - 



import pandas as pd

df = pd.read_csv("../input/week-3-dataset/boston.csv")
# Type your code here - 



df.head()
# Type your code here - 



features = df[['crim', 'rm', 'lstat']]

features.describe()
# Type your code here - 



from IPython.display import Image

import os

Image("../input/week-3-images/ML.jpeg")
target = df['medv'] # Data of target



split_num = int(len(features)*0.7) # Find 70% position



train_x = features[:split_num] # Get features of training set

train_y = target[:split_num] # Get target of training set



test_x = features[split_num:] # Get features of testing set

test_y = target[split_num:] # Get target of testing set
# Type your code here -

from sklearn.linear_model import LinearRegression

model = LinearRegression() # Establish the model

model.fit(train_x, train_y) # Train

model.coef_, model.intercept_ # Print the parameters
# Type your code here -

preds = model.predict(test_x) # Train and predict

preds # Show results
# Type your code here -



def mae_value(y_true, y_pred):

    n = len(y_true)

    mae = sum(np.abs(y_true - y_pred))/n

    return mae
# Type your code here -



def mse_value(y_true, y_pred):

    n = len(y_true)

    mse = sum(np.square(y_true - y_pred))/n

    return mse
# Type your code here -



mae = mae_value(test_y.values, preds)

mse = mse_value(test_y.values, preds)



print("MAE: ", mae)

print("MSE: ", mse)
# Type your code here -


