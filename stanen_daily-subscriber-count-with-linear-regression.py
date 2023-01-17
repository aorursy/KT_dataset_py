import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
# Storing the data in a pandas data frame.



df = pd.read_csv("../input/subscriber-count/Subscriber.csv")

df.head(10)
x = df.iloc[:, :-1]

y = df.iloc[:, 1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
simplelinearRegression = LinearRegression()

simplelinearRegression.fit(x_train, y_train)
y_predict = simplelinearRegression.predict(x_test)
predict = pd.DataFrame(y_predict)
predict.apply(np.round)
i = 21

while i <= 28:

  print("Total number of increase in subscribers on September %d ==>" %(i) , int(simplelinearRegression.predict([[i]])))

  i= i+1