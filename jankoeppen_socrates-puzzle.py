import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression





data_dict = {

  "in": [1, 3, 4, 5, 6, 7],

  "out": [8, 46, 77, 116, 163, 218],

}



numbers = pd.DataFrame.from_dict(data_dict)

numbers



plt.scatter(numbers['in'], numbers['out'])

plt.xlabel("Input")

plt.ylabel("Output")

plt.show()
plt.scatter(numbers['in'], numbers['out'])

plt.xlabel("Input")

plt.ylabel("Output")



lr = LinearRegression()

lr.fit(numbers[['in']], numbers['out'])



y_pred = lr.predict(numbers[['in']])



plt.plot(numbers['in'], y_pred, color='blue', linewidth=3)



plt.show



lr.predict([[2]])