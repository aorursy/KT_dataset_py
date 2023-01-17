# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns # more plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class LinearRegressor:

    m = 0

    b = 0



    def linear_regression_last_squares(self, data_x, data_y):

        xmean = data_x.sum() / data_x.count()

        ymean = data_y.sum() / data_y.count()



        num = 0

        den = 0

        for x, y in zip(data_x, data_y):

            num += (x - xmean) * (y - ymean)

            den += (x - xmean) ** 2



        # m bestimmen

        self.m = num / den

        

        # b bestimmen

        self.b = ymean - self.m * xmean



    def predict(self, x):

        """

        Liefert für die Eingabe einer Fernsehzeit die Dauer des Tiefschlafs



        :param x: Fernsehzeit

        :return: Dauer des Tiefschlafes

        """

        # x in ermittelte Funtktion einsetzen

        return self.m * x + self.b

X = 'fernsehzeit'

Y = 'tiefschlaf'



data = pd.DataFrame({X: [0.3, 2.2, 0.5, 0.7, 1.0, 1.8, 3.0, 0.2, 2.3],

                     Y: [5.8, 4.4, 6.5, 5.8, 5.6, 5.0, 4.8, 6.0, 6.1]})



print(data.head(9))



plt.figure(figsize=(8,6))

sns.regplot(x=X, y=Y, data=data)

plt.show()
# Linearen Regressor erzeugen und Regressionsgerade ermitteln

lr = LinearRegressor()

lr.linear_regression_last_squares(data[X], data[Y])

print('m=', lr.m)

print('b=', lr.b)

print('y=',lr.m,'* x +',lr.b)
y = []

x = np.arange(0.0, 4.0, 0.5)

for i in x:

    y.append(lr.predict(i))



plt.figure(figsize=(8,6))

plt.ylim(4.0,7.0)

plt.xlim(0.0,3.5)

sns.regplot(x=X, y=Y, data=data)

sns.lineplot(x=x, y=y)

plt.show()