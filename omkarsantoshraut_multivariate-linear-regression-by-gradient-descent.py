import numpy as np

from matplotlib import pyplot as plt

import pandas as pd

%matplotlib inline



df = pd.read_excel('../input/house-price/houseprice.xlsx')

df

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())

df


count = 5000

L = 0.000000166

n = float(df.area.count()) # number of entries in the data.



y = df.iloc[:, -1]

x1 =df['area']

x2 = df['bedrooms']

x3 = df['age']

x = df.iloc[: , :3]



B = np.zeros(x.shape[1])




for iteration in range(count):

    h = np.dot(x, B)

    loss = h-y

    gradient = (x.T.dot(loss))/ n

    B = B - (L * gradient)

    
B
a = B.dot([5000, 6, 10])

a
y_predicted = B.dot(x.T)

y_mean = df.price.mean()

r_square = 1- sum((y-y_predicted)*(y-y_predicted))/sum((y-y_mean)*(y-y_mean))

r_square