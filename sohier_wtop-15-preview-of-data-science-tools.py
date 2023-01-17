import numpy as np

x = np.arange(1, 10)

x
x ** 2
[val ** 2 for val in range(1, 10)]
M = x.reshape((3, 3))

M
M.T
np.dot(M, [5, 6, 7])
np.linalg.eigvals(M)
import pandas as pd

df = pd.DataFrame({'label': ['A', 'B', 'C', 'A', 'B', 'C'],

                   'value': [1, 2, 3, 4, 5, 6]})

df
df['label']
df['label'].str.lower()
df['value'].sum()
df.groupby('label').sum()
# run this if using Jupyter notebook

%matplotlib notebook
import matplotlib.pyplot as plt

plt.style.use('ggplot')  # make graphs in the style of R's ggplot
x = np.linspace(0, 10)  # range of values from 0 to 10

y = np.sin(x)           # sine of these values

plt.plot(x, y);         # plot as a line
from scipy import interpolate



# choose eight points between 0 and 10

x = np.linspace(0, 10, 8)

y = np.sin(x)



# create a cubic interpolation function

func = interpolate.interp1d(x, y, kind='cubic')



# interpolate on a grid of 1,000 points

x_interp = np.linspace(0, 10, 1000)

y_interp = func(x_interp)



# plot the results

plt.figure()  # new figure

plt.plot(x, y, 'o')

plt.plot(x_interp, y_interp);