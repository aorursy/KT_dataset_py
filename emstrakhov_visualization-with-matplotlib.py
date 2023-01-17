import matplotlib as mpl

import matplotlib.pyplot as plt
# Версия библиотеки

mpl.__version__
import numpy as np



x = np.linspace(0, 10, 100)



plt.plot(x, np.sin(x))

plt.plot(x, np.cos(x))



plt.show() # в блокнотах эта команда не является обязательной
import numpy as np

x = np.linspace(0, 10, 100)



plt.plot(x, np.sin(x), '-')

plt.plot(x, np.cos(x), '--')
plt.plot(x, np.sin(x), '-')

plt.plot(x, np.cos(x), '--');
import numpy as np

x = np.linspace(0, 1.5, 100)



fig = plt.figure() # step 1



plt.plot(x, np.sqrt(x)) # step 2

plt.plot(x, np.exp(-2*x)) # step 2



fig.savefig('my_first_picture.png'); # step 3
# Как просмотреть файл?

from IPython.display import Image

Image('my_first_picture.png')
fig.canvas.get_supported_filetypes()
x = np.linspace(0, 10, 100)



plt.figure()  # create a plot figure



# create the first of two panels and set current axis

plt.subplot(2, 1, 1) # (rows, columns, panel number)

plt.plot(x, np.sin(x))



# create the second panel and set current axis

plt.subplot(2, 1, 2)

plt.plot(x, np.cos(x));
# First create a grid of plots

# ax will be an array of two Axes objects

fig, ax = plt.subplots(2)



# Call plot() method on the appropriate object

ax[0].plot(x, np.sin(x))

ax[1].plot(x, np.cos(x));