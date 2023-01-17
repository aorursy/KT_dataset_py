# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams["figure.figsize"] = (20,10)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
plt.plot([1,2,3], [1,2,3]);
x = np.linspace(0, 10, 50)  # creating linear relationship

y = x  # the same



# Creating plot

plt.title('Linear relationship y = x') # plot title

plt.xlabel('x')

plt.ylabel('y')

plt.grid() # add grid

plt.plot(x, y) # plotting data
plt.plot(x, y, 'g--')  # g - green; -- type
x = np.linspace(0, 10, 50)  # linear

y1 = x



y2 = [i**2 for i in x] # quadratic

# Plotting

plt.title('Relations: y1 = x, y2 = x^2') # title

plt.xlabel('x')

plt.ylabel('y1, y2')

plt.grid() # add grid

plt.plot(x, y1, x, y2) # create one plot with two functions
x = [1, 5, 10, 15, 20]

y1 = [1, 7, 3, 5, 11]

y2 = [i*1.2 + 1 for i in y1]

y3 = [i*1.2 + 1 for i in y2]

y4 = [i*1.2 + 1 for i in y3]

plt.plot(x, y1, '-', x, y2, '--', x, y3, '-.', x, y4, ':')
x = np.linspace(0, 10, 50)  # linear

y1 = x



y2 = [i**2 for i in x]  # quadratic



# Построение графиков

plt.figure(figsize=(9, 9))  

plt.subplot(2, 1, 1)  # first plot with linear relationship

plt.plot(x, y1) # plotting first

plt.title('Relations: y1 = x, y2 = x^2') # заголовок

plt.ylabel('y1', fontsize=14) # y axis

plt.grid(True) # add grid

plt.subplot(2, 1, 2)  # first plot with quadratic relationship

plt.plot(x, y2) # plotting second

plt.xlabel('x', fontsize=14) # x axis

plt.ylabel('y2', fontsize=14) # y axis

plt.grid(True) # включение отображение сетки
cars = ['Alfa Romeo 159', 'BMW X7', 'Peugeot 3008', 'Renault Logan', 'Alfa Romeo 156']

quantity = [3, 12, 5, 21, 4]

plt.bar(cars, quantity)

plt.title('Cars Distribution')

plt.xlabel('Car')

plt.ylabel('Quantity')
plt.barh(cars, quantity)
x = [1999, 2003, 2007, 2013, 2018]  # some data on x axis

y = [25000, 19000, 16000, 12000, 6700]  # some data on y axis

plt.plot(x, y, label='Car price')

plt.title('Car price', fontsize=15)

plt.xlabel('Year', fontsize=12, color='blue')

plt.ylabel('Price', fontsize=12, color='blue')

plt.legend()

plt.grid(True)

plt.text(1999, 25000, 'We loose money!')
plt.plot(x, y, 'ro');  # car prices data; ro - 
plt.plot(x, y, 'bx');
x = [2, 7, 13, 22, 45]

y1 = [3, 3, 22, 7, 4]

y2 = [i*1.6 + 1 for i in y1]

y3 = [i*1.3 + 2 for i in y2]

y4 = [i*0.5 + 7 for i in y3]



# Setting sizes of the substrate

plt.figure(figsize=(12, 7))



# Creating plot next to each other

plt.subplot(2, 2, 1)

plt.plot(x, y1, '-')

plt.subplot(2, 2, 2)

plt.plot(x, y2, '--')

plt.subplot(2, 2, 3)

plt.plot(x, y3, '-.')

plt.subplot(2, 2, 4)

plt.plot(x, y4, ':')
plt.subplot(221)

plt.plot(x, y1, '-')

plt.subplot(222)

plt.plot(x, y2, '--')

plt.subplot(223)

plt.plot(x, y3, '-.')

plt.subplot(224)

plt.plot(x, y4, ':')
np.random.seed(123)

values = np.random.randint(10, size=(7, 7))

plt.pcolor(values);
np.random.seed(123)

values = np.random.randint(10, size=(7, 7))

plt.pcolor(values)

plt.colorbar()  # add color bar right next to matrix
percentage = [63, 5, 10, 11, 11]  # share for each product

labels = ['iPhone', 'Other products', 'iPad', 'Mac', 'Services']  # labels of apple products

fig, ax = plt.subplots()

ax.pie(percentage, labels=labels);  # plotting pie with shares as first argument and labels as second
percentage = [63, 5, 10, 11, 11]  # share for each product

labels = ['iPhone', 'Other products', 'iPad', 'Mac', 'Services']  # labels of apple products

fig, ax = plt.subplots()

ax.pie(percentage, labels=labels, wedgeprops=dict(width=0.5));