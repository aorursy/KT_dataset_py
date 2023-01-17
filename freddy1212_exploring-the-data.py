# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv("../input/Iris.csv")

print (data)



# Any results you write to the current directory are saved as output.
d0 = data[data['Species'] == 'Iris-setosa']

x0 = d0['PetalLengthCm']

y0 = d0['PetalWidthCm']



d1 = data[data['Species'] == 'Iris-versicolor']

x1 = d1['PetalLengthCm']

y1 = d1['PetalWidthCm']



d2 = data[data['Species'] == 'Iris-virginica']

x2 = d2['PetalLengthCm']

y2 = d2['PetalWidthCm']



plt.scatter(x0, y0, color='red', alpha=0.3, label='Iris-setosa')

plt.scatter(x1, y1, color='green', alpha=0.3, label='Iris-versicolor')

plt.scatter(x2, y2, color='blue', alpha=0.3, label='Iris-virginica')

plt.legend(loc=4)

plt.xlabel('Petal Length (cm)')

plt.ylabel('Petal Width (cm)')

plt.show()