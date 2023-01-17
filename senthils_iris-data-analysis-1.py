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

# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv")
iris1 = iris.assign(SpeciesType = lambda x: (100))
iris1.ix[iris1.Species == 'Iris-setosa','SpeciesType'] = 100;
iris1.ix[iris1.Species == 'Iris-versicolor','SpeciesType'] = 200;
iris1.ix[iris1.Species == 'Iris-virginica','SpeciesType'] = 300;
iris1[iris1.Species == 'Iris-versicolor']
plt.scatter(iris1.PetalLengthCm, iris1.PetalWidthCm, c=iris1.SpeciesType)
plt.gray()

plt.show()
iris2 = iris1[iris1.Species == 'Iris-versicolor']
iris2
plt.scatter(iris2.PetalLengthCm, iris2.PetalWidthCm, c=iris2.SpeciesType)
plt.gray()

plt.show()
iris3 = iris1[iris1.Species == 'Iris-virginica']
plt.scatter(iris3.PetalLengthCm, iris3.PetalWidthCm, c=iris3.SpeciesType)
plt.gray()

plt.show()
