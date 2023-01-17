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
# Load data

iris = pd.read_csv('../input/Iris.csv')

iris.shape
iris.head()
plt.scatter(iris['SepalLengthCm'], iris['SepalWidthCm'], sizes=50 * iris['PetalWidthCm'])
iris['Species'].value_counts()
def specie_color(x):

    if x == "Iris-setosa":

        return 0

    elif x == 'Iris-virginica':

        return 1

    return 2
iris['SpeciesNumber'] = iris['Species'].apply(specie_color)
iris.tail()
plt.scatter(

    iris['SepalLengthCm'], iris['SepalWidthCm'], sizes=25 * iris['PetalWidthCm'],

    c=iris['SpeciesNumber'], cmap='viridis', alpha=0.4

)