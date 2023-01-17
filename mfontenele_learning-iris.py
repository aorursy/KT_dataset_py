# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Iris.csv")
data.columns
print(data['Species'].unique())
colors=['blue' if s == 'Iris-setosa' else 'red' if s == 'Iris-virginica' else 'green' for s in data['Species']]
plt.scatter(data['SepalLengthCm'], data['SepalWidthCm'], c=colors)
plt.scatter(data['PetalLengthCm'], data['PetalWidthCm'], c=colors)
clf = neighbors.KNeighborsClassifier()
clf.fit(data[['PetalLengthCm', 'PetalWidthCm']], data['Species'])

sum(clf.predict(data[['PetalLengthCm', 'PetalWidthCm']]) == data['Species']) / len(data['Species'])

plt.scatter(data['PetalLengthCm'], data['SepalLengthCm'], c=colors)
plt.scatter(data['PetalLengthCm'], data['SepalWidthCm'], c=colors)
plt.scatter(data['PetalWidthCm'], data['SepalWidthCm'], c=colors)
plt.scatter(data['PetalWidthCm'], data['SepalLengthCm'], c=colors)
