# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = pd.read_csv("../input/train.csv", parse_dates=True)
print(X) 
plot = plt.figure().gca(projection='3d')
plot.scatter(X.X, X.Y, X.Dates)
plot.set_xlabel("X")
plot.set_ylabel("Y")
plot.set_zlabel("Z")

plot.show()