import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# path for data

#path = 'C:\\Users\\ravishankar.kv\\Documents\\Python\\'

#file_name = 'Iris.csv'

#path '../input'

file_path_name = path + file_name

print(file_path_name)
# reading the file

iris = pd.read_csv("../input/Iris.csv")
type(iris)
iris.head()
iris.tail()
iris.info()
iris.drop('Id', axis = 1, inplace = True)
fig = iris.plot(kind = 'scatter', x= 'SepalLengthCm', y='SepalWidthCm', color= 'orange')
iris.info()
fig = iris[iris.Species == 'Iris-setosa'].plot(kind = 'scatter', x= 'SepalLengthCm', y='SepalWidthCm', color= 'orange', label= 'Setosa')

iris[iris.Species == 'Iris-versicolor'].plot(kind = 'scatter', x= 'SepalLengthCm', y='SepalWidthCm', color= 'blue', label= 'Versicolor', ax = fig)

iris[iris.Species == 'Iris-virginica'].plot(kind = 'scatter', x= 'SepalLengthCm', y='SepalWidthCm', color= 'green', label= 'Virginica', ax = fig)