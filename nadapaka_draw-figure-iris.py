import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#path = "C:\\Users\\nadapaka\\Documents\\Bigdata-Analytics\\kaggle\\"#path of data file

#file_name = "Iris.csv"

#file_path_name = path+file_name

#print (file_path_name)
#reading the file

iris = pd.read_csv("../input/Iris.csv")
type(iris)
iris.head()
iris.tail()
iris.info()
iris.drop('Id', axis=1, inplace=True)
fig = iris.plot(kind='scatter', x = 'SepalLengthCm', y ='SepalWidthCm', color = 'orange')
fig = iris [iris.Species == 'Iris-setosa' ].plot(kind='scatter', x = 'SepalLengthCm', y ='SepalWidthCm', color = 'orange', label = 'Setosa')
fig = iris [iris.Species == 'Iris-setosa' ].plot(kind='scatter', x = 'SepalLengthCm', y ='SepalWidthCm', color = 'orange', label = 'Setosa')

iris [iris.Species == 'Iris-versicolor' ].plot(kind='scatter', x = 'SepalLengthCm', y ='SepalWidthCm', color = 'blue', label = 'Versicolor', ax = fig)

iris [iris.Species == 'Iris-virginica' ].plot(kind='scatter', x = 'SepalLengthCm', y ='SepalWidthCm', color = 'green', label = 'Virginica', ax = fig)
