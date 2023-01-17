from sklearn import datasets #library for standard dataset download
import numpy as np #numerical python library to work with numbers
import pandas as pd #pandas to work with data frames
import matplotlib as plt #basic plotting library 
from matplotlib.pyplot import plot #import plot function from matplotlib
data = datasets.load_iris() #load iris dataset from datasets
target_names = data.target_names #store target_names in a variable
target = data.target #store target values in a variable

print("Target Names -> " , target_names)
print("Target Values -> ", target)
plot(target) #plot the target values - Line graph
train = pd.read_csv("../input/kaggle-sumukh/Kaggle.csv") #upload and load csv from kaggle 
train #print the table
train.columns[1]
#print the column names
train.plot(x=train.columns[1], y=train.columns[2], style='o') #create a plot for x-axis and y-axis with scatter style
train[train.columns[1]].plot(kind='hist') #histogram plot
train.hist(column = train.columns[2]) #histogram plotting with grid
train.hist(column = train.columns[2], bins=50) #plotting with bins