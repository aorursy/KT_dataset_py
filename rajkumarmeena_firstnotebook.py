# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

##from subprocess import check_output
##print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print("This is new Notebook which i am going to work on")
allData = pd.read_csv("../input/Iris.csv")
allData
allData.shape
allData.iloc[2:3]
allData.head()
allData["Species"]
allData["Species"].value_counts()
allData.iloc[0:3,0:1] # This is iloc feature of data frame where we can select specific rows and coulmns.
# start range is row and with , it will give coulmns.
allData.get_values()
allData.head()
allData.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm")
import seaborn as sns
sns.jointplot(x="SepalLengthCm",y="SepalWidthCm",data=allData,size=5)
sns.jointplot(x="SepalLengthCm",y="SepalWidthCm",data=allData,size=10)
## This is quite good library for producing graphs. I feel like it is better then pandas basic plot 
# since this shows p rations and parsonr cofficient so we would know the coraletion between variables
# as well. This saves time in coding and analysis as well.
import matplotlib.pyplot as plt
sns.FacetGrid(allData,hue="Species",size=7).map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend()
import tensorflow as tf
import keras as kr
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
import torch
