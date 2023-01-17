import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk # Sci-Kit Library

import matplotlib.pyplot as plt # Data Visualization

import seaborn as sb # Data Visualization

#to plot inline instead of an external window

%matplotlib inline 
TrainingData = pd.read_csv('../input/train.csv')

TrainingData.shape
TrainingData.head(5)
TrainingData.isnull().values.any()
TrainingData.info()
TrainingData.corr()


from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))