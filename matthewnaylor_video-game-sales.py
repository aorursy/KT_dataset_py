import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

file = pd.read_csv ("../input/videogames-sales-dataset/Video_Games_Sales_as_at_22_Dec_2016.csv")
file.head(10)
file.describe()
file.boxplot()
file.hist()
file.info()
file.isnull()
file.isnull().sum()
file.Rating
print(file.isnull().sum())
threshold = len(file)* 0.1

threshold
file.dropna(thresh=threshold, axis=1, inplace=True)
file.shape