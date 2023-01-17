#Handling Imports

import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



csvData = pd.read_csv("../input/spam.csv",encoding='latin-1')

csvData.head()
csvData["v1"].value_counts()