import numpy as np 
import pandas as pd
import seaborn as sn
sn.set(color_codes = True, style="white")
import matplotlib.pyplot as ml
import warnings
warnings.filterwarnings("ignore")
happy = pd.read_csv("../input/2017.csv", sep = ",", header = 0)
print ("It consists of",happy.shape, "rows and columns")
print(happy.head(10))
print(happy.corr())
sn.heatmap(happy.corr())
