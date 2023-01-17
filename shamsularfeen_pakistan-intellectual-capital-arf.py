import matplotlib.pyplot as plt

import plotly.plotly as py



import numpy as np 

import pandas as pd



df=pd.read_csv("../input/Pakistan Intellectual Capital - Computer Science - Ver 1.csv", encoding = "ISO-8859-1")

df.head()





# Universities Ranking by Teachers

university="University Currently Teaching"

plt.rcParams['figure.figsize']=(12,50)

df[university].value_counts().sort_values().plot(kind="barh")

plt.show()








