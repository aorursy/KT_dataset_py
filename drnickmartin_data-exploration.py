import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv(r"../input/elnino.csv")

df.columns = [col.strip() for col in df.columns]

df.columns = [col.replace(' ','_') for col in df.columns]

df.head()
df2 = df[df.Sea_Surface_Temp!='.']



plt.figure()

plt.scatter(df2.Date,df2.Sea_Surface_Temp)

plt.show()