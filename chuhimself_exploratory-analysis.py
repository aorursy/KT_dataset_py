import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs

df = pd.read_csv("../input/loan.csv", low_memory=False)

df["addr_state"].value_counts().plot(kind='bar')
df["annual_inc"].plot.density()
df["funded_amnt"].hist()
plt.title('Funded Amount')

