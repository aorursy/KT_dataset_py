import numpy as np

import pandas as pd

import random
months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

days = [i for i in range(1,31)]
print(months)

print(days)
Total_element = 30*12





# Simulation 

count = np.random.rand(30,12)
count.shape
df = pd.DataFrame(count,columns = months,index=days)



df
import matplotlib.pyplot as plt

import seaborn as sns





plt.figure(figsize=(12,30))

sns.heatmap(df, cbar = False)