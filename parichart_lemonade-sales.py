import numpy as np 
import pandas as pd
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/Lemonade.csv")
df
avg = df['Sales'].mean()
avg
df[df.Sales < avg]
import matplotlib.pylab as plt
df.plot(kind='scatter', x='Sales', y='Temperature', color ='purple')
import seaborn
seaborn.set() 
from itertools import cycle, islice

Monday_avg = df[df['Day']=='Monday']['Sales'].mean()
Tuesday_avg = df[df['Day']=='Tuesday']['Sales'].mean()
Wednesday_avg = df[df['Day']=='Wednesday']['Sales'].mean()
Thursday_avg = df[df['Day']=='Thursday']['Sales'].mean()
Friday_avg = df[df['Day']=='Friday']['Sales'].mean()
Saturday_avg = df[df['Day']=='Saturday']['Sales'].mean()
Sunday_avg = df[df['Day']=='Sunday']['Sales'].mean()

df_class = pd.DataFrame([Monday_avg,Tuesday_avg,Wednesday_avg,Thursday_avg,Friday_avg,Saturday_avg,Sunday_avg])
df_class.index = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df_class.plot(kind='bar', figsize=(6,4), colormap='Paired',title="Average sale on each day")
plt.ylabel('Amount of sale',Fontsize = 10)
