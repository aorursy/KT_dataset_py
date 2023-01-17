import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
filename='/kaggle/input/top10dataset/top10s.xlsx'
df=pd.read_excel(filename,encoding='ISO-8859-1')
df.head()

#Visualising the variables against popularity using pairplot

plt.figure(figsize=(20,20))
sns.pairplot(df)