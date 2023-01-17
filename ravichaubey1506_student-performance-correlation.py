import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

df.head()
plt.figure(dpi=100)

plt.title('Correlation Analysis')

sns.heatmap(df.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()
corr = df.corr()

mask = np.triu(np.ones_like(corr,dtype = bool))



plt.figure(dpi=100)

plt.title('Correlation Analysis')

sns.heatmap(df.corr(),mask=mask,annot=True,lw=1,linecolor='white',cmap='viridis')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()