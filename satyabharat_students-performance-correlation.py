import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df.corr()
plt.figure(figsize=(8,6))
plt.title('Correlation Analysis')
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu');
plt.yticks(rotation=45);