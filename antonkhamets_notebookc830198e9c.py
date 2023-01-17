import pandas as pd # для работы с данными

import matplotlib.pyplot as plt # для визуализации

import seaborn as sns # для визуализации

students = pd.read_csv("/kaggle/input/student-alcohol-consumption/student-mat.csv")
students.info()
plt.hist(students.age)
sns.boxplot(students.Dalc)
sns.heatmap(students.corr())